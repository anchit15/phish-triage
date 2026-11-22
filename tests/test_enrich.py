import json
import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import dns.exception
import dns.resolver
import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from phish_triage import enrich

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 22, 0, 0, tzinfo=UTC)
SUSPECT_URL = "hxxps://login[.]example[.]com/verify"

IANA_DNS = "https://data.iana.org/rdap/dns.json"
IANA_IPV4 = "https://data.iana.org/rdap/ipv4.json"
IANA_IPV6 = "https://data.iana.org/rdap/ipv6.json"
DOMAIN_RDAP = "https://rdap.registry.example/com/domain/example.com"
IP_RDAP = "https://rdap.rir-b.example/ip/192.0.2.10"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status: int = 200, body: object = None, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> object:
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeHttp:
    """Serves canned responses by URL and records every requested URL."""

    def __init__(self, routes: dict[str, object]):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        assert "timeout" in kwargs, "every HTTP call must set a timeout"
        self.calls.append(url)
        route = self.routes.get(url, FakeResponse(404))
        if isinstance(route, list):
            route = route.pop(0)
        if isinstance(route, Exception):
            raise route
        return route if isinstance(route, FakeResponse) else FakeResponse(200, route)


class FakeDns:
    """Answers (name, rdtype) queries from a table; anything else is NXDOMAIN."""

    def __init__(self, answers: dict[tuple[str, str], object]):
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def resolve(self, name: str, rdtype: str, lifetime: float) -> list:
        self.calls.append((name, rdtype))
        answer = self.answers.get((name, rdtype), dns.resolver.NXDOMAIN())
        if isinstance(answer, Exception):
            raise answer
        if rdtype == "A":
            return [SimpleNamespace(address=ip) for ip in answer]
        return [SimpleNamespace(strings=(answer.encode(),))]


STANDARD_ROUTES = {
    IANA_DNS: load("iana_dns.json"),
    IANA_IPV4: load("iana_ipv4.json"),
    IANA_IPV6: load("iana_ipv6.json"),
    DOMAIN_RDAP: load("rdap_domain.json"),
    IP_RDAP: load("rdap_ip_nested.json"),
}
STANDARD_DNS = {
    ("login.example.com", "A"): ["192.0.2.10", "192.0.2.11"],
    (
        "10.2.0.192.origin.asn.cymru.com",
        "TXT",
    ): "64500 64501 | 192.0.2.0/24 | ZZ | test | 2020-01-01",
    ("AS64500.asn.cymru.com", "TXT"): "64500 | ZZ | test | 2020-01-01 | EXAMPLE-HOSTING, ZZ",
}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHISH_TRIAGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PHISH_TRIAGE_NO_TOUCH", raising=False)
    monkeypatch.delenv("PHISH_TRIAGE_TIMEOUT", raising=False)
    monkeypatch.setattr(enrich.time, "sleep", lambda seconds: None)


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
    fake = FakeHttp(dict(STANDARD_ROUTES))
    monkeypatch.setattr(enrich.requests, "get", fake.get)
    return fake


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> FakeDns:
    fake = FakeDns(dict(STANDARD_DNS))
    monkeypatch.setattr(enrich.dns.resolver, "resolve", fake.resolve)
    return fake


def self_signed_der(common_name: str, not_before: datetime) -> bytes:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_before + timedelta(days=90))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)


class FakeSocket:
    def __enter__(self) -> "FakeSocket":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeTls(FakeSocket):
    def __init__(self, der: bytes):
        self.der = der

    def getpeercert(self, binary_form: bool = False) -> bytes:
        assert binary_form, "the raw certificate must be requested"
        return self.der


@pytest.fixture
def fake_tls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake the TCP connection and handshake, recording what was contacted."""
    record: dict = {"connections": [], "der": self_signed_der("login.example.com", NOW)}

    def create_connection(address: tuple[str, int], timeout: float) -> FakeSocket:
        record["connections"].append((address, timeout))
        return FakeSocket()

    def wrap_socket(context: ssl.SSLContext, sock: object, server_hostname: str) -> FakeTls:
        record["verify_mode"] = context.verify_mode
        record["check_hostname"] = context.check_hostname
        record["server_hostname"] = server_hostname
        return FakeTls(record["der"])

    monkeypatch.setattr(enrich.socket, "create_connection", create_connection)
    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", wrap_socket)
    return record


# --- RDAP parsing -------------------------------------------------------------------


def test_parse_domain_rdap_reads_registration_registrar_and_nested_abuse() -> None:
    parsed = enrich.parse_domain_rdap(load("rdap_domain.json"), NOW)

    assert parsed == {
        "created": "2026-11-20T08:14:00+00:00",
        "age_days": 1,
        "registrar": "Example Registrar LLC",
        "registrar_abuse_email": "abuse@registrar.example",
    }


def test_parse_domain_rdap_returns_none_for_missing_data() -> None:
    parsed = enrich.parse_domain_rdap({"events": [], "entities": []}, NOW)

    assert set(parsed.values()) == {None}


def test_parse_ip_rdap_finds_abuse_nested_in_registrant() -> None:
    assert enrich.parse_ip_rdap(load("rdap_ip_nested.json")) == {
        "hosting_org": "Example Hosting Ltd",
        "host_abuse_email": "abuse@hosting.example",
    }


def test_parse_ip_rdap_uses_top_level_abuse_and_network_name_fallback() -> None:
    assert enrich.parse_ip_rdap(load("rdap_ip_toplevel.json")) == {
        "hosting_org": "EXAMPLE-CLOUD-NET",
        "host_abuse_email": "abuse@cloud.example",
    }


# --- Bootstrap ----------------------------------------------------------------------


def test_rdap_base_for_domain_matches_tld_case_insensitively(http: FakeHttp) -> None:
    assert enrich.rdap_base_for_domain("Example.COM") == "https://rdap.registry.example/com/"


def test_rdap_base_for_domain_prefers_https_and_adds_trailing_slash(http: FakeHttp) -> None:
    assert enrich.rdap_base_for_domain("shop.xyz") == "https://rdap.xyz-registry.example/"


def test_rdap_base_for_domain_without_service_raises(http: FakeHttp) -> None:
    with pytest.raises(enrich.EnrichmentError, match="no RDAP service listed for .zz"):
        enrich.rdap_base_for_domain("example.zz")


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("192.0.2.10", "https://rdap.rir-b.example/"),
        ("192.0.3.1", "https://rdap.rir-a.example/"),
        ("2001:db8::1", "https://rdap.rir-b.example/"),
    ],
)
def test_rdap_base_for_ip_uses_most_specific_range(http: FakeHttp, ip: str, expected: str) -> None:
    assert enrich.rdap_base_for_ip(ip) == expected


def test_rdap_base_for_unlisted_ip_raises(http: FakeHttp) -> None:
    with pytest.raises(enrich.EnrichmentError, match="no RDAP service"):
        enrich.rdap_base_for_ip("203.0.113.5")


# --- HTTP: cache, back-off, failures ------------------------------------------------


def test_fetch_json_uses_cache_on_second_call(http: FakeHttp) -> None:
    first = enrich.fetch_json(DOMAIN_RDAP, ttl=60)
    second = enrich.fetch_json(DOMAIN_RDAP, ttl=60)

    assert first == second
    assert http.calls == [DOMAIN_RDAP]


def test_fetch_json_refetches_expired_cache(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    enrich.fetch_json(DOMAIN_RDAP, ttl=60)
    later = enrich.time.time() + 120
    monkeypatch.setattr(enrich.time, "time", lambda: later)

    enrich.fetch_json(DOMAIN_RDAP, ttl=60)

    assert http.calls == [DOMAIN_RDAP, DOMAIN_RDAP]


def test_fetch_json_backs_off_on_429_using_retry_after(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(enrich.time, "sleep", sleeps.append)
    http.routes[DOMAIN_RDAP] = [
        FakeResponse(429, headers={"Retry-After": "5"}),
        FakeResponse(429, headers={"Retry-After": "120"}),
        FakeResponse(200, {"ok": True}),
    ]

    assert enrich.fetch_json(DOMAIN_RDAP, ttl=60) == {"ok": True}
    assert sleeps == [5, 30]


def test_fetch_json_gives_up_after_repeated_429(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(enrich.time, "sleep", sleeps.append)
    http.routes[DOMAIN_RDAP] = [FakeResponse(429) for _ in range(3)]

    with pytest.raises(enrich.EnrichmentError, match="rate limited by rdap.registry.example"):
        enrich.fetch_json(DOMAIN_RDAP, ttl=60)
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    ("route", "message"),
    [
        (requests.Timeout(), "timed out after 10s contacting rdap.registry.example"),
        (requests.ConnectionError(), "request to rdap.registry.example failed"),
        (FakeResponse(404), "has no record"),
        (FakeResponse(500), "returned HTTP 500"),
        (FakeResponse(200, None), "invalid JSON"),
    ],
)
def test_fetch_json_turns_failures_into_reasons(
    http: FakeHttp, route: object, message: str
) -> None:
    http.routes[DOMAIN_RDAP] = route

    with pytest.raises(enrich.EnrichmentError, match=message):
        enrich.fetch_json(DOMAIN_RDAP, ttl=60)


# --- DNS and ASN --------------------------------------------------------------------


def test_resolve_a_keeps_resolver_order(fake_dns: FakeDns) -> None:
    assert enrich.resolve_a("login.example.com") == ["192.0.2.10", "192.0.2.11"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (dns.resolver.NXDOMAIN(), "does not resolve"),
        (dns.resolver.NoAnswer(), "no A record"),
        (dns.exception.Timeout(), "timed out"),
        (dns.resolver.NoNameservers(), "DNS lookup failed"),
    ],
)
def test_resolve_a_failures_become_reasons(
    fake_dns: FakeDns, error: Exception, message: str
) -> None:
    fake_dns.answers[("gone.example.com", "A")] = error

    with pytest.raises(enrich.EnrichmentError, match=message):
        enrich.resolve_a("gone.example.com")


def test_lookup_asn_parses_cymru_answers(fake_dns: FakeDns) -> None:
    assert enrich.lookup_asn("192.0.2.10") == (64500, "EXAMPLE-HOSTING, ZZ")


def test_lookup_asn_without_name_still_returns_number(fake_dns: FakeDns) -> None:
    del fake_dns.answers[("AS64500.asn.cymru.com", "TXT")]

    assert enrich.lookup_asn("192.0.2.10") == (64500, None)


def test_lookup_asn_uses_ipv6_zone(fake_dns: FakeDns) -> None:
    with pytest.raises(enrich.EnrichmentError, match="ASN lookup failed"):
        enrich.lookup_asn("2001:db8::1")

    assert fake_dns.calls[0][0].endswith(".8.b.d.0.1.0.0.2.origin6.asn.cymru.com")


def test_lookup_asn_rejects_malformed_answer(fake_dns: FakeDns) -> None:
    fake_dns.answers[("10.2.0.192.origin.asn.cymru.com", "TXT")] = "garbage"

    with pytest.raises(enrich.EnrichmentError, match="unexpected answer"):
        enrich.lookup_asn("192.0.2.10")


# --- TLS ----------------------------------------------------------------------------


def test_read_certificate_reads_self_signed_cert_without_verifying(fake_tls: dict) -> None:
    issuer, not_before = enrich.read_certificate("login.example.com", 443)

    assert issuer == "CN=login.example.com"
    assert not_before == "2026-11-22T00:00:00+00:00"
    assert fake_tls["verify_mode"] == ssl.CERT_NONE
    assert fake_tls["check_hostname"] is False
    assert fake_tls["server_hostname"] == "login.example.com"
    assert fake_tls["connections"] == [(("login.example.com", 443), 10.0)]


def test_read_certificate_handshake_failure_becomes_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(address: tuple[str, int], timeout: float) -> None:
        raise ConnectionRefusedError()

    monkeypatch.setattr(enrich.socket, "create_connection", refused)

    with pytest.raises(enrich.EnrichmentError, match="TLS handshake failed"):
        enrich.read_certificate("login.example.com")


def test_read_certificate_timeout_becomes_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(address: tuple[str, int], timeout: float) -> None:
        raise TimeoutError()

    monkeypatch.setattr(enrich.socket, "create_connection", slow)

    with pytest.raises(enrich.EnrichmentError, match="timed out"):
        enrich.read_certificate("login.example.com")


def test_read_certificate_unparseable_cert_becomes_reason(fake_tls: dict) -> None:
    fake_tls["der"] = b"not a certificate"

    with pytest.raises(enrich.EnrichmentError, match="could not be parsed"):
        enrich.read_certificate("login.example.com")


def test_enrich_tls_uses_explicit_https_port(fake_tls: dict) -> None:
    enrich.enrich_tls("hxxps://login[.]example[.]com:8443/x", touch=True)

    assert fake_tls["connections"][0][0] == ("login.example.com", 8443)


def test_enrich_tls_skips_http_urls(fake_tls: dict) -> None:
    info = enrich.enrich_tls("hxxp://login[.]example[.]com/", touch=True)

    assert fake_tls["connections"] == []
    assert info.errors["issuer"] == "skipped: URL uses http, not https"


def test_enrich_tls_no_touch_never_opens_a_connection(fake_tls: dict) -> None:
    info = enrich.enrich_tls(SUSPECT_URL, touch=False)

    assert fake_tls["connections"] == []
    assert info.issuer is None
    assert info.errors == {
        "issuer": "skipped: no-touch mode",
        "not_before": "skipped: no-touch mode",
    }


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_no_touch_env_var_skips_handshake(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("PHISH_TRIAGE_NO_TOUCH", value)

    result = enrich.enrich(SUSPECT_URL, now=NOW)

    assert fake_tls["connections"] == []
    assert result.tls.errors["issuer"] == "skipped: no-touch mode"


# --- End to end ---------------------------------------------------------------------


def test_enrich_collects_all_evidence(http: FakeHttp, fake_dns: FakeDns, fake_tls: dict) -> None:
    result = enrich.enrich(SUSPECT_URL, touch=True, now=NOW)

    assert result.domain == enrich.DomainInfo(
        registered_domain="example.com",
        created="2026-11-20T08:14:00+00:00",
        age_days=1,
        registrar="Example Registrar LLC",
        registrar_abuse_email="abuse@registrar.example",
        rdap_url=DOMAIN_RDAP,
        errors={},
    )
    assert result.host == enrich.HostInfo(
        hostname="login.example.com",
        ips=["192.0.2.10", "192.0.2.11"],
        ip="192.0.2.10",
        asn=64500,
        asn_name="EXAMPLE-HOSTING, ZZ",
        hosting_org="Example Hosting Ltd",
        host_abuse_email="abuse@hosting.example",
        errors={},
    )
    assert result.tls.issuer == "CN=login.example.com"
    assert result.tls.errors == {}


def test_enrich_never_sends_http_to_the_suspect_host(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict
) -> None:
    enrich.enrich(SUSPECT_URL, touch=True, now=NOW)

    contacted = {urlsplit(url).hostname for url in http.calls}
    assert contacted == {"data.iana.org", "rdap.registry.example", "rdap.rir-b.example"}
    assert "login.example.com" not in contacted


def test_enrich_ip_host_skips_dns_and_domain_rdap(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict
) -> None:
    http.routes["https://rdap.rir-b.example/ip/198.51.100.7"] = load("rdap_ip_toplevel.json")

    result = enrich.enrich("hxxps://198[.]51[.]100[.]7/login", touch=False, now=NOW)

    assert result.domain.registered_domain is None
    assert result.domain.errors["registrar"] == "host is an IP address; no domain to look up"
    assert ("198.51.100.7", "A") not in fake_dns.calls
    assert result.host.ips == ["198.51.100.7"]
    assert result.host.hosting_org == "EXAMPLE-CLOUD-NET"
    assert result.host.host_abuse_email == "abuse@cloud.example"
    assert result.host.errors["asn"].startswith("ASN lookup failed")


def test_enrich_unresolvable_domain_gives_null_fields_with_reasons(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict
) -> None:
    result = enrich.enrich("hxxps://gone[.]example[.]com/", touch=False, now=NOW)

    assert result.host.ips == []
    assert result.host.ip is None
    assert result.host.errors["host_abuse_email"] == "gone.example.com does not resolve (NXDOMAIN)"
    assert result.domain.registrar == "Example Registrar LLC"


def test_enrich_missing_abuse_contact_is_null_with_reason(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict
) -> None:
    body = load("rdap_domain.json")
    body["entities"][0].pop("entities")
    http.routes[DOMAIN_RDAP] = body

    result = enrich.enrich(SUSPECT_URL, touch=False, now=NOW)

    assert result.domain.registrar_abuse_email is None
    assert result.domain.errors == {"registrar_abuse_email": "not present in RDAP response"}


def test_enrich_never_raises_when_every_lookup_fails(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict
) -> None:
    http.routes = {url: requests.ConnectionError() for url in STANDARD_ROUTES}
    fake_dns.answers = {key: dns.exception.Timeout() for key in STANDARD_DNS}

    result = enrich.enrich(SUSPECT_URL, touch=False, now=NOW)

    assert result.domain.errors["registrar"].startswith("request to data.iana.org failed")
    assert result.host.errors["ip"].startswith("DNS lookup timed out")


def test_enrich_url_without_host() -> None:
    result = enrich.enrich("hxxp://", touch=False, now=NOW)

    assert result.domain.errors["registrar"] == "URL has no host"
    assert result.host.errors["ip"] == "URL has no host"


def test_enrich_skips_tls_when_host_does_not_resolve(
    http: FakeHttp, fake_dns: FakeDns, fake_tls: dict
) -> None:
    result = enrich.enrich("hxxps://gone[.]example[.]com/", touch=True, now=NOW)

    assert fake_tls["connections"] == []
    assert result.tls.errors["issuer"] == "skipped: host did not resolve"


def test_read_certificate_name_resolution_failure_becomes_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unresolvable(address: tuple[str, int], timeout: float) -> None:
        raise enrich.socket.gaierror()

    monkeypatch.setattr(enrich.socket, "create_connection", unresolvable)

    with pytest.raises(enrich.EnrichmentError, match="host name could not be resolved"):
        enrich.read_certificate("login.example.com")
