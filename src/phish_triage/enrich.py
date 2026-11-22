"""Passive enrichment: domain RDAP, DNS A record, IP RDAP and ASN, TLS certificate.

Safety: no HTTP request is ever sent to the suspect URL or host. RDAP queries go to
registries found through IANA's bootstrap files, and DNS queries go to the configured
resolver. A TLS handshake to read the certificate is the only direct contact with the
suspect host, and it is skipped in no-touch mode.

Every lookup fails soft: a field that cannot be filled is ``None`` and its reason is
recorded in the section's ``errors`` mapping.
"""

import hashlib
import ipaddress
import json
import os
import socket
import ssl
import time
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import dns.exception
import dns.resolver
import requests
from cryptography import x509

from phish_triage import __version__
from phish_triage.defang import normalise, refang
from phish_triage.features import has_ip_host, registered_domain, url_host

IANA_BOOTSTRAP_URL = "https://data.iana.org/rdap/{kind}.json"
USER_AGENT = f"phish-triage/{__version__} (+https://github.com/anchit15/phish-triage)"
RDAP_TTL_SECONDS = 24 * 3600
BOOTSTRAP_TTL_SECONDS = 7 * 24 * 3600
MAX_ATTEMPTS = 3
MAX_RETRY_AFTER_SECONDS = 30
DEFAULT_TLS_PORT = 443


class EnrichmentError(Exception):
    """A lookup failed; the message is the human-readable reason."""


@dataclass
class DomainInfo:
    """Registration data for the URL's registered domain."""

    registered_domain: str | None = None
    created: str | None = None
    age_days: int | None = None
    registrar: str | None = None
    registrar_abuse_email: str | None = None
    rdap_url: str | None = None
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class HostInfo:
    """Where the host resolves and who runs that network."""

    hostname: str | None = None
    ips: list[str] = field(default_factory=list)
    ip: str | None = None
    asn: int | None = None
    asn_name: str | None = None
    hosting_org: str | None = None
    host_abuse_email: str | None = None
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class TlsInfo:
    """Certificate details read during a TLS handshake."""

    issuer: str | None = None
    not_before: str | None = None
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class Enrichment:
    """All enrichment results for one URL."""

    domain: DomainInfo
    host: HostInfo
    tls: TlsInfo


def get_timeout() -> float:
    """Return the per-call network timeout in seconds from PHISH_TRIAGE_TIMEOUT (default 10)."""
    return float(os.environ.get("PHISH_TRIAGE_TIMEOUT") or 10)


def get_cache_dir() -> Path:
    """Return the cache directory from PHISH_TRIAGE_CACHE_DIR (default ``.cache``)."""
    return Path(os.environ.get("PHISH_TRIAGE_CACHE_DIR") or ".cache")


def no_touch_from_env() -> bool:
    """Return True if PHISH_TRIAGE_NO_TOUCH asks to skip contacting the suspect host."""
    return os.environ.get("PHISH_TRIAGE_NO_TOUCH", "").strip().lower() in {"1", "true", "yes"}


def _note_missing(info: Any, reason: str, names: tuple[str, ...] | None = None) -> None:
    """Record ``reason`` for every empty field (or each of ``names``) without a reason yet."""
    targets = names or tuple(f.name for f in fields(info) if f.name != "errors")
    for name in targets:
        value = getattr(info, name)
        if (value is None or value == []) and name not in info.errors:
            info.errors[name] = reason


# --- HTTP with cache and back-off ---------------------------------------------------


def _cache_path(url: str) -> Path:
    return get_cache_dir() / "rdap" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"


def _read_cache(url: str, ttl: float) -> Any | None:
    try:
        entry = json.loads(_cache_path(url).read_text(encoding="utf-8"))
        if time.time() - entry["fetched_at"] > ttl:
            return None
        return entry["body"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(url: str, body: Any) -> None:
    path = _cache_path(url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"url": url, "fetched_at": time.time(), "body": body}
        path.write_text(json.dumps(entry), encoding="utf-8")
    except OSError:
        # The cache only saves repeat queries; failing to write it must not fail the lookup.
        pass


def _retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after.isdigit():
        return min(int(retry_after), MAX_RETRY_AFTER_SECONDS)
    return float(2 ** (attempt + 1))


def fetch_json(url: str, ttl: float) -> Any:
    """GET a registry JSON document, using the cache and backing off on HTTP 429.

    Only call this with registry URLs (IANA, RDAP servers), never the suspect URL.

    Args:
        url: The registry URL to fetch.
        ttl: How long a cached copy stays valid, in seconds.

    Returns:
        The decoded JSON body.

    Raises:
        EnrichmentError: On timeout, connection failure, HTTP error, rate limiting
            after ``MAX_ATTEMPTS`` tries, or an invalid JSON body.
    """
    cached = _read_cache(url, ttl)
    if cached is not None:
        return cached

    server = urlsplit(url).hostname
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rdap+json, application/json"}
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(url, headers=headers, timeout=get_timeout())
        except requests.Timeout as exc:
            raise EnrichmentError(
                f"timed out after {get_timeout():g}s contacting {server}"
            ) from exc
        except requests.RequestException as exc:
            raise EnrichmentError(f"request to {server} failed ({type(exc).__name__})") from exc

        if response.status_code == 429:
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(_retry_delay(response, attempt))
            continue
        if response.status_code == 404:
            raise EnrichmentError(f"{server} has no record (HTTP 404)")
        if not response.ok:
            raise EnrichmentError(f"{server} returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise EnrichmentError(f"{server} returned invalid JSON") from exc
        _write_cache(url, body)
        return body

    raise EnrichmentError(f"rate limited by {server} (HTTP 429 after {MAX_ATTEMPTS} tries)")


# --- RDAP server discovery (IANA bootstrap, RFC 9224) -------------------------------


def _bootstrap_services(kind: str) -> list:
    data = fetch_json(IANA_BOOTSTRAP_URL.format(kind=kind), BOOTSTRAP_TTL_SECONDS)
    return data.get("services", []) if isinstance(data, dict) else []


def _pick_base_url(urls: list[str]) -> str:
    base = next((u for u in urls if u.startswith("https://")), urls[0])
    return base if base.endswith("/") else base + "/"


def rdap_base_for_domain(domain: str) -> str:
    """Return the RDAP base URL for a domain's TLD from IANA's bootstrap file.

    Args:
        domain: A registered domain, e.g. ``example.com``.

    Returns:
        The base URL, ending in ``/``.

    Raises:
        EnrichmentError: If the bootstrap file cannot be fetched or lists no server.
    """
    tld = domain.rstrip(".").rsplit(".", 1)[-1].lower()
    for entries, urls in _bootstrap_services("dns"):
        if urls and tld in (entry.lower() for entry in entries):
            return _pick_base_url(urls)
    raise EnrichmentError(f"no RDAP service listed for .{tld}")


def rdap_base_for_ip(ip: str) -> str:
    """Return the RDAP base URL for the most specific IANA range containing ``ip``.

    Args:
        ip: An IPv4 or IPv6 address.

    Returns:
        The base URL, ending in ``/``.

    Raises:
        EnrichmentError: If the bootstrap file cannot be fetched or no range matches.
    """
    address = ipaddress.ip_address(ip)
    kind = "ipv4" if address.version == 4 else "ipv6"
    best: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, list[str]] | None = None
    for prefixes, urls in _bootstrap_services(kind):
        for prefix in prefixes:
            network = ipaddress.ip_network(prefix, strict=False)
            if (
                urls
                and address in network
                and (best is None or network.prefixlen > best[0].prefixlen)
            ):
                best = (network, urls)
    if best is None:
        raise EnrichmentError(f"no RDAP service listed for {ip}")
    return _pick_base_url(best[1])


# --- RDAP response parsing ----------------------------------------------------------


def _vcard_value(entity: dict | None, key: str) -> str | None:
    """Return the first value of ``key`` (e.g. ``fn``, ``email``) from an entity's jCard."""
    if not entity:
        return None
    vcard = entity.get("vcardArray")
    if not (isinstance(vcard, list) and len(vcard) == 2 and isinstance(vcard[1], list)):
        return None
    for prop in vcard[1]:
        if isinstance(prop, list) and len(prop) >= 4 and prop[0] == key:
            value = str(prop[3]).strip().removeprefix("mailto:")
            return value or None
    return None


def _find_entity(entities: list | None, role: str) -> dict | None:
    """Return the first entity with ``role``, checking each level before nested ones."""
    entities = [e for e in entities or [] if isinstance(e, dict)]
    for entity in entities:
        if role in entity.get("roles", []):
            return entity
    for entity in entities:
        found = _find_entity(entity.get("entities"), role)
        if found:
            return found
    return None


def _parse_rdap_date(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_domain_rdap(data: dict, now: datetime) -> dict[str, Any]:
    """Extract creation date, age, registrar and registrar abuse email from domain RDAP.

    Args:
        data: A decoded RDAP domain response.
        now: The current time (timezone-aware), used to compute the age.

    Returns:
        A mapping with ``created``, ``age_days``, ``registrar`` and
        ``registrar_abuse_email``; values missing from the response are ``None``.
    """
    created = None
    for event in data.get("events", []):
        if isinstance(event, dict) and event.get("eventAction") == "registration":
            created = _parse_rdap_date(event.get("eventDate"))
            break

    entities = data.get("entities")
    registrar = _find_entity(entities, "registrar")
    # ICANN's RDAP profile nests the registrar's abuse contact inside the registrar entity.
    abuse = _find_entity(registrar.get("entities"), "abuse") if registrar else None
    abuse = abuse or _find_entity(entities, "abuse")

    return {
        "created": created.isoformat() if created else None,
        "age_days": (now - created).days if created else None,
        "registrar": _vcard_value(registrar, "fn"),
        "registrar_abuse_email": _vcard_value(abuse, "email"),
    }


def parse_ip_rdap(data: dict) -> dict[str, str | None]:
    """Extract the hosting organisation and abuse email from IP network RDAP.

    Args:
        data: A decoded RDAP ip network response.

    Returns:
        A mapping with ``hosting_org`` (registrant name, else network name) and
        ``host_abuse_email``; missing values are ``None``.
    """
    entities = data.get("entities")
    org = _vcard_value(_find_entity(entities, "registrant"), "fn") or data.get("name")
    return {
        "hosting_org": org,
        "host_abuse_email": _vcard_value(_find_entity(entities, "abuse"), "email"),
    }


# --- DNS and ASN --------------------------------------------------------------------


def _resolve(name: str, rdtype: str) -> dns.resolver.Answer:
    try:
        return dns.resolver.resolve(name, rdtype, lifetime=get_timeout())
    except dns.resolver.NXDOMAIN as exc:
        raise EnrichmentError(f"{name} does not resolve (NXDOMAIN)") from exc
    except dns.resolver.NoAnswer as exc:
        raise EnrichmentError(f"no {rdtype} record for {name}") from exc
    except dns.exception.Timeout as exc:
        raise EnrichmentError(f"DNS lookup timed out after {get_timeout():g}s") from exc
    except dns.exception.DNSException as exc:
        raise EnrichmentError(f"DNS lookup failed ({type(exc).__name__})") from exc


def resolve_a(host: str) -> list[str]:
    """Return the host's IPv4 addresses in the order the resolver gave them.

    Raises:
        EnrichmentError: If the name does not exist, has no A record, or DNS fails.
    """
    return [record.address for record in _resolve(host, "A")]


def _txt(name: str) -> str:
    answer = _resolve(name, "TXT")
    return b"".join(next(iter(answer)).strings).decode("utf-8", errors="replace")


def lookup_asn(ip: str) -> tuple[int, str | None]:
    """Return the origin ASN and AS name for an IP using Team Cymru's DNS service.

    Args:
        ip: An IPv4 or IPv6 address.

    Returns:
        The ASN and its name (``None`` if the name lookup fails).

    Raises:
        EnrichmentError: If the origin lookup fails or its answer is malformed.
    """
    address = ipaddress.ip_address(ip)
    reversed_ip = address.reverse_pointer.rsplit(".", 2)[0]
    zone = "origin.asn.cymru.com" if address.version == 4 else "origin6.asn.cymru.com"
    try:
        # Answer format: "64500 | 192.0.2.0/24 | ZZ | registry | date"; the first
        # field can list several ASNs, of which the first is used.
        origin = _txt(f"{reversed_ip}.{zone}")
        asn = int(origin.split("|")[0].split()[0])
    except EnrichmentError as exc:
        raise EnrichmentError(f"ASN lookup failed: {exc}") from exc
    except (ValueError, IndexError) as exc:
        raise EnrichmentError("ASN lookup returned an unexpected answer") from exc

    try:
        # Answer format: "64500 | ZZ | registry | date | AS-NAME, ZZ"
        asn_name = _txt(f"AS{asn}.asn.cymru.com").split("|")[-1].strip() or None
    except EnrichmentError:
        asn_name = None
    return asn, asn_name


# --- TLS ----------------------------------------------------------------------------


def read_certificate(host: str, port: int = DEFAULT_TLS_PORT) -> tuple[str, str]:
    """Complete a TLS handshake, read the server certificate, and close.

    Verification is off so self-signed, expired or mismatched certificates can still
    be read; nothing is trusted or sent beyond the handshake.

    Args:
        host: Hostname or IP to connect to (also sent as SNI).
        port: TCP port.

    Returns:
        The issuer (RFC 4514 string) and not-before time (ISO 8601, UTC).

    Raises:
        EnrichmentError: If the connection, handshake or certificate parsing fails.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=get_timeout()) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
    except TimeoutError as exc:
        raise EnrichmentError(f"TLS handshake timed out after {get_timeout():g}s") from exc
    except OSError as exc:
        raise EnrichmentError(f"TLS handshake failed ({type(exc).__name__})") from exc

    if not der:
        raise EnrichmentError("server sent no certificate")
    try:
        cert = x509.load_der_x509_certificate(der)
    except ValueError as exc:
        raise EnrichmentError("certificate could not be parsed") from exc
    return cert.issuer.rfc4514_string(), cert.not_valid_before_utc.isoformat()


# --- Orchestration ------------------------------------------------------------------


def enrich_domain(url: str, now: datetime) -> DomainInfo:
    """Look up the registered domain's RDAP record.

    Args:
        url: The suspect URL, plain or defanged.
        now: The current time, used for the domain age.

    Returns:
        Domain details, with reasons for any empty field.
    """
    info = DomainInfo()
    if not url_host(url):
        _note_missing(info, "URL has no host")
        return info
    if has_ip_host(url):
        _note_missing(info, "host is an IP address; no domain to look up")
        return info

    info.registered_domain = registered_domain(url)
    try:
        info.rdap_url = (
            f"{rdap_base_for_domain(info.registered_domain)}domain/{info.registered_domain}"
        )
        data = fetch_json(info.rdap_url, RDAP_TTL_SECONDS)
    except EnrichmentError as exc:
        _note_missing(info, str(exc))
        return info

    for name, value in parse_domain_rdap(data, now).items():
        setattr(info, name, value)
    _note_missing(info, "not present in RDAP response")
    return info


def enrich_host(url: str) -> HostInfo:
    """Resolve the host, then look up the first IP's ASN and network RDAP record.

    Args:
        url: The suspect URL, plain or defanged.

    Returns:
        Host details, with reasons for any empty field.
    """
    host = url_host(url)
    info = HostInfo(hostname=host or None)
    if not host:
        _note_missing(info, "URL has no host")
        return info

    try:
        info.ips = [host] if has_ip_host(url) else resolve_a(host)
    except EnrichmentError as exc:
        _note_missing(info, str(exc))
        return info
    info.ip = info.ips[0]

    try:
        info.asn, info.asn_name = lookup_asn(info.ip)
    except EnrichmentError as exc:
        _note_missing(info, str(exc), ("asn", "asn_name"))

    try:
        data = fetch_json(f"{rdap_base_for_ip(info.ip)}ip/{info.ip}", RDAP_TTL_SECONDS)
    except EnrichmentError as exc:
        _note_missing(info, str(exc), ("hosting_org", "host_abuse_email"))
    else:
        for name, value in parse_ip_rdap(data).items():
            setattr(info, name, value)

    _note_missing(info, "not present in lookup response")
    return info


def enrich_tls(url: str, touch: bool) -> TlsInfo:
    """Read the certificate of an https (or scheme-less) URL's host, unless no-touch.

    Args:
        url: The suspect URL, plain or defanged.
        touch: False to skip the handshake, so the suspect host is never contacted.

    Returns:
        Certificate details, with reasons for any empty field.
    """
    info = TlsInfo()
    if not touch:
        _note_missing(info, "skipped: no-touch mode")
        return info

    refanged = refang(url)
    scheme = refanged.split("://", 1)[0].lower() if "://" in refanged else ""
    if scheme not in ("", "https"):
        _note_missing(info, f"skipped: URL uses {scheme}, not https")
        return info

    host = url_host(url)
    if not host:
        _note_missing(info, "URL has no host")
        return info
    try:
        port = urlsplit(normalise(url)).port or DEFAULT_TLS_PORT
        info.issuer, info.not_before = read_certificate(host, port)
    except ValueError:
        _note_missing(info, "URL has an invalid port")
    except EnrichmentError as exc:
        _note_missing(info, str(exc))
    return info


def enrich(url: str, touch: bool | None = None, now: datetime | None = None) -> Enrichment:
    """Collect domain, host and TLS evidence for a suspect URL without visiting it.

    Args:
        url: The suspect URL, plain or defanged.
        touch: Whether the TLS handshake may contact the host. ``None`` reads
            PHISH_TRIAGE_NO_TOUCH.
        now: The current time; defaults to now in UTC.

    Returns:
        The combined enrichment result. Never raises for lookup failures.
    """
    if touch is None:
        touch = not no_touch_from_env()
    now = now or datetime.now(UTC)
    return Enrichment(
        domain=enrich_domain(url, now),
        host=enrich_host(url),
        tls=enrich_tls(url, touch),
    )
