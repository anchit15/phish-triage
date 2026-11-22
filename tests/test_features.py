import pytest

from phish_triage.features import (
    MODEL_FEATURES,
    canonical_host,
    digit_count,
    features_frame,
    has_ip_host,
    host_entropy,
    host_features,
    keyword_count,
    registered_domain,
    url_flags,
    url_length,
)


@pytest.mark.parametrize(
    ("url", "length", "digits", "ip_host"),
    [
        ("http://example.com", 18, 0, 0),
        ("hxxp://example[.]com", 18, 0, 0),
        ("EXAMPLE.COM/a1b2", 23, 2, 0),
        ("HXXPS://192[.]0[.]2[.]1/login", 23, 6, 1),
        ("192.0.2.1/login", 22, 6, 1),
        ("http://[2001:db8::1]:8080/x", 27, 10, 1),
        ("http://[::1", 11, 1, 0),
        ("http://ex²am٣ple.com/9", 22, 1, 0),
    ],
)
def test_url_length_digit_count_and_ip_host(
    url: str, length: int, digits: int, ip_host: int
) -> None:
    assert url_length(url) == length
    assert digit_count(url) == digits
    assert has_ip_host(url) == ip_host


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.Example.com/login?x=1", "example.com"),
        ("hxxp://user@login[.]example[.]test:8443/a", "login.example.test"),
        ("example.com.", "example.com"),
        ("http://[2001:db8::1]/", "2001:db8::1"),
        ("http://[::1", ""),
    ],
)
def test_canonical_host_strips_scheme_www_path_userinfo_and_port(url: str, expected: str) -> None:
    assert canonical_host(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://a.b.example.co.uk/x", "example.co.uk"),
        ("hxxp://secure-login[.]example[.]com", "example.com"),
        ("http://localhost/", "localhost"),
        ("http://192.0.2.1/", "192.0.2.1"),
    ],
)
def test_registered_domain(url: str, expected: str) -> None:
    assert registered_domain(url) == expected


def test_host_features_for_a_combosquatting_host() -> None:
    features = host_features("hxxps://secure-login.paypa1-account[.]example[.]xyz/verify")
    assert features["host_length"] == len("secure-login.paypa1-account.example.xyz")
    assert features["digit_count"] == 1
    assert features["has_ip_host"] == 0
    assert features["hyphen_count"] == 2
    assert features["subdomain_count"] == 2
    assert features["domain_name_length"] == len("example")
    assert features["keyword_count"] == 3
    assert features["is_punycode"] == 0
    assert features["tld"] == "xyz"


def test_host_features_ignore_path_and_www() -> None:
    assert host_features("https://www.example.com/a/b?c=1") == host_features("example.com")


def test_host_features_for_ip_and_punycode_hosts() -> None:
    ip = host_features("http://192.0.2.1/login")
    assert ip["has_ip_host"] == 1
    assert ip["tld"] == "ip"
    assert ip["subdomain_count"] == 0
    assert host_features("http://xn--pypal-4ve.example/")["is_punycode"] == 1


def test_host_entropy() -> None:
    assert host_entropy("") == 0.0
    assert host_entropy("aaaa") == 0.0
    assert host_entropy("abcd") == pytest.approx(2.0)


def test_keyword_count_counts_distinct_keywords() -> None:
    assert keyword_count("login-login.example") == 1
    assert keyword_count("example.com") == 0


def test_url_flags() -> None:
    assert url_flags("hxxp://paypal.com@evil[.]example/") == {
        "url_length": len("http://paypal.com@evil.example/"),
        "has_at_symbol": 1,
        "has_double_slash_redirect": 0,
    }
    assert url_flags("http://example.com/r//http://other.example")["has_double_slash_redirect"] == 1


def test_features_frame_has_model_columns_in_order() -> None:
    frame = features_frame(["example.com", "hxxp://192[.]0[.]2[.]1/"])
    assert list(frame.columns) == MODEL_FEATURES
    assert len(frame) == 2
