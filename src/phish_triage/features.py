"""Lexical and host features computed from the raw URL string only.

The model uses host features only. In PhiUSIIL every legitimate URL is a bare
``https://www.<domain>`` homepage, so the scheme, ``www`` and any path identify the
label perfectly; path-level flags are extracted for analyst evidence, not for training.
"""

import ipaddress
import math
from collections import Counter
from collections.abc import Iterable
from urllib.parse import urlsplit

import pandas as pd
import tldextract

from phish_triage.defang import normalise

# An empty suffix_list_urls makes tldextract use its bundled Public Suffix List
# snapshot, so feature extraction never touches the network.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_ASCII_DIGITS = frozenset("0123456789")
SUSPICIOUS_KEYWORDS = (
    "login",
    "signin",
    "verify",
    "secure",
    "account",
    "update",
    "wallet",
    "support",
    "banking",
    "confirm",
    "password",
    "auth",
)

NUMERIC_FEATURES = [
    "host_length",
    "digit_count",
    "has_ip_host",
    "hyphen_count",
    "subdomain_count",
    "domain_name_length",
    "host_entropy",
    "keyword_count",
    "is_punycode",
]
CATEGORICAL_FEATURES = ["tld"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def url_length(url: str) -> int:
    """Return the number of characters in the URL.

    Args:
        url: The URL to measure, plain or defanged.

    Returns:
        The length of the normalised (refanged, scheme-prefixed) URL in characters.
    """
    return len(normalise(url))


def digit_count(url: str) -> int:
    """Return the number of digit characters in the URL.

    Args:
        url: The URL to scan, plain or defanged.

    Returns:
        How many characters in the URL are ASCII digits 0-9.
    """
    return sum(char in _ASCII_DIGITS for char in normalise(url))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def canonical_host(url: str) -> str:
    """Return the URL's host, lowercased, without a leading ``www.`` or trailing dot.

    Args:
        url: The URL to parse, plain or defanged, with or without a scheme.

    Returns:
        The canonical host, or an empty string if the URL has no parseable host.
    """
    try:
        host = urlsplit(normalise(url)).hostname or ""
    except ValueError:
        return ""
    host = host.rstrip(".")
    return host.removeprefix("www.")


def has_ip_host(url: str) -> int:
    """Return whether the URL's host is an IP address rather than a domain name.

    Args:
        url: The URL to inspect, plain or defanged. Both IPv4 and IPv6 hosts count.

    Returns:
        1 if the host is an IP address, otherwise 0.
    """
    return int(_is_ip(canonical_host(url)))


def registered_domain(url: str) -> str:
    """Return the registered domain (e.g. ``example.co.uk``), or the IP for IP hosts.

    Args:
        url: The URL to parse, plain or defanged.

    Returns:
        The registered domain, the IP address, or the bare host if neither applies.
    """
    host = canonical_host(url)
    if _is_ip(host):
        return host
    return _EXTRACT(host).top_domain_under_public_suffix or host


def host_entropy(host: str) -> float:
    """Return the Shannon entropy of the host's characters, in bits per character.

    Args:
        host: A hostname.

    Returns:
        0.0 for an empty host; higher values mean more random-looking text.
    """
    if not host:
        return 0.0
    total = len(host)
    return -sum(n / total * math.log2(n / total) for n in Counter(host).values())


def keyword_count(host: str) -> int:
    """Return how many distinct suspicious keywords appear in the host.

    Args:
        host: A hostname.

    Returns:
        The number of entries in ``SUSPICIOUS_KEYWORDS`` found in the host.
    """
    return sum(keyword in host for keyword in SUSPICIOUS_KEYWORDS)


def host_features(url: str) -> dict[str, int | float | str]:
    """Compute the model's features from the URL's canonical host.

    Args:
        url: The URL to featurise, plain or defanged.

    Returns:
        A mapping with one entry per name in ``MODEL_FEATURES``.
    """
    host = canonical_host(url)
    is_ip = _is_ip(host)
    parts = _EXTRACT(host)
    subdomains = parts.subdomain.split(".") if parts.subdomain and not is_ip else []
    if is_ip:
        tld = "ip"
    else:
        tld = parts.suffix or "none"
    return {
        "host_length": len(host),
        "digit_count": digit_count(host),
        "has_ip_host": int(is_ip),
        "hyphen_count": host.count("-"),
        "subdomain_count": len(subdomains),
        "domain_name_length": 0 if is_ip else len(parts.domain),
        "host_entropy": host_entropy(host),
        "keyword_count": keyword_count(host),
        "is_punycode": int("xn--" in host),
        "tld": tld,
    }


def has_at_symbol(url: str) -> int:
    """Return 1 if the URL has userinfo (``user@host``), which hides the real host.

    Args:
        url: The URL to inspect, plain or defanged.

    Returns:
        1 if the authority part contains ``@``, otherwise 0.
    """
    try:
        return int("@" in urlsplit(normalise(url)).netloc)
    except ValueError:
        return 0


def has_double_slash_redirect(url: str) -> int:
    """Return 1 if the path contains ``//``, a common sign of an embedded redirect.

    Args:
        url: The URL to inspect, plain or defanged.

    Returns:
        1 if ``//`` appears in the path, otherwise 0.
    """
    try:
        return int("//" in urlsplit(normalise(url)).path)
    except ValueError:
        return 0


def url_flags(url: str) -> dict[str, int]:
    """Compute path-level evidence flags. These are reported, not used by the model.

    Args:
        url: The URL to inspect, plain or defanged.

    Returns:
        A mapping of flag name to value.
    """
    return {
        "url_length": url_length(url),
        "has_at_symbol": has_at_symbol(url),
        "has_double_slash_redirect": has_double_slash_redirect(url),
    }


def features_frame(urls: Iterable[str]) -> pd.DataFrame:
    """Build the model's feature matrix for many URLs.

    Args:
        urls: URLs, plain or defanged.

    Returns:
        A DataFrame with one row per URL and columns ``MODEL_FEATURES``.
    """
    return pd.DataFrame([host_features(url) for url in urls], columns=MODEL_FEATURES)
