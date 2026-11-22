"""Assemble one defanged report that the terminal, JSON and notice outputs share."""

import re
from dataclasses import asdict
from datetime import datetime
from typing import Any

from phish_triage import __version__
from phish_triage.defang import defang
from phish_triage.enrich import Enrichment, HostInfo
from phish_triage.explain import Prediction, Signal
from phish_triage.features import registered_domain, url_flags, url_host

# CDNs and reverse proxies that hide the origin server. General-purpose clouds are
# not listed: there the resolved IP is usually the origin itself.
CDN_PROVIDERS: tuple[tuple[str, frozenset[int], tuple[str, ...]], ...] = (
    ("Cloudflare", frozenset({13335, 209242}), ("cloudflare",)),
    ("Akamai", frozenset({20940, 16625, 16702, 32787}), ("akamai",)),
    ("Fastly", frozenset({54113}), ("fastly",)),
    ("Imperva", frozenset({19551}), ("incapsula", "imperva")),
    ("Sucuri", frozenset({30148}), ("sucuri",)),
    ("CDN77", frozenset({60068}), ("cdn77",)),
    ("Edgio", frozenset({15133}), ("edgecast", "edgio")),
)


def format_probability(probability: float) -> str:
    """Format a probability for display without rounding to a false certainty.

    Args:
        probability: A value between 0 and 1.

    Returns:
        Two decimals, or ``>0.99`` / ``<0.01`` where rounding would show 1.00 or 0.00.
    """
    if probability > 0.995:
        return ">0.99"
    if probability < 0.005:
        return "<0.01"
    return f"{probability:.2f}"


def detect_cdn(host: HostInfo) -> str | None:
    """Return the CDN or reverse-proxy provider in front of the host, if recognised.

    Args:
        host: Enrichment results for the host.

    Returns:
        The provider name, or ``None``.
    """
    names = f"{host.hosting_org or ''} {host.asn_name or ''}".lower()
    for provider, asns, keywords in CDN_PROVIDERS:
        if host.asn in asns or any(keyword in names for keyword in keywords):
            return provider
    return None


def _age_days(iso_timestamp: str | None, now: datetime) -> int | None:
    if not iso_timestamp:
        return None
    try:
        return (now - datetime.fromisoformat(iso_timestamp)).days
    except (TypeError, ValueError):
        return None


def _defang_strings(value: Any, pattern: re.Pattern[str] | None, keep_emails: bool = True) -> Any:
    """Defang every match of the suspect host names and IPs in nested string values.

    Email fields are left alone so the analyst can use them, except inside ``errors``
    mappings, where the values are reasons that may name the suspect host.
    """
    if pattern is None:
        return value
    if isinstance(value, str):
        return pattern.sub(lambda match: defang(match.group(0)), value)
    if isinstance(value, list):
        return [_defang_strings(item, pattern, keep_emails) for item in value]
    if isinstance(value, dict):
        return {
            key: item
            if keep_emails and key.endswith("email")
            else _defang_strings(item, pattern, keep_emails=keep_emails and key != "errors")
            for key, item in value.items()
        }
    return value


def _suspect_pattern(names: list[str]) -> re.Pattern[str] | None:
    # Longest first, so "login.example.com" is defanged whole before "example.com".
    needles = sorted({name for name in names if name and "." in name}, key=len, reverse=True)
    if not needles:
        return None
    return re.compile("|".join(re.escape(needle) for needle in needles), re.IGNORECASE)


def build_report(
    url: str,
    prediction: Prediction,
    signals: list[Signal],
    enrichment: Enrichment,
    now: datetime,
) -> dict[str, Any]:
    """Combine the verdict, SHAP signals, enrichment and contacts into one report.

    Every suspect URL, host and IP in the result is defanged.

    Args:
        url: The suspect URL, plain or defanged.
        prediction: The model verdict.
        signals: The top SHAP signals.
        enrichment: Passive enrichment results.
        now: The report time (timezone-aware, UTC).

    Returns:
        A JSON-serialisable report dictionary.
    """
    cdn = detect_cdn(enrichment.host)
    contacts = []
    if enrichment.host.host_abuse_email:
        contacts.append(
            {"role": "hosting", "email": enrichment.host.host_abuse_email, "via_cdn": bool(cdn)}
        )
    if enrichment.domain.registrar_abuse_email:
        contacts.append(
            {
                "role": "registrar",
                "email": enrichment.domain.registrar_abuse_email,
                "via_cdn": False,
            }
        )

    details = asdict(enrichment)
    details["domain"].pop("rdap_url")
    details["tls"]["age_days"] = _age_days(enrichment.tls.not_before, now)
    flags = url_flags(url)

    report = {
        "tool": {"name": "phish-triage", "version": __version__},
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": defang(url),
        "host": url_host(url),
        "verdict": {
            "label": "phishing" if prediction.is_phishing else "not phishing",
            "probability": round(prediction.probability, 4),
            "threshold": round(prediction.threshold, 4),
            "target_precision": prediction.target_precision,
            "model": prediction.model,
        },
        "signals": [asdict(signal) for signal in signals],
        "enrichment": details,
        "cdn": {"name": cdn} if cdn else None,
        "url_flags": {
            "has_at_symbol": bool(flags["has_at_symbol"]),
            "has_double_slash_redirect": bool(flags["has_double_slash_redirect"]),
        },
        "abuse_contacts": contacts,
    }
    suspects = [url_host(url), registered_domain(url), *enrichment.host.ips]
    return _defang_strings(report, _suspect_pattern(suspects))
