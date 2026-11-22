import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import SUSPECT_HOST, SUSPECT_URL, make_enrichment

from phish_triage.enrich import HostInfo
from phish_triage.explain import load_bundle, predict, top_signals
from phish_triage.report import build_report, detect_cdn, format_probability

NOW = datetime(2026, 11, 22, 14, 3, tzinfo=UTC)


def report_for(model_path: Path, **enrichment_options: object) -> dict:
    bundle = load_bundle(model_path)
    return build_report(
        SUSPECT_URL,
        predict(bundle, SUSPECT_URL),
        top_signals(bundle, SUSPECT_URL),
        make_enrichment(**enrichment_options),
        NOW,
    )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (HostInfo(asn=13335), "Cloudflare"),
        (HostInfo(asn=64500, hosting_org="Akamai Technologies, Inc."), "Akamai"),
        (HostInfo(asn=64500, asn_name="FASTLY - Fastly, Inc., US"), "Fastly"),
        (HostInfo(asn=64500, hosting_org="Example Hosting Ltd"), None),
        (HostInfo(), None),
    ],
)
def test_detect_cdn(host: HostInfo, expected: str | None) -> None:
    assert detect_cdn(host) == expected


def test_report_defangs_every_suspect_host_and_ip(model_path: Path) -> None:
    report = report_for(model_path)
    text = json.dumps(report)

    for fanged in (SUSPECT_HOST, "example.xyz", "192.0.2.10", "192.0.2.11", "https://"):
        assert fanged not in text
    assert report["url"] == "hxxps://secure-login[.]account-verify[.]example[.]xyz/verify"
    assert report["host"] == "secure-login[.]account-verify[.]example[.]xyz"
    assert report["enrichment"]["host"]["ips"] == ["192[.]0[.]2[.]10", "192[.]0[.]2[.]11"]
    assert (
        report["enrichment"]["tls"]["issuer"] == "CN=secure-login[.]account-verify[.]example[.]xyz"
    )


def test_report_keeps_abuse_emails_usable(model_path: Path) -> None:
    report = report_for(model_path)

    assert report["abuse_contacts"] == [
        {"role": "hosting", "email": "abuse@hosting.example", "via_cdn": False},
        {"role": "registrar", "email": "abuse@registrar.example", "via_cdn": False},
    ]


def test_report_defangs_hosts_inside_error_reasons(model_path: Path) -> None:
    bundle = load_bundle(model_path)
    enrichment = make_enrichment()
    reason = f"{SUSPECT_HOST} does not resolve (NXDOMAIN)"
    enrichment.host.errors["asn"] = reason
    enrichment.host.errors["host_abuse_email"] = reason

    report = build_report(SUSPECT_URL, predict(bundle, SUSPECT_URL), [], enrichment, NOW)

    errors = report["enrichment"]["host"]["errors"]
    assert errors["asn"].startswith("secure-login[.]account-verify")
    assert errors["host_abuse_email"].startswith("secure-login[.]account-verify")


def test_report_marks_cdn_contact(model_path: Path) -> None:
    report = report_for(model_path, cdn=True)

    assert report["cdn"] == {"name": "Cloudflare"}
    assert report["abuse_contacts"][0]["via_cdn"] is True


def test_report_shape(model_path: Path) -> None:
    report = report_for(model_path)

    assert report["generated_at"] == "2026-11-22T14:03:00Z"
    assert report["verdict"]["label"] == "phishing"
    assert report["verdict"]["threshold"] == 0.5
    assert len(report["signals"]) == 5
    assert report["url_flags"] == {"has_at_symbol": False, "has_double_slash_redirect": False}
    assert report["enrichment"]["tls"]["age_days"] == 1
    assert "rdap_url" not in report["enrichment"]["domain"]


@pytest.mark.parametrize(
    ("probability", "shown"),
    [
        (0.9999, ">0.99"),
        (0.9951, ">0.99"),
        (0.994, "0.99"),
        (0.5, "0.50"),
        (0.006, "0.01"),
        (0.0049, "<0.01"),
        (0.0, "<0.01"),
    ],
)
def test_format_probability(probability: float, shown: str) -> None:
    assert format_probability(probability) == shown
