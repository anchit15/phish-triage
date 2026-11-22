from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import SUSPECT_HOST, SUSPECT_URL, make_enrichment

from phish_triage.explain import Prediction, Signal, load_bundle, top_signals
from phish_triage.notice import render_notice
from phish_triage.report import build_report

NOW = datetime(2026, 11, 22, 14, 3, tzinfo=UTC)


def notice_for(
    model_path: Path, *, is_phishing: bool = True, url: str = SUSPECT_URL, **options: object
) -> str:
    bundle = load_bundle(model_path)
    prediction = Prediction(
        probability=0.93 if is_phishing else 0.12,
        threshold=0.7982,
        is_phishing=is_phishing,
        model="logistic_regression",
        target_precision=0.98,
    )
    report = build_report(
        url, prediction, top_signals(bundle, url), make_enrichment(**options), NOW
    )
    return render_notice(report)


def test_notice_starts_with_draft_banner_and_has_utc_timestamp(model_path: Path) -> None:
    notice = notice_for(model_path)

    assert notice.startswith("> **DRAFT for analyst review — not sent.**")
    assert "- Report time: 2026-11-22 14:03 UTC" in notice
    assert "Warning" not in notice


def test_notice_states_verdict_probability_and_threshold(model_path: Path) -> None:
    notice = notice_for(model_path)

    assert (
        "- Verdict: **phishing** (automated classification; probability 0.93, "
        "decision threshold 0.80)"
    ) in notice


def test_notice_is_defanged(model_path: Path) -> None:
    notice = notice_for(model_path)

    assert SUSPECT_HOST not in notice
    assert "192.0.2.10" not in notice
    assert "`hxxps://secure-login[.]account-verify[.]example[.]xyz/verify`" in notice
    assert (
        "**Subject:** Phishing site report: secure-login[.]account-verify[.]example[.]xyz" in notice
    )


def test_notice_to_hosting_with_registrar_in_cc(model_path: Path) -> None:
    notice = notice_for(model_path)

    assert "**To:** abuse@hosting.example\n**Cc:** abuse@registrar.example" in notice
    assert "served through your network" in notice
    assert "Please remove or disable access to the content" in notice


def test_notice_to_registrar_uses_registrar_wording(model_path: Path) -> None:
    notice = notice_for(model_path, hosting_email=None)

    assert "**To:** abuse@registrar.example" in notice
    assert "**Cc:**" not in notice
    assert "a domain registered through your service" in notice
    assert "served through your network" not in notice
    assert "> Note: no hosting abuse contact was found (not present in lookup response)." in notice


def test_notice_without_contacts_asks_for_recipient(model_path: Path) -> None:
    notice = notice_for(model_path, hosting_email=None, registrar_email=None)

    assert "**To:** [no abuse contact found; add a recipient]" in notice
    assert "served through your network" not in notice
    assert "registered through your service" not in notice


def test_notice_cdn_note_when_cdn_is_recipient(model_path: Path) -> None:
    notice = notice_for(model_path, cdn=True)

    assert "## Note on hosting" in notice
    assert "served through Cloudflare, a content delivery network" in notice
    assert "We are contacting you as the provider able to identify the origin host" in notice


def test_notice_cdn_note_when_registrar_is_recipient(model_path: Path) -> None:
    notice = notice_for(model_path, cdn=True, hosting_email=None)

    assert "Cloudflare can identify the origin host" in notice
    assert "We are contacting you as the provider" not in notice


def test_notice_without_cdn_has_no_hosting_note(model_path: Path) -> None:
    assert "## Note on hosting" not in notice_for(model_path)


def test_notice_evidence_order_puts_verifiable_facts_first(model_path: Path) -> None:
    notice = notice_for(model_path)

    facts = notice.index("Registration, hosting and certificate details")
    classifier = notice.index("URL characteristics identified by the automated classifier")
    structure = notice.index("URL structure:")
    assert facts < classifier < structure
    assert (
        "- Domain example[.]xyz registered 2026-11-20 (2 days before this report) with "
        "Example Registrar LLC."
    ) in notice
    assert (
        "- Resolves to 192[.]0[.]2[.]10 (AS64500 EXAMPLE-HOSTING, ZZ, Example Hosting Ltd)."
        in notice
    )
    assert "- No userinfo (@) or embedded-redirect (//) tricks found in the URL." in notice


def test_notice_lists_structure_flags(model_path: Path) -> None:
    notice = notice_for(model_path, url="hxxp://paypal.com@evil[.]example[.]xyz/r//next")

    assert "The URL contains an @ before the host" in notice
    assert "The URL path contains //" in notice


@pytest.mark.parametrize("is_phishing", [False])
def test_notice_for_benign_verdict_carries_warning(model_path: Path, is_phishing: bool) -> None:
    notice = notice_for(model_path, is_phishing=is_phishing)

    assert "> **Warning:** the model did NOT classify this URL as phishing" in notice
    assert "- Verdict: **not phishing**" in notice


def _report_with_signals(
    model_path: Path, directions: list[str], probability: float = 0.93
) -> dict:
    prediction = Prediction(
        probability=probability,
        threshold=0.7982,
        is_phishing=probability >= 0.7982,
        model="logistic_regression",
        target_precision=0.98,
    )
    signals = [
        Signal(feature=f"f{i}", value=1, contribution=0.1, direction=d, text=f"Signal {i} ({d}).")
        for i, d in enumerate(directions)
    ]
    return build_report(SUSPECT_URL, prediction, signals, make_enrichment(), NOW)


def test_notice_lists_only_phishing_direction_signals(model_path: Path) -> None:
    report = _report_with_signals(model_path, ["phishing", "legitimate", "phishing"])

    notice = render_notice(report)

    assert "- Signal 0 (phishing)." in notice
    assert "- Signal 2 (phishing)." in notice
    assert "legitimate)." not in notice
    assert len(report["signals"]) == 3


def test_notice_says_so_when_no_signal_points_to_phishing(model_path: Path) -> None:
    notice = render_notice(_report_with_signals(model_path, ["legitimate", "legitimate"]))

    assert "- None of the classifier's top signals pointed towards phishing." in notice
    assert "Signal 0" not in notice


@pytest.mark.parametrize(
    ("probability", "shown"),
    [(0.9999, "probability >0.99, decision threshold 0.80"), (0.93, "probability 0.93,")],
)
def test_notice_never_rounds_probability_to_certainty(
    model_path: Path, probability: float, shown: str
) -> None:
    notice = render_notice(_report_with_signals(model_path, ["phishing"], probability))

    assert shown in notice
    assert "1.00" not in notice


def test_notice_warning_shows_tiny_probability_as_below_one_percent(model_path: Path) -> None:
    notice = render_notice(_report_with_signals(model_path, ["legitimate"], probability=0.001))

    assert "(probability <0.01, threshold 0.80)" in notice
    assert "0.00" not in notice
