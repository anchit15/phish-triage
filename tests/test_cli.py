import json
from pathlib import Path

import pytest
from conftest import SUSPECT_HOST, SUSPECT_URL, make_enrichment
from typer.testing import CliRunner

from phish_triage import cli
from phish_triage.explain import Prediction

runner = CliRunner()


@pytest.fixture
def enrich_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace enrichment with a canned result and record how it was called."""
    calls: list[dict] = []

    def fake_enrich(url: str, touch: bool | None = None) -> object:
        calls.append({"url": url, "touch": touch})
        return make_enrichment()

    monkeypatch.setattr(cli, "enrich", fake_enrich)
    return calls


def run(*args: str) -> object:
    return runner.invoke(cli.app, list(args))


def test_terminal_report(model_path: Path, enrich_calls: list[dict]) -> None:
    result = run(SUSPECT_URL, "--model", str(model_path))

    assert result.exit_code == 0, result.output
    assert "PHISHING | probability" in result.stdout
    assert "Model signals: top 5 SHAP contributions" in result.stdout
    assert "Enrichment: registry, DNS and TLS facts (not used by the model)" in result.stdout
    assert "URL structure flags (not used by the model)" in result.stdout
    assert "abuse@hosting.example" in result.stdout
    assert SUSPECT_HOST not in result.stdout
    assert "192.0.2.10" not in result.stdout


def test_terminal_report_is_ascii_for_legacy_windows_consoles(
    model_path: Path, enrich_calls: list[dict]
) -> None:
    result = run(SUSPECT_URL, "--model", str(model_path))

    assert result.stdout.isascii()


def test_json_output(model_path: Path, enrich_calls: list[dict]) -> None:
    result = run(SUSPECT_URL, "--json", "--model", str(model_path))

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert set(report) == {
        "tool",
        "generated_at",
        "url",
        "host",
        "verdict",
        "signals",
        "enrichment",
        "cdn",
        "url_flags",
        "abuse_contacts",
    }
    assert report["url"].startswith("hxxps://")
    assert SUSPECT_HOST not in result.stdout


def test_notice_option_writes_draft(
    model_path: Path, enrich_calls: list[dict], tmp_path: Path
) -> None:
    out = tmp_path / "notice.md"

    result = run(SUSPECT_URL, "--json", "--notice", str(out), "--model", str(model_path))

    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    assert "Notice draft written to" in result.stderr
    notice = out.read_text(encoding="utf-8")
    assert notice.startswith("> **DRAFT for analyst review")
    assert SUSPECT_HOST not in notice


def test_missing_model_exits_with_instructions(enrich_calls: list[dict], tmp_path: Path) -> None:
    result = run(SUSPECT_URL, "--model", str(tmp_path / "missing.joblib"))

    assert result.exit_code == 2
    assert "python scripts/download_data.py && python -m phish_triage.train" in result.stderr
    assert enrich_calls == []


@pytest.mark.parametrize(("flags", "touch"), [(["--no-touch"], False), ([], None)])
def test_no_touch_passes_through(
    model_path: Path, enrich_calls: list[dict], flags: list[str], touch: bool | None
) -> None:
    result = run(SUSPECT_URL, *flags, "--model", str(model_path))

    assert result.exit_code == 0, result.output
    assert enrich_calls == [{"url": SUSPECT_URL, "touch": touch}]


def test_url_without_host_is_rejected(model_path: Path, enrich_calls: list[dict]) -> None:
    result = run("hxxp://", "--model", str(model_path))

    assert result.exit_code == 2
    assert "no host found" in result.stderr
    assert enrich_calls == []


def test_terminal_shows_capped_probability_but_json_keeps_raw_value(
    model_path: Path, enrich_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def near_certain(bundle: dict, url: str) -> Prediction:
        return Prediction(
            probability=0.99987,
            threshold=0.7982,
            is_phishing=True,
            model="logistic_regression",
            target_precision=0.98,
        )

    monkeypatch.setattr(cli, "predict", near_certain)

    terminal = run(SUSPECT_URL, "--model", str(model_path))
    as_json = run(SUSPECT_URL, "--json", "--model", str(model_path))

    assert "probability >0.99 | threshold 0.80" in terminal.stdout
    assert "1.00" not in terminal.stdout
    assert json.loads(as_json.stdout)["verdict"]["probability"] == 0.9999
