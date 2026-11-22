import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from phish_triage import fresh
from phish_triage.evaluate import fresh_summary_markdown

# Training reference: a dedicated domain and one host on a shared platform.
SEEN_HOSTS = {"knownphish-fixture.com", "old.pages.dev", "popular.example"}
SEEN_DOMAINS = {"knownphish-fixture.com", "pages.dev", "popular.example"}

FEED_URLS = {
    "openphish": [
        "https://login.knownphish-fixture.com/a",  # dedicated domain seen in training
        "https://old.pages.dev/x",  # shared-platform host seen in training
        "https://new.pages.dev/x",  # shared-platform host not seen
        "https://new.pages.dev/y",  # same host, different URL
        "",
    ],
    "phishunt": [
        "https://new.pages.dev/x",  # duplicate URL across feeds
        "https://secure-verify.freshphish-fixture.com/login",
        "http://192.0.2.7/login",
    ],
}


@pytest.fixture
def feed_dir(tmp_path: Path) -> Path:
    for source, urls in FEED_URLS.items():
        (tmp_path / f"{source}.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    return tmp_path


def test_load_feed_skips_blank_lines_and_records_source(feed_dir: Path) -> None:
    feed = fresh.load_feed(feed_dir)

    assert len(feed) == 7
    assert feed["source"].value_counts().to_dict() == {"openphish": 4, "phishunt": 3}


@pytest.mark.parametrize(
    ("rule", "seen"),
    [
        ("registered_domain", {"login.knownphish-fixture.com", "old.pages.dev", "new.pages.dev"}),
        ("shared_aware", {"login.knownphish-fixture.com", "old.pages.dev"}),
        ("full_host", {"old.pages.dev"}),
    ],
)
def test_seen_mask_rules(feed_dir: Path, rule: str, seen: set[str]) -> None:
    frame = fresh.annotate(fresh.load_feed(feed_dir).drop_duplicates("url"))

    mask = fresh.seen_mask(frame, rule, SEEN_HOSTS, SEEN_DOMAINS)

    assert set(frame.loc[mask, "host"]) == seen


def test_seen_mask_rejects_unknown_rule(feed_dir: Path) -> None:
    frame = fresh.annotate(fresh.load_feed(feed_dir))

    with pytest.raises(ValueError, match="Unknown exclusion rule"):
        fresh.seen_mask(frame, "nope", SEEN_HOSTS, SEEN_DOMAINS)


def test_prepare_positives_dedupes_by_host_and_reports_every_rule(feed_dir: Path) -> None:
    positives, counts = fresh.prepare_positives(fresh.load_feed(feed_dir), SEEN_HOSTS, SEEN_DOMAINS)

    assert sorted(positives["host"]) == [
        "192.0.2.7",
        "new.pages.dev",
        "secure-verify.freshphish-fixture.com",
    ]
    assert counts["unique_urls"] == 6
    assert counts["unique_hosts"] == 5
    assert counts["hosts_removed_by_rule"] == {
        "registered_domain": 3,
        "shared_aware": 2,
        "full_host": 1,
    }
    assert counts["rule_applied"] == "shared_aware"


def test_load_tranco_reads_top_n_in_rank_order(tmp_path: Path) -> None:
    csv = tmp_path / "tranco.csv"
    csv.write_text("1,Example.COM\n2,second.example\n3,third.example\n", encoding="utf-8")

    assert fresh.load_tranco(csv, top=2) == ["example.com", "second.example"]


def test_sample_negatives_excludes_seen_and_feed_and_is_reproducible() -> None:
    domains = [
        "popular.example",
        "freshphish-fixture.com",
        *[f"site{i}.example" for i in range(20)],
    ]
    feed_hosts = {"freshphish-fixture.com"}

    first, counts = fresh.sample_negatives(domains, 5, SEEN_DOMAINS, feed_hosts, seed=1)
    second, _ = fresh.sample_negatives(domains, 5, SEEN_DOMAINS, feed_hosts, seed=1)

    assert first["url"].tolist() == second["url"].tolist()
    assert len(first) == 5
    assert all(url.startswith("https://site") and url.endswith(".example/") for url in first["url"])
    assert counts == {
        "tranco_domains": 22,
        "overlap_with_training": 1,
        "overlap_with_feed_hosts": 1,
        "eligible": 20,
    }


def test_sample_negatives_needs_enough_domains() -> None:
    with pytest.raises(ValueError, match="Only 1 eligible"):
        fresh.sample_negatives(["one.example"], 2, set(), set())


class FixedModel:
    """Returns preset probabilities in input order."""

    def __init__(self, proba: list[float]):
        self.proba = np.array(proba)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.c_[1 - self.proba, self.proba]


def test_evaluate_fresh_metrics_and_groups() -> None:
    positives = fresh.annotate(
        pd.DataFrame(
            {
                "url": [
                    "https://a.pages.dev/",
                    "https://b.pages.dev/",
                    "https://c.vercel.app/",
                    "https://dedicated.example/",
                    "http://192.0.2.7/",
                ],
                "source": ["openphish", "phishunt", "phishunt", "phishunt", "openphish"],
            }
        )
    )
    negatives = fresh.annotate(
        pd.DataFrame({"url": [f"https://benign{i}.example/" for i in range(5)]})
    )
    # Positives: 4 of 5 caught; negatives: 1 false positive.
    model = FixedModel([0.9, 0.2, 0.95, 0.85, 0.99, 0.1, 0.9, 0.1, 0.1, 0.1])

    result = fresh.evaluate_fresh(model, 0.8, positives, negatives)

    m = result["metrics"]["at_chosen_threshold"]
    assert m["recall"] == 0.8
    assert m["false_positive_rate"] == 0.2
    assert m["precision"] == 0.8
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 4}
    assert result["recall_by_group"] == {
        "shared_platform": {"count": 3, "recall": 0.6667},
        "dedicated_domain": {"count": 1, "recall": 1.0},
        "ip_host": {"count": 1, "recall": 1.0},
    }
    assert result["recall_by_top_shared_platform"] == {
        "pages.dev": {"count": 2, "recall": 0.5},
        "vercel.app": {"count": 1, "recall": 1.0},
    }
    assert result["recall_by_source"]["openphish"] == {"count": 2, "recall": 1.0}
    assert "at_0_5" in result["metrics"]


def test_main_writes_counts_only_report_and_summary(
    feed_dir: Path, tmp_path: Path, model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "feeds" / "2026-11-22"
    snapshot.mkdir(parents=True)
    for path in feed_dir.glob("*.txt"):
        (snapshot / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tranco = tmp_path / "tranco_TEST_top100000.csv"
    tranco.write_text("".join(f"{i},site{i}.example\n" for i in range(1, 30)), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(json.dumps(_minimal_metrics()), encoding="utf-8")
    training = pd.DataFrame(
        {"url": ["https://login.knownphish-fixture.com/", "https://old.pages.dev/"]}
    )

    monkeypatch.setattr(fresh, "FEEDS_DIR", tmp_path / "feeds")
    monkeypatch.setattr(fresh, "REPORTS_DIR", reports)
    monkeypatch.setattr(fresh, "MODEL_PATH", model_path)
    monkeypatch.setattr(fresh, "load_dataset", lambda path: training)
    monkeypatch.setattr(fresh, "write_summary", lambda: _write_summary(reports))

    fresh.main(["--date", "2026-11-22", "--tranco", str(tranco)])

    text = (reports / "fresh_2026-11-22.json").read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["sample_sizes"] == {"phishing": 3, "benign": 3}
    assert report["shared_platforms"]["added_from_this_feed"]
    assert any("easy negatives" in caveat for caveat in report["caveats"])
    for leaked in ("knownphish-fixture", "freshphish-fixture", "192.0.2.7", "pages.dev/"):
        assert leaked not in text
    summary = (reports / "summary.md").read_text(encoding="utf-8")
    assert "### Fresh feed test (2026-11-22)" in summary
    for leaked in ("freshphish-fixture", "192.0.2.7", "new.pages"):
        assert leaked not in summary


def test_fresh_summary_lists_caveats_and_groups() -> None:
    text = fresh_summary_markdown(_minimal_fresh())

    assert "| shared_platform | 3 | 0.6667 |" in text
    assert "- Tranco homepages are easy negatives" in text


def _write_summary(reports: Path) -> str:
    from phish_triage.evaluate import write_summary

    return write_summary(reports)


def _minimal_fresh() -> dict:
    metrics = {
        "threshold": 0.8,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "false_positive_rate": 0.2,
        "confusion_matrix": {"tn": 4, "fp": 1, "fn": 1, "tp": 4},
    }
    return {
        "snapshot_date": "2026-11-22",
        "model": "logistic_regression",
        "sample_sizes": {"phishing": 5, "benign": 5},
        "metrics": {"at_chosen_threshold": metrics, "at_0_5": metrics, "roc_auc": 0.9},
        "recall_by_group": {"shared_platform": {"count": 3, "recall": 0.6667}},
        "tranco": {"list_id": "TEST"},
        "caveats": fresh.CAVEATS,
    }


def _minimal_metrics() -> dict:
    m = {
        "threshold": 0.8,
        "precision": 0.9,
        "recall": 0.7,
        "f1": 0.8,
        "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
    }
    return {
        "test_set": {"total": 2, "phishing": 1, "legitimate": 1},
        "chosen_model": "logistic_regression",
        "target_precision": 0.98,
        "models": {"logistic_regression": {"roc_auc": 0.9, "at_chosen_threshold": m, "at_0_5": m}},
        "bias_check": {
            "artifact_rates_by_class": {
                "legitimate": {"has_subdomain": 0.03},
                "phishing": {"has_subdomain": 0.5},
            },
            "artifact_only_model_test_roc_auc": 0.99,
        },
        "error_analysis": {"false_negatives": 0, "categories": {}},
    }


def test_model_fixture_is_loadable(model_path: Path) -> None:
    assert joblib.load(model_path)["chosen"] == "logistic_regression"
