"""Evaluate the saved model on a live phishing-feed snapshot vs a Tranco benign sample.

Run from the repository root after training and downloading the Tranco list:
``python -m phish_triage.fresh --date 2026-11-22``.

Classification only: no enrichment and no contact with any phishing host. Feed URLs
are parsed locally and never written out; the report holds counts and metrics only.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phish_triage.evaluate import DEFAULT_THRESHOLD, REPORTS_DIR, threshold_metrics, write_summary
from phish_triage.features import (
    SHARED_PLATFORMS,
    SHARED_PLATFORMS_ADDED_FROM_FEED,
    canonical_host,
    features_frame,
    has_ip_host,
    registered_domain,
)
from phish_triage.train import DATA_PATH, MODEL_PATH, SEED, load_dataset

FEEDS_DIR = Path("data/feeds")
TRANCO_DIR = Path("data/tranco")
TRANCO_TOP = 100_000
TOP_PLATFORMS = 5
EXCLUSION_RULES = ("registered_domain", "shared_aware", "full_host")
CHOSEN_RULE = "shared_aware"
TRANCO_CITATION = (
    "Le Pochat et al., 'Tranco: A Research-Oriented Top Sites Ranking Hardened Against "
    "Manipulation', NDSS 2019. https://tranco-list.eu"
)
FEED_ATTRIBUTION = {
    "openphish": "OpenPhish community feed, https://openphish.com (not redistributed)",
    "phishunt": "phishunt.io feed, https://phishunt.io (CC0)",
}
CAVEATS = [
    "Tranco homepages are easy negatives: bare popular domains with the same shape as "
    "PhiUSIIL's legitimate URLs, so the false-positive rate is optimistic.",
    "Removing Tranco domains already in PhiUSIIL leaves less popular domains, which still "
    "look nothing like typical phishing URLs.",
    "Feeds can contain entries that are not phishing, or sites already taken down; entries "
    "were not verified because that would require contacting them.",
    "Precision is measured at a 1:1 phishing-to-benign balance; real-world precision at the "
    "same false-positive rate would be far lower because phishing is rare.",
    "Recall on shared-platform hosts is probably flattered: PhiUSIIL's legitimate URLs "
    "almost never have subdomains, so any subdomain pushes towards phishing.",
    "One day's snapshot; the sample is small.",
]


def load_feed(feed_dir: Path) -> pd.DataFrame:
    """Read every ``*.txt`` feed in a snapshot directory.

    Args:
        feed_dir: Directory holding one URL-per-line feed file per source.

    Returns:
        Columns ``url`` and ``source`` (file stem), one row per line, blanks skipped.
    """
    rows = [
        (path.stem, line.strip())
        for path in sorted(feed_dir.glob("*.txt"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return pd.DataFrame(rows, columns=["source", "url"])[["url", "source"]]


def annotate(urls: pd.DataFrame) -> pd.DataFrame:
    """Add canonical host, registered domain and host-type columns.

    Args:
        urls: A frame with a ``url`` column.

    Returns:
        A copy with ``host``, ``registered_domain``, ``is_ip`` and ``shared`` columns.
    """
    out = urls.copy()
    out["host"] = out["url"].map(canonical_host)
    out["registered_domain"] = out["url"].map(registered_domain)
    out["is_ip"] = out["url"].map(has_ip_host).astype(bool)
    out["shared"] = out["registered_domain"].isin(SHARED_PLATFORMS)
    return out


def seen_mask(
    frame: pd.DataFrame, rule: str, seen_hosts: set[str], seen_domains: set[str]
) -> pd.Series:
    """Flag rows whose domain or host appears in the training data under one rule.

    Rules: ``registered_domain`` matches the registered domain everywhere;
    ``full_host`` matches the full host everywhere; ``shared_aware`` matches the full
    host on shared platforms (where the registered domain is the platform's) and the
    registered domain elsewhere.

    Args:
        frame: Output of ``annotate``.
        rule: One of ``EXCLUSION_RULES``.
        seen_hosts: Canonical hosts in the training data.
        seen_domains: Registered domains in the training data.

    Returns:
        A boolean Series, True where the row counts as seen.
    """
    by_host = frame["host"].isin(seen_hosts)
    by_domain = frame["registered_domain"].isin(seen_domains)
    if rule == "registered_domain":
        return by_domain
    if rule == "full_host":
        return by_host
    if rule == "shared_aware":
        return by_host.where(frame["shared"], by_domain)
    raise ValueError(f"Unknown exclusion rule: {rule}")


def prepare_positives(
    feed: pd.DataFrame, seen_hosts: set[str], seen_domains: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deduplicate feed URLs by host and remove hosts seen in training.

    Args:
        feed: Output of ``load_feed``.
        seen_hosts: Canonical hosts in the training data.
        seen_domains: Registered domains in the training data.

    Returns:
        The remaining positives (one row per host) and a dictionary of counts,
        including how many hosts each exclusion rule would remove.
    """
    unique_urls = annotate(feed.drop_duplicates("url"))
    with_host = unique_urls[unique_urls["host"] != ""]
    hosts = with_host.drop_duplicates("host").reset_index(drop=True)
    removed = {
        rule: int(seen_mask(hosts, rule, seen_hosts, seen_domains).sum())
        for rule in EXCLUSION_RULES
    }
    kept = hosts[~seen_mask(hosts, CHOSEN_RULE, seen_hosts, seen_domains)].reset_index(drop=True)
    counts = {
        "feed_lines": {k: int(v) for k, v in feed.groupby("source").size().items()},
        "unique_urls": int(len(unique_urls)),
        "urls_without_host": int(len(unique_urls) - len(with_host)),
        "unique_hosts": int(len(hosts)),
        "hosts_removed_by_rule": removed,
        "rule_applied": CHOSEN_RULE,
    }
    return kept, counts


def load_tranco(path: Path, top: int = TRANCO_TOP) -> list[str]:
    """Read the first ``top`` domains of a Tranco ``rank,domain`` CSV, in rank order."""
    frame = pd.read_csv(path, header=None, names=["rank", "domain"], nrows=top)
    return frame["domain"].astype(str).str.strip().str.lower().tolist()


def sample_negatives(
    domains: list[str],
    size: int,
    seen_domains: set[str],
    feed_hosts: set[str],
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Draw a fixed-seed sample of Tranco homepages not seen in training or in the feed.

    Args:
        domains: Tranco domains in rank order.
        size: Number of negatives to draw.
        seen_domains: Registered domains in the training data.
        feed_hosts: Canonical hosts of the phishing feed.
        seed: Random seed.

    Returns:
        The sample (``url`` = ``https://<domain>/``, annotated) and overlap counts.

    Raises:
        ValueError: If fewer than ``size`` eligible domains remain.
    """
    tranco = annotate(pd.DataFrame({"url": [f"https://{d}/" for d in dict.fromkeys(domains)]}))
    in_training = tranco["registered_domain"].isin(seen_domains)
    in_feed = tranco["host"].isin(feed_hosts) & ~in_training
    eligible = tranco[~in_training & ~in_feed]
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} eligible Tranco domains for {size} negatives")
    counts = {
        "tranco_domains": int(len(tranco)),
        "overlap_with_training": int(in_training.sum()),
        "overlap_with_feed_hosts": int(in_feed.sum()),
        "eligible": int(len(eligible)),
    }
    return eligible.sample(n=size, random_state=seed).reset_index(drop=True), counts


def _recall(y_pred: np.ndarray) -> float:
    return round(float(y_pred.mean()), 4) if len(y_pred) else 0.0


def evaluate_fresh(
    model: Any, threshold: float, positives: pd.DataFrame, negatives: pd.DataFrame
) -> dict[str, Any]:
    """Score positives and negatives and compute overall and per-group metrics.

    Args:
        model: The fitted chosen pipeline.
        threshold: Its saved decision threshold.
        positives: Annotated phishing rows.
        negatives: Annotated benign rows.

    Returns:
        Metrics at the chosen threshold and at 0.5, ROC-AUC, and recall by host
        group, by top shared platform and by feed source.
    """
    both = pd.concat([positives, negatives], ignore_index=True)
    y_true = np.r_[np.ones(len(positives), int), np.zeros(len(negatives), int)]
    proba = model.predict_proba(features_frame(both["url"]))[:, 1]

    metrics = {}
    for key, cut in (("at_chosen_threshold", threshold), ("at_0_5", DEFAULT_THRESHOLD)):
        result = threshold_metrics(y_true, proba, cut)
        cm = result["confusion_matrix"]
        result["false_positive_rate"] = round(cm["fp"] / max(cm["fp"] + cm["tn"], 1), 4)
        metrics[key] = result
    metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)

    hits = proba[: len(positives)] >= threshold
    group = np.select(
        [positives["is_ip"], positives["shared"]],
        ["ip_host", "shared_platform"],
        "dedicated_domain",
    )
    by_group = {
        name: {"count": int((group == name).sum()), "recall": _recall(hits[group == name])}
        for name in ("shared_platform", "dedicated_domain", "ip_host")
        if (group == name).any()
    }
    platforms = positives.loc[positives["shared"], "registered_domain"].value_counts()
    by_platform = {
        name: {
            "count": int(n),
            "recall": _recall(hits[(positives["registered_domain"] == name).to_numpy()]),
        }
        for name, n in platforms.head(TOP_PLATFORMS).items()
    }
    by_source = {
        name: {"count": int(mask.sum()), "recall": _recall(hits[mask])}
        for name in sorted(positives["source"].unique())
        for mask in [(positives["source"] == name).to_numpy()]
    }
    return {
        "metrics": metrics,
        "recall_by_group": by_group,
        "recall_by_top_shared_platform": by_platform,
        "recall_by_source": by_source,
    }


def _latest_tranco_csv() -> Path:
    candidates = sorted(
        TRANCO_DIR.glob(f"tranco_*_top{TRANCO_TOP}.csv"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError("No Tranco list found. Run: python scripts/download_tranco.py")
    return candidates[-1]


def main(argv: list[str] | None = None) -> None:
    """Run the fresh-feed evaluation and write ``reports/fresh_<date>.json``."""
    parser = argparse.ArgumentParser(description="Evaluate on a live phishing-feed snapshot.")
    parser.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--tranco", type=Path, help="Tranco CSV (default: latest download).")
    args = parser.parse_args(argv)

    tranco_path = args.tranco or _latest_tranco_csv()
    tranco_meta_path = tranco_path.with_suffix(".json")
    tranco_meta = json.loads(tranco_meta_path.read_text()) if tranco_meta_path.exists() else {}

    training = load_dataset(DATA_PATH)
    seen_hosts = set(training["url"].map(canonical_host))
    seen_domains = set(training["url"].map(registered_domain))

    positives, positive_counts = prepare_positives(
        load_feed(FEEDS_DIR / args.date), seen_hosts, seen_domains
    )
    negatives, negative_counts = sample_negatives(
        load_tranco(tranco_path), len(positives), seen_domains, set(positives["host"])
    )

    bundle = joblib.load(MODEL_PATH)
    chosen = bundle["chosen"]
    results = evaluate_fresh(
        bundle["models"][chosen], bundle["thresholds"][chosen], positives, negatives
    )

    report = {
        "snapshot_date": args.date,
        "model": chosen,
        "sample_sizes": {"phishing": int(len(positives)), "benign": int(len(negatives))},
        "positives": positive_counts,
        "negatives": negative_counts,
        "seed": SEED,
        **results,
        "shared_platforms": {
            "list_size": len(SHARED_PLATFORMS),
            "added_from_this_feed": sorted(SHARED_PLATFORMS_ADDED_FROM_FEED),
            "note": "These platforms were added to the shared-platform list after reviewing "
            "this feed snapshot. The list affects grouping and the exclusion rule only, "
            "not the model.",
        },
        "tranco": {
            "list_id": tranco_meta.get("list_id", tranco_path.stem),
            "date_range": [
                tranco_meta.get("configuration", {}).get("startDate"),
                tranco_meta.get("configuration", {}).get("endDate"),
            ],
            "top": TRANCO_TOP,
            "citation": TRANCO_CITATION,
        },
        "feed_attribution": FEED_ATTRIBUTION,
        "caveats": CAVEATS,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"fresh_{args.date}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary()
    m = results["metrics"]["at_chosen_threshold"]
    print(
        f"{len(positives)} phishing vs {len(negatives)} benign | recall {m['recall']} | "
        f"FPR {m['false_positive_rate']} | precision {m['precision']} | F1 {m['f1']}"
    )
    print(f"Saved {out} and updated {REPORTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
