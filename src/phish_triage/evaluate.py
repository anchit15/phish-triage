"""Evaluate the trained models on the held-out test set and write reports/.

Run from the repository root after training: ``python -m phish_triage.evaluate``.
"""

import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from phish_triage.defang import defang
from phish_triage.features import canonical_host, features_frame, host_features, registered_domain
from phish_triage.train import MODEL_PATH, SEED, SPLITS_DIR

REPORTS_DIR = Path("reports")
DEFAULT_THRESHOLD = 0.5
TOP_FEATURES = 10
MISS_SAMPLE_SIZE = 25
MISS_CATEGORIES = ["ip_host", "shared_platform", "plain_looking", "other"]

# Registered domains of free hosting, site-builder, cloud and IPFS-gateway platforms
# where anyone can publish under a shared domain. Used only for error analysis.
SHARED_PLATFORMS = frozenset(
    {
        "000webhostapp.com",
        "amazonaws.com",
        "appspot.com",
        "azurewebsites.net",
        "blogspot.com",
        "cloudflare-ipfs.com",
        "dweb.link",
        "firebaseapp.com",
        "github.io",
        "glitch.me",
        "godaddysites.com",
        "google.com",
        "herokuapp.com",
        "infura-ipfs.io",
        "ipfs.io",
        "jimdosite.com",
        "mystrikingly.com",
        "netlify.app",
        "pages.dev",
        "r2.dev",
        "repl.co",
        "sharepoint.com",
        "square.site",
        "vercel.app",
        "web.app",
        "webflow.io",
        "weebly.com",
        "windows.net",
        "wixsite.com",
        "wordpress.com",
        "workers.dev",
        "yolasite.com",
    }
)


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Return precision, recall, F1 and the confusion matrix at one threshold.

    Args:
        y_true: True labels (1 = phishing).
        proba: Predicted phishing probabilities.
        threshold: Probability at or above which a URL is called phishing.

    Returns:
        A JSON-serialisable mapping of metric name to value.
    """
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_model(
    model: Pipeline, x_test: pd.DataFrame, y_test: np.ndarray, threshold: float
) -> dict:
    """Score one model at the chosen threshold and at 0.5.

    Args:
        model: A fitted pipeline.
        x_test: Test features.
        y_test: Test labels.
        threshold: The threshold chosen on validation data.

    Returns:
        ROC-AUC plus threshold metrics at the chosen threshold and at 0.5.
    """
    proba = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "at_chosen_threshold": threshold_metrics(y_test, proba, threshold),
        "at_0_5": threshold_metrics(y_test, proba, DEFAULT_THRESHOLD),
    }


def top_importances(model: Pipeline, x_test: pd.DataFrame, y_test: np.ndarray) -> list[dict]:
    """Rank features by permutation importance (drop in test ROC-AUC when shuffled).

    Args:
        model: A fitted pipeline.
        x_test: Test features.
        y_test: Test labels.

    Returns:
        Up to ``TOP_FEATURES`` entries of feature name and mean importance, largest first.
    """
    result = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1
    )
    ranked = sorted(
        zip(x_test.columns, result.importances_mean, strict=True), key=lambda item: -item[1]
    )
    return [
        {"feature": name, "importance": round(float(value), 4)}
        for name, value in ranked[:TOP_FEATURES]
    ]


def artifact_flags(urls: pd.Series) -> pd.DataFrame:
    """Return the URL properties that perfectly separate PhiUSIIL legitimate URLs.

    Args:
        urls: Raw URLs from the dataset.

    Returns:
        Columns ``is_https``, ``has_www`` and ``has_path`` as 0/1 integers.
    """
    parts = urls.map(urlsplit)
    return pd.DataFrame(
        {
            "is_https": parts.map(lambda p: p.scheme == "https"),
            "has_www": parts.map(lambda p: (p.hostname or "").startswith("www.")),
            "has_path": parts.map(lambda p: p.path not in ("", "/") or bool(p.query)),
        }
    ).astype(int)


def bias_check(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Measure how much of the label the dataset's URL format alone gives away.

    Args:
        train_df: Training split with ``url`` and ``phishing`` columns.
        test_df: Test split with ``url`` and ``phishing`` columns.

    Returns:
        Per-class rates of each artifact flag across both splits, and the test
        ROC-AUC of a model trained on those flags alone. The subdomain rate is
        reported too: it is a model feature, but partly a dataset artifact.
    """
    both = pd.concat([train_df, test_df], ignore_index=True)
    flags = artifact_flags(both["url"])
    flags["has_subdomain"] = (
        features_frame(both["url"])["subdomain_count"].gt(0).astype(int).to_numpy()
    )
    rates = flags.groupby(both["phishing"].map({0: "legitimate", 1: "phishing"})).mean()

    artifact_model = LogisticRegression().fit(artifact_flags(train_df["url"]), train_df["phishing"])
    proba = artifact_model.predict_proba(artifact_flags(test_df["url"]))[:, 1]
    return {
        "artifact_rates_by_class": rates.round(3).to_dict(orient="index"),
        "artifact_only_model_test_roc_auc": round(
            float(roc_auc_score(test_df["phishing"], proba)), 4
        ),
    }


def categorise_miss(url: str) -> str:
    """Assign a missed phishing URL to a rough category, checked in a fixed order.

    Args:
        url: A phishing URL the model scored below the threshold.

    Returns:
        ``ip_host``, ``shared_platform``, ``plain_looking`` (no digits, hyphens,
        keywords, subdomains or punycode) or ``other``.
    """
    features = host_features(url)
    if features["has_ip_host"]:
        return "ip_host"
    if registered_domain(url) in SHARED_PLATFORMS:
        return "shared_platform"
    plain_keys = ("digit_count", "hyphen_count", "keyword_count", "subdomain_count", "is_punycode")
    if all(features[key] == 0 for key in plain_keys):
        return "plain_looking"
    return "other"


def error_analysis(urls: pd.Series) -> dict:
    """Count false negatives per category.

    Args:
        urls: Phishing URLs the model missed.

    Returns:
        The total and, per category, the count and percentage of misses.
    """
    counts = urls.map(categorise_miss).value_counts().reindex(MISS_CATEGORIES, fill_value=0)
    total = int(counts.sum())
    return {
        "false_negatives": total,
        "categories": {
            name: {"count": int(n), "percent": round(100 * n / total, 1) if total else 0.0}
            for name, n in counts.items()
        },
    }


def sample_missed_hosts(urls: pd.Series, n: int = MISS_SAMPLE_SIZE) -> list[str]:
    """Return a reproducible random sample of distinct missed hosts, defanged.

    Args:
        urls: Phishing URLs the model missed.
        n: Maximum number of hosts to return.

    Returns:
        Defanged canonical hosts.
    """
    hosts = urls.map(canonical_host).drop_duplicates()
    sample = hosts.sample(n=min(n, len(hosts)), random_state=SEED)
    return [defang(host) for host in sample]


def summary_markdown(metrics: dict) -> str:
    """Render the headline numbers as Markdown for the README.

    Args:
        metrics: The dictionary written to ``metrics.json``.

    Returns:
        A Markdown summary.
    """
    size = metrics["test_set"]
    lines = [
        f"Test set: {size['total']:,} URLs ({size['phishing']:,} phishing, "
        f"{size['legitimate']:,} legitimate), split by registered domain.",
        "",
        "| Model | Threshold | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, result in metrics["models"].items():
        for key in ("at_chosen_threshold", "at_0_5"):
            m = result[key]
            lines.append(
                f"| {name} | {m['threshold']} | {m['precision']} | {m['recall']} "
                f"| {m['f1']} | {result['roc_auc']} |"
            )
    bias = metrics["bias_check"]
    rates = bias["artifact_rates_by_class"]
    lines += [
        "",
        f"Chosen model: {metrics['chosen_model']}. Each model's first threshold was picked "
        f"on validation data for precision >= {metrics['target_precision']}.",
        "",
        "Bias check: a model using only https / www / has-path reaches test ROC-AUC "
        f"{bias['artifact_only_model_test_roc_auc']}; these properties are excluded "
        "from the model, which sees the host only. Subdomains remain a model feature "
        f"but are also skewed: {rates['legitimate']['has_subdomain']:.1%} of legitimate "
        f"vs {rates['phishing']['has_subdomain']:.1%} of phishing URLs have one.",
    ]
    errors = metrics["error_analysis"]
    lines += [
        "",
        f"Missed phishing URLs ({metrics['chosen_model']} at its chosen threshold): "
        f"{errors['false_negatives']:,}.",
        "",
        "| Category | Count | Share |",
        "|---|---|---|",
    ]
    for name, category in errors["categories"].items():
        lines.append(f"| {name} | {category['count']:,} | {category['percent']}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Evaluate both saved models and write ``metrics.json`` and ``summary.md``."""
    bundle = joblib.load(MODEL_PATH)
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    x_test = features_frame(test_df["url"])
    y_test = test_df["phishing"].to_numpy()

    chosen = bundle["chosen"]
    chosen_proba = bundle["models"][chosen].predict_proba(x_test)[:, 1]
    missed = (y_test == 1) & (chosen_proba < bundle["thresholds"][chosen])
    missed_urls = test_df.loc[missed, "url"]

    metrics = {
        "test_set": {
            "total": int(len(test_df)),
            "phishing": int(y_test.sum()),
            "legitimate": int(len(y_test) - y_test.sum()),
        },
        "chosen_model": chosen,
        "target_precision": bundle["target_precision"],
        "validation_roc_auc": bundle["validation_roc_auc"],
        "models": {
            name: evaluate_model(model, x_test, y_test, bundle["thresholds"][name])
            for name, model in bundle["models"].items()
        },
        "top_feature_importances": top_importances(bundle["models"][chosen], x_test, y_test),
        "bias_check": bias_check(train_df, test_df),
        "error_analysis": error_analysis(missed_urls),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "summary.md").write_text(summary_markdown(metrics), encoding="utf-8")
    print(summary_markdown(metrics))
    # The sample is printed only: reports/summary.md feeds the README, which should
    # not list live phishing hosts.
    print(f"Random sample of missed hosts (defanged, seed {SEED}):")
    for host in sample_missed_hosts(missed_urls):
        print(f"  {host}")


if __name__ == "__main__":
    main()
