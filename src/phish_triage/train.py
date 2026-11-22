"""Train the phishing classifier on PhiUSIIL URLs and save it to models/.

Run from the repository root: ``python -m phish_triage.train``.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phish_triage.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    features_frame,
    registered_domain,
)

DATA_PATH = Path("data/PhiUSIIL_Phishing_URL_Dataset.csv")
SPLITS_DIR = Path("data/splits")
MODEL_PATH = Path("models/model.joblib")
TARGET_PRECISION = 0.98
SEED = 42
MIN_TLD_COUNT = 50


def load_dataset(path: Path) -> pd.DataFrame:
    """Load PhiUSIIL using only the URL and label, with phishing as the positive class.

    PhiUSIIL labels legitimate URLs 1 and phishing 0, so the label is flipped.

    Args:
        path: Path to the PhiUSIIL CSV.

    Returns:
        A DataFrame with columns ``url`` and ``phishing`` (1 = phishing), without
        missing values or duplicate URLs.
    """
    df = pd.read_csv(path, usecols=["URL", "label"]).dropna()
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    return pd.DataFrame({"url": df["URL"], "phishing": 1 - df["label"].astype(int)})


def grouped_split(
    df: pd.DataFrame, test_size: float, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so that no registered domain appears on both sides.

    Args:
        df: A DataFrame with a ``url`` column.
        test_size: Fraction of groups to put in the second part.
        seed: Random seed for a reproducible split.

    Returns:
        The (train, test) parts, each with a fresh index.
    """
    groups = df["url"].map(registered_domain)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_pipeline(estimator: object) -> Pipeline:
    """Wrap an estimator with feature scaling and TLD one-hot encoding.

    Args:
        estimator: An unfitted scikit-learn classifier.

    Returns:
        A pipeline that takes a DataFrame of ``MODEL_FEATURES``.
    """
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "tld",
                # TLDs seen fewer than 50 times share one "infrequent" column, which
                # also absorbs TLDs never seen in training.
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def candidate_models() -> dict[str, Pipeline]:
    """Return the unfitted baseline and tree pipelines, keyed by name."""
    return {
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=1000)),
        "hist_gradient_boosting": build_pipeline(HistGradientBoostingClassifier(random_state=SEED)),
    }


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> float:
    """Return the lowest threshold whose precision reaches the target.

    The lowest such threshold keeps as much recall as possible.

    Args:
        y_true: True labels (1 = phishing).
        proba: Predicted phishing probabilities.
        target_precision: Minimum precision required.

    Returns:
        The decision threshold.

    Raises:
        ValueError: If no threshold reaches the target precision.
    """
    precision, _, thresholds = precision_recall_curve(y_true, proba)
    meets_target = np.flatnonzero(precision[:-1] >= target_precision)
    if meets_target.size == 0:
        raise ValueError(f"No threshold reaches precision {target_precision}")
    return float(thresholds[meets_target[0]])


def explanation_stats(model: Pipeline, x_fit: pd.DataFrame, y_fit: np.ndarray) -> dict:
    """Summarise the training data for explaining single predictions later.

    Args:
        model: The fitted pipeline to be explained.
        x_fit: The features it was fitted on.
        y_fit: The matching labels (1 = phishing).

    Returns:
        ``background_mean``: mean of each transformed column (the SHAP baseline);
        ``legitimate_medians``: median of each numeric feature over legitimate URLs;
        ``tld_phishing_rate``: phishing share and count for TLDs with at least
        ``MIN_TLD_COUNT`` training URLs.
    """
    transformed = model.named_steps["preprocess"].transform(x_fit)
    legitimate = x_fit.loc[y_fit == 0, NUMERIC_FEATURES]
    by_tld = pd.DataFrame({"tld": x_fit["tld"], "phishing": y_fit}).groupby("tld")["phishing"]
    rates = by_tld.agg(["mean", "count"])
    rates = rates[rates["count"] >= MIN_TLD_COUNT]
    return {
        "background_mean": np.asarray(transformed.mean(axis=0)).ravel(),
        "legitimate_medians": {name: float(v) for name, v in legitimate.median().items()},
        "tld_phishing_rate": {
            tld: {"rate": round(float(row["mean"]), 4), "count": int(row["count"])}
            for tld, row in rates.iterrows()
        },
    }


def main() -> None:
    """Train both candidates, pick one on validation data, and save the bundle."""
    df = load_dataset(DATA_PATH)
    train_df, test_df = grouped_split(df, test_size=0.2)
    fit_df, val_df = grouped_split(train_df, test_size=0.2)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    x_fit, y_fit = features_frame(fit_df["url"]), fit_df["phishing"].to_numpy()
    x_val, y_val = features_frame(val_df["url"]), val_df["phishing"].to_numpy()

    models = candidate_models()
    val_auc = {}
    thresholds = {}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        val_proba = model.predict_proba(x_val)[:, 1]
        val_auc[name] = float(roc_auc_score(y_val, val_proba))
        thresholds[name] = choose_threshold(y_val, val_proba, TARGET_PRECISION)
        print(
            f"{name}: validation ROC-AUC {val_auc[name]:.4f}, "
            f"threshold {thresholds[name]:.4f} for precision >= {TARGET_PRECISION}"
        )

    chosen = max(val_auc, key=val_auc.get)
    print(f"Chosen: {chosen}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "chosen": chosen,
            "thresholds": thresholds,
            "target_precision": TARGET_PRECISION,
            "validation_roc_auc": val_auc,
            "features": MODEL_FEATURES,
            "explanation": explanation_stats(models[chosen], x_fit, y_fit),
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
