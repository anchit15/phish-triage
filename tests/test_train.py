from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phish_triage.features import registered_domain
from phish_triage.train import choose_threshold, grouped_split, load_dataset


def test_load_dataset_flips_labels_and_ignores_other_columns(tmp_path: Path) -> None:
    csv = tmp_path / "phiusiil.csv"
    pd.DataFrame(
        {
            "URL": [
                "https://www.example.com",
                "http://login.example.test/x",
                "https://www.example.com",
            ],
            "URLSimilarityIndex": [100.0, 12.5, 100.0],
            "label": [1, 0, 1],
        }
    ).to_csv(csv, index=False)

    df = load_dataset(csv)

    assert list(df.columns) == ["url", "phishing"]
    assert df["phishing"].tolist() == [0, 1]


def test_grouped_split_keeps_each_domain_on_one_side() -> None:
    urls = [f"https://{sub}.site{n}.example/page" for n in range(20) for sub in ("a", "b", "c")]
    df = pd.DataFrame({"url": urls, "phishing": [n % 2 for n in range(len(urls))]})

    train, test = grouped_split(df, test_size=0.25)

    train_domains = set(train["url"].map(registered_domain))
    test_domains = set(test["url"].map(registered_domain))
    assert train_domains.isdisjoint(test_domains)
    assert len(train) + len(test) == len(df)


def test_choose_threshold_returns_lowest_threshold_meeting_precision() -> None:
    y_true = np.array([0, 0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    assert choose_threshold(y_true, proba, target_precision=1.0) == pytest.approx(0.8)


def test_choose_threshold_raises_when_target_unreachable() -> None:
    y_true = np.array([1, 0])
    proba = np.array([0.2, 0.9])

    with pytest.raises(ValueError):
        choose_threshold(y_true, proba, target_precision=1.0)
