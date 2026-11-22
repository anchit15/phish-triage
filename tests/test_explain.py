from pathlib import Path

import joblib
import numpy as np
import pytest
from conftest import SUSPECT_URL
from sklearn.ensemble import HistGradientBoostingClassifier

from phish_triage.explain import (
    ModelNotFoundError,
    UnsupportedModelError,
    describe,
    load_bundle,
    predict,
    shap_contributions,
    top_signals,
)
from phish_triage.features import features_frame

STATS = {
    "legitimate_medians": {
        "host_length": 15.0,
        "digit_count": 0.0,
        "domain_name_length": 10.0,
        "host_entropy": 3.34,
    },
    "tld_phishing_rate": {"xyz": {"rate": 0.8125, "count": 916}},
}


def test_load_bundle_missing_file_says_how_to_fix(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="python -m phish_triage.train"):
        load_bundle(tmp_path / "missing.joblib")


def test_predict_uses_chosen_model_and_saved_threshold(model_path: Path) -> None:
    bundle = load_bundle(model_path)

    phishy = predict(bundle, SUSPECT_URL)
    benign = predict(bundle, "https://www.brand7.com")

    assert phishy.is_phishing and not benign.is_phishing
    assert phishy.threshold == 0.5
    assert phishy.model == "logistic_regression"


def test_shap_contributions_match_coefficient_times_distance_from_mean(model_path: Path) -> None:
    bundle = load_bundle(model_path)
    pipeline = bundle["models"]["logistic_regression"]
    preprocess, classifier = pipeline.named_steps["preprocess"], pipeline.named_steps["classifier"]
    x = preprocess.transform(features_frame([SUSPECT_URL]))[0]
    mean = bundle["explanation"]["background_mean"]

    contributions = shap_contributions(bundle, SUSPECT_URL)

    manual = classifier.coef_[0] * (x - mean)
    assert sum(contributions.values()) == pytest.approx(manual.sum())
    base = classifier.intercept_[0] + classifier.coef_[0] @ mean
    log_odds = pipeline.decision_function(features_frame([SUSPECT_URL]))[0]
    assert base + sum(contributions.values()) == pytest.approx(log_odds)


def test_shap_contributions_sum_tld_columns_into_one_feature(model_path: Path) -> None:
    contributions = shap_contributions(load_bundle(model_path), SUSPECT_URL)

    assert "tld" in contributions
    assert not any(name.startswith("tld_") for name in contributions)
    assert len(contributions) == 10


def test_top_signals_are_ranked_and_described(model_path: Path) -> None:
    signals = top_signals(load_bundle(model_path), SUSPECT_URL)

    assert len(signals) == 5
    sizes = [abs(signal.contribution) for signal in signals]
    assert sizes == sorted(sizes, reverse=True)
    for signal in signals:
        assert signal.direction == ("phishing" if signal.contribution > 0 else "legitimate")
        assert signal.text.endswith(".")


def test_unsupported_model_gives_clear_error(model_path: Path, tmp_path: Path) -> None:
    bundle = load_bundle(model_path)
    pipeline = bundle["models"]["logistic_regression"]
    tree = type(pipeline)(
        [
            ("preprocess", pipeline.named_steps["preprocess"]),
            ("classifier", HistGradientBoostingClassifier()),
        ]
    )
    bundle["models"] = {"hist_gradient_boosting": tree}
    bundle["chosen"] = "hist_gradient_boosting"
    joblib.dump(bundle, tmp_path / "tree.joblib")

    with pytest.raises(UnsupportedModelError, match="logistic regression only"):
        shap_contributions(load_bundle(tmp_path / "tree.joblib"), SUSPECT_URL)


@pytest.mark.parametrize(
    ("feature", "value", "host", "expected"),
    [
        (
            "tld",
            "xyz",
            "a.example.xyz",
            "The domain uses the .xyz TLD; 81% of .xyz URLs in the training dataset were phishing.",
        ),
        (
            "tld",
            "zz",
            "a.example.zz",
            "The domain uses .zz, a TLD rarely seen in the training dataset.",
        ),
        ("tld", "ip", "192.0.2.1", "The host is an IP address, so it has no TLD."),
        (
            "host_length",
            38,
            "",
            "The host is 38 characters long; legitimate hosts in the training dataset "
            "have a median of 15.",
        ),
        ("digit_count", 0, "", "The host contains no digits, like most legitimate hosts."),
        (
            "digit_count",
            1,
            "",
            "The host contains 1 digit; legitimate hosts in the training dataset "
            "have a median of 0.",
        ),
        ("has_ip_host", 1, "", "The URL uses a raw IP address instead of a domain name."),
        ("hyphen_count", 0, "", "The host contains no hyphens."),
        (
            "hyphen_count",
            2,
            "",
            "The host contains 2 hyphens, common in brand-imitation domains such as "
            "secure-brand-login.",
        ),
        (
            "subdomain_count",
            1,
            "",
            "The host has 1 subdomain level in front of the registered domain, "
            "a common way to bury the real domain.",
        ),
        (
            "keyword_count",
            2,
            "secure-login.example.com",
            "The host contains words often used in lures: login, secure.",
        ),
        (
            "is_punycode",
            1,
            "",
            "The host uses punycode (xn--), which can disguise look-alike characters.",
        ),
    ],
)
def test_describe(feature: str, value: object, host: str, expected: str) -> None:
    assert describe(feature, value, host, STATS) == expected


def test_describe_entropy_mentions_value_and_median() -> None:
    text = describe("host_entropy", np.float64(4.1234), "", STATS)

    assert "4.12 bits per character" in text
    assert "legitimate median 3.34" in text
