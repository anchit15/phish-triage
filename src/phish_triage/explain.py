"""Load the trained model, score a URL and explain the score with SHAP.

For logistic regression with features treated as independent, a feature's SHAP value
is ``coefficient * (value - training mean)`` in log-odds. ``shap.LinearExplainer``
computes exactly that from the training means saved in the model bundle, so no
training data is needed at prediction time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phish_triage.features import (
    SUSPICIOUS_KEYWORDS,
    canonical_host,
    features_frame,
    host_features,
)

TOP_SIGNALS = 5


class ModelNotFoundError(Exception):
    """The model bundle does not exist."""


class UnsupportedModelError(Exception):
    """The chosen model cannot be explained by this module."""


@dataclass
class Prediction:
    """The model's verdict for one URL."""

    probability: float
    threshold: float
    is_phishing: bool
    model: str
    target_precision: float


@dataclass
class Signal:
    """One feature's contribution to a prediction, with a plain-English description."""

    feature: str
    value: int | float | str
    contribution: float
    direction: str
    text: str


def load_bundle(path: Path) -> dict[str, Any]:
    """Load the model bundle written by ``phish_triage.train``.

    The bundle is a pickle, so only load files you created yourself.

    Args:
        path: Path to ``model.joblib``.

    Returns:
        The bundle dictionary.

    Raises:
        ModelNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model not found at {path}. Run: "
            "python scripts/download_data.py && python -m phish_triage.train"
        )
    return joblib.load(path)


def _chosen(bundle: dict[str, Any]) -> Pipeline:
    return bundle["models"][bundle["chosen"]]


def predict(bundle: dict[str, Any], url: str) -> Prediction:
    """Score a URL with the bundle's chosen model and its saved threshold.

    Args:
        bundle: The loaded model bundle.
        url: The URL, plain or defanged.

    Returns:
        The probability of phishing, the threshold and the resulting verdict.
    """
    name = bundle["chosen"]
    probability = float(_chosen(bundle).predict_proba(features_frame([url]))[0, 1])
    threshold = float(bundle["thresholds"][name])
    return Prediction(
        probability=probability,
        threshold=threshold,
        is_phishing=probability >= threshold,
        model=name,
        target_precision=float(bundle["target_precision"]),
    )


def shap_contributions(bundle: dict[str, Any], url: str) -> dict[str, float]:
    """Return each model feature's SHAP value (log-odds) for one URL.

    The one-hot TLD columns are summed into a single ``tld`` contribution.

    Args:
        bundle: The loaded model bundle.
        url: The URL, plain or defanged.

    Returns:
        A mapping of feature name to contribution; positive pushes towards phishing.

    Raises:
        UnsupportedModelError: If the chosen model is not logistic regression.
    """
    pipeline = _chosen(bundle)
    preprocess, classifier = pipeline.named_steps["preprocess"], pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise UnsupportedModelError(
            f"Explanations support logistic regression only; the bundle's chosen model is "
            f"{bundle['chosen']}."
        )

    # Imported here because shap takes about two seconds to load.
    import shap

    mean = np.asarray(bundle["explanation"]["background_mean"])
    # A (mean, covariance) pair is the masker for independent features; the
    # covariance is unused in that mode, so zeros suffice.
    explainer = shap.LinearExplainer(classifier, (mean, np.zeros((mean.size, mean.size))))
    values = explainer.shap_values(preprocess.transform(features_frame([url])))[0]

    contributions: dict[str, float] = {}
    for column, value in zip(preprocess.get_feature_names_out(), values, strict=True):
        transformer, feature = column.split("__", 1)
        name = "tld" if transformer == "tld" else feature
        contributions[name] = contributions.get(name, 0.0) + float(value)
    return contributions


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def describe(feature: str, value: Any, host: str, stats: dict[str, Any]) -> str:
    """Describe one feature's value in plain English for an abuse desk.

    Args:
        feature: The model feature name.
        value: This URL's value for the feature.
        host: The canonical host, used to name lure words found.
        stats: The bundle's ``explanation`` statistics.

    Returns:
        A one-sentence description.
    """
    median = stats["legitimate_medians"].get(feature)
    if feature == "tld":
        if value == "ip":
            return "The host is an IP address, so it has no TLD."
        if value == "none":
            return "The host has no recognised public suffix (TLD)."
        info = stats["tld_phishing_rate"].get(value)
        if info:
            return (
                f"The domain uses the .{value} TLD; {info['rate']:.0%} of .{value} URLs "
                "in the training dataset were phishing."
            )
        return f"The domain uses .{value}, a TLD rarely seen in the training dataset."
    if feature == "host_length":
        return (
            f"The host is {value} characters long; legitimate hosts in the training "
            f"dataset have a median of {median:g}."
        )
    if feature == "digit_count":
        if not value:
            return "The host contains no digits, like most legitimate hosts."
        return (
            f"The host contains {_plural(value, 'digit')}; legitimate hosts in the "
            f"training dataset have a median of {median:g}."
        )
    if feature == "has_ip_host":
        if value:
            return "The URL uses a raw IP address instead of a domain name."
        return "The URL uses a domain name rather than a raw IP address."
    if feature == "hyphen_count":
        if not value:
            return "The host contains no hyphens."
        return (
            f"The host contains {_plural(value, 'hyphen')}, common in brand-imitation "
            "domains such as secure-brand-login."
        )
    if feature == "subdomain_count":
        if not value:
            return "The host has no subdomains in front of the registered domain."
        return (
            f"The host has {_plural(value, 'subdomain level')} in front of the registered "
            "domain, a common way to bury the real domain."
        )
    if feature == "domain_name_length":
        return (
            f"The registered name is {value} characters long; legitimate domains in the "
            f"training dataset have a median of {median:g}."
        )
    if feature == "host_entropy":
        return (
            f"The host's character mix has an entropy of {value:.2f} bits per character "
            f"(legitimate median {median:.2f}); higher values look more random."
        )
    if feature == "keyword_count":
        found = [word for word in SUSPICIOUS_KEYWORDS if word in host]
        if not found:
            return "The host contains none of the common lure words (login, verify, account, ...)."
        return f"The host contains words often used in lures: {', '.join(found)}."
    if feature == "is_punycode":
        if value:
            return "The host uses punycode (xn--), which can disguise look-alike characters."
        return "The host does not use punycode."
    return f"{feature} = {value}."


def top_signals(bundle: dict[str, Any], url: str, k: int = TOP_SIGNALS) -> list[Signal]:
    """Return the ``k`` largest SHAP contributions as plain-English signals.

    Args:
        bundle: The loaded model bundle.
        url: The URL, plain or defanged.
        k: How many signals to return.

    Returns:
        Signals ordered by absolute contribution, largest first.
    """
    contributions = shap_contributions(bundle, url)
    values = host_features(url)
    host = canonical_host(url)
    ranked = sorted(contributions.items(), key=lambda item: -abs(item[1]))[:k]
    return [
        Signal(
            feature=name,
            value=values[name],
            contribution=round(contribution, 4),
            direction="phishing" if contribution > 0 else "legitimate",
            text=describe(name, values[name], host, bundle["explanation"]),
        )
        for name, contribution in ranked
    ]
