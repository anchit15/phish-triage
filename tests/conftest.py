import socket
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from phish_triage.enrich import DomainInfo, Enrichment, HostInfo, TlsInfo
from phish_triage.features import MODEL_FEATURES, features_frame
from phish_triage.train import build_pipeline, explanation_stats

SUSPECT_URL = "hxxps://secure-login.account-verify[.]example[.]xyz/verify"
SUSPECT_HOST = "secure-login.account-verify.example.xyz"


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a real network connection."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Network access attempted during a test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small real model bundle trained on made-up URLs, in the same layout as train.py."""
    legitimate = [f"https://www.brand{i}.com" for i in range(60)]
    phishing = [f"http://secure-login-{i}.account-verify.example.xyz/x" for i in range(60)]
    x = features_frame(legitimate + phishing)
    y = np.array([0] * len(legitimate) + [1] * len(phishing))

    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(x, y)
    bundle = {
        "models": {"logistic_regression": model},
        "chosen": "logistic_regression",
        "thresholds": {"logistic_regression": 0.5},
        "target_precision": 0.98,
        "validation_roc_auc": {"logistic_regression": 1.0},
        "features": MODEL_FEATURES,
        "explanation": explanation_stats(model, x, y),
    }
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(bundle, path)
    return path


def make_enrichment(
    *,
    cdn: bool = False,
    hosting_email: str | None = "abuse@hosting.example",
    registrar_email: str | None = "abuse@registrar.example",
) -> Enrichment:
    """Build a canned enrichment result for the suspect URL, as enrich() would return it."""
    host = HostInfo(
        hostname=SUSPECT_HOST,
        ips=["192.0.2.10", "192.0.2.11"],
        ip="192.0.2.10",
        asn=13335 if cdn else 64500,
        asn_name="CLOUDFLARENET - Cloudflare, Inc., US" if cdn else "EXAMPLE-HOSTING, ZZ",
        hosting_org="Cloudflare, Inc." if cdn else "Example Hosting Ltd",
        host_abuse_email=hosting_email,
    )
    if hosting_email is None:
        host.errors["host_abuse_email"] = "not present in lookup response"
    domain = DomainInfo(
        registered_domain="example.xyz",
        created="2026-11-20T08:14:00+00:00",
        age_days=2,
        registrar="Example Registrar LLC",
        registrar_abuse_email=registrar_email,
        rdap_url="https://rdap.registry.example/domain/example.xyz",
    )
    if registrar_email is None:
        domain.errors["registrar_abuse_email"] = "not present in RDAP response"
    tls = TlsInfo(issuer=f"CN={SUSPECT_HOST}", not_before="2026-11-21T00:00:00+00:00")
    return Enrichment(domain=domain, host=host, tls=tls)
