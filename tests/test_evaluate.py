import pandas as pd
import pytest

from phish_triage.evaluate import categorise_miss, error_analysis, sample_missed_hosts


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("hxxp://192[.]0[.]2[.]1/login", "ip_host"),
        ("https://secure-bank-login.firebaseapp.com/", "shared_platform"),
        ("hxxps://caseid123[.]web[.]app", "shared_platform"),
        ("https://meadowfield.com/", "plain_looking"),
        ("https://www.meadowfield.com/index.html", "plain_looking"),
        ("https://account-verify.example.com/", "other"),
        ("https://shop24.example/", "other"),
    ],
)
def test_categorise_miss(url: str, expected: str) -> None:
    assert categorise_miss(url) == expected


def test_error_analysis_counts_and_percentages() -> None:
    urls = pd.Series(
        ["http://192.0.2.1/", "https://a.web.app/", "https://b.web.app/", "x1.example/"]
    )

    result = error_analysis(urls)

    assert result["false_negatives"] == 4
    assert result["categories"]["shared_platform"] == {"count": 2, "percent": 50.0}
    assert result["categories"]["ip_host"] == {"count": 1, "percent": 25.0}


def test_error_analysis_handles_no_misses() -> None:
    result = error_analysis(pd.Series([], dtype=str))

    assert result["false_negatives"] == 0
    assert result["categories"]["other"] == {"count": 0, "percent": 0.0}


def test_error_analysis_lists_every_category_even_when_empty() -> None:
    result = error_analysis(pd.Series(["http://192.0.2.1/"]))

    assert list(result["categories"]) == ["ip_host", "shared_platform", "plain_looking", "other"]
    assert result["categories"]["shared_platform"] == {"count": 0, "percent": 0.0}


def test_sample_missed_hosts_is_defanged_and_distinct() -> None:
    urls = pd.Series(
        ["https://www.a.example.com/x", "https://a.example.com/y", "http://b.example.org/"]
    )

    hosts = sample_missed_hosts(urls, n=25)

    assert sorted(hosts) == ["a[.]example[.]com", "b[.]example[.]org"]
