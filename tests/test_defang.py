import pytest

from phish_triage.defang import defang, normalise, refang


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://login.example.com/a", "hxxp://login[.]example[.]com/a"),
        ("HTTPS://example.com", "hxxps://example[.]com"),
        ("login.example.com", "login[.]example[.]com"),
        ("hxxp://example[.]com", "hxxp://example[.]com"),
        ("2001:db8::1", "2001:db8::1"),
    ],
)
def test_defang(raw: str, expected: str) -> None:
    assert defang(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hxxp://example[.]com", "http://example.com"),
        ("HXXPS://login(.)example{.}test/path", "https://login.example.test/path"),
        ("http://192[.]0[.]2[.]1[:]8080/", "http://192.0.2.1:8080/"),
        ("  https://example.com  ", "https://example.com"),
        ("https://example.com", "https://example.com"),
    ],
)
def test_refang(raw: str, expected: str) -> None:
    assert refang(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example[.]com/login", "http://example.com/login"),
        ("hxxps://example[.]com", "https://example.com"),
        ("ftp://example.com", "ftp://example.com"),
    ],
)
def test_normalise_adds_missing_scheme(raw: str, expected: str) -> None:
    assert normalise(raw) == expected
