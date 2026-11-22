"""Convert URLs between defanged (``hxxp://example[.]com``) and parseable forms."""

import re

_DEFANGED_SCHEME = re.compile(r"^hxxp(s?)://", re.IGNORECASE)
_FANGED_SCHEME = re.compile(r"^http(s?)://", re.IGNORECASE)
_DEFANGED_DOT = re.compile(r"\[\.\]|\(\.\)|\{\.\}")
_HAS_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def refang(url: str) -> str:
    """Undo common defanging of a URL.

    Handles ``hxxp(s)://`` -> ``http(s)://``, ``[.]``/``(.)``/``{.}`` -> ``.`` and
    ``[:]`` -> ``:``.

    Args:
        url: A URL, defanged or not.

    Returns:
        The URL with defanging removed and surrounding whitespace stripped.
    """
    url = _DEFANGED_SCHEME.sub(lambda m: f"http{m.group(1).lower()}://", url.strip())
    return _DEFANGED_DOT.sub(".", url).replace("[:]", ":")


def defang(url: str) -> str:
    """Make a URL or host unclickable for display: ``http`` -> ``hxxp``, ``.`` -> ``[.]``.

    Args:
        url: A URL or bare host, defanged or not.

    Returns:
        The defanged text. Already-defanged input is not defanged twice.
    """
    url = _FANGED_SCHEME.sub(lambda m: f"hxxp{m.group(1).lower()}://", refang(url))
    return url.replace(".", "[.]")


def normalise(url: str) -> str:
    """Refang a URL and prefix ``http://`` if it has no scheme, so it parses consistently.

    Args:
        url: A URL, defanged or not, with or without a scheme.

    Returns:
        A refanged URL that always has a scheme.
    """
    url = refang(url)
    if not _HAS_SCHEME.match(url):
        url = "http://" + url
    return url
