"""Smoke tests: the package imports and the console script is wired up."""

from typer.testing import CliRunner

from phish_triage import __version__
from phish_triage.cli import app


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_cli_prints_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
