"""Command-line entry point for phish-triage."""

import typer

from phish_triage import __version__

app = typer.Typer(add_completion=False, help="Triage a suspicious URL.")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"phish-triage {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Triage a suspicious URL."""
