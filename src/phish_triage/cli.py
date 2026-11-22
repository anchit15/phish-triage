"""Command-line entry point: verdict, evidence, abuse contacts and a takedown notice draft.

The suspect URL is never fetched. The only network activity is passive enrichment.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape

from phish_triage import __version__
from phish_triage.enrich import enrich
from phish_triage.explain import (
    ModelNotFoundError,
    UnsupportedModelError,
    load_bundle,
    predict,
    top_signals,
)
from phish_triage.features import url_host
from phish_triage.notice import render_notice
from phish_triage.report import build_report, format_probability

DEFAULT_MODEL_PATH = Path("models/model.joblib")
LABEL_WIDTH = 15

app = typer.Typer(add_completion=False, help="Triage a suspicious URL.")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"phish-triage {__version__}")
        raise typer.Exit()


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _row(console: Console, label: str, value: str) -> None:
    console.print(f"  {label:<{LABEL_WIDTH}}{escape(value)}")


def _domain_line(domain: dict[str, Any]) -> str | None:
    parts = []
    if domain["registered_domain"]:
        parts.append(domain["registered_domain"])
    if domain["created"]:
        parts.append(f"registered {domain['created'][:10]} ({domain['age_days']} days ago)")
    if domain["registrar"]:
        parts.append(domain["registrar"])
    return " | ".join(parts) or None


def _hosting_line(host: dict[str, Any]) -> str | None:
    if not host["ip"]:
        return None
    asn = " ".join(filter(None, [f"AS{host['asn']}" if host["asn"] else None, host["asn_name"]]))
    return " | ".join(filter(None, [host["ip"], asn, host["hosting_org"]]))


def _certificate_line(tls: dict[str, Any]) -> str | None:
    if not tls["issuer"]:
        return None
    line = tls["issuer"]
    if tls["not_before"]:
        line += f" | valid from {tls['not_before'][:10]} ({tls['age_days']} days old)"
    return line


def print_report(report: dict[str, Any], console: Console) -> None:
    """Print the report for a terminal.

    Args:
        report: The output of ``phish_triage.report.build_report``.
        console: The Rich console to print to.
    """
    verdict = report["verdict"]
    enrichment = report["enrichment"]
    style = "bold red" if verdict["label"] == "phishing" else "bold green"
    probability = format_probability(verdict["probability"])

    console.print(f"phish-triage {report['tool']['version']} | {report['generated_at']}")
    _row(console, "URL", report["url"])
    console.print(
        f"  {'Verdict':<{LABEL_WIDTH}}[{style}]{verdict['label'].upper()}[/] | probability "
        f"{probability} | threshold {verdict['threshold']:.2f} (set for "
        f"{verdict['target_precision']:.0%} precision on validation data)"
    )

    console.print(f"\n[bold]Model signals[/]: top {len(report['signals'])} SHAP contributions")
    for signal in report["signals"]:
        arrow = "+ phishing  " if signal["direction"] == "phishing" else "- legitimate"
        console.print(f"  {arrow}  {escape(signal['text'])}")

    console.print("\n[bold]Enrichment[/]: registry, DNS and TLS facts (not used by the model)")
    for label, line in (
        ("Domain", _domain_line(enrichment["domain"])),
        ("Hosting", _hosting_line(enrichment["host"])),
        ("Certificate", _certificate_line(enrichment["tls"])),
    ):
        if line:
            _row(console, label, line)
    if report["cdn"]:
        name = report["cdn"]["name"]
        _row(
            console,
            "! CDN",
            f"{name} is a CDN/reverse proxy: the origin server is hidden. Report to "
            f"{name}'s abuse contact; they can identify the origin host.",
        )

    flags = report["url_flags"]
    console.print("\n[bold]URL structure flags[/] (not used by the model)")
    console.print(
        f"  @ before host (hides real host)     {'yes' if flags['has_at_symbol'] else 'no'}"
    )
    console.print(
        f"  // inside path (embedded redirect)  "
        f"{'yes' if flags['has_double_slash_redirect'] else 'no'}"
    )

    console.print("\n[bold]Abuse contacts[/]")
    if not report["abuse_contacts"]:
        console.print("  none found")
    for contact in report["abuse_contacts"]:
        label = contact["role"].capitalize() + (" (CDN)" if contact["via_cdn"] else "")
        _row(console, label, contact["email"])

    unavailable = []
    for section in ("domain", "host", "tls"):
        # One line per reason, so a failed lookup is not repeated for every field.
        fields_by_reason: dict[str, list[str]] = {}
        for name, reason in enrichment[section]["errors"].items():
            fields_by_reason.setdefault(reason, []).append(name)
        for reason, names in fields_by_reason.items():
            unavailable.append(f"{section} ({', '.join(names)}): {reason}")
    if unavailable:
        console.print("\n[bold]Unavailable[/]")
        for line in unavailable:
            console.print(f"  {escape(line)}")


@app.command()
def main(
    url: Annotated[
        str, typer.Argument(help="Suspect URL, plain or defanged (hxxp://example[.]com).")
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    notice: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Write a takedown notice draft (Markdown) here."),
    ] = None,
    no_touch: Annotated[
        bool,
        typer.Option(help="Skip the TLS handshake so the suspect host is never contacted."),
    ] = False,
    model: Annotated[
        Path, typer.Option(help="Model bundle written by phish_triage.train.")
    ] = DEFAULT_MODEL_PATH,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Classify a suspect URL, collect passive evidence and draft a takedown notice."""
    if not url_host(url):
        _fail(f"no host found in {url!r}.")
    try:
        bundle = load_bundle(model)
        prediction = predict(bundle, url)
        signals = top_signals(bundle, url)
    except (ModelNotFoundError, UnsupportedModelError) as exc:
        _fail(str(exc))

    # None lets enrich() fall back to PHISH_TRIAGE_NO_TOUCH.
    enrichment = enrich(url, touch=False if no_touch else None)
    report = build_report(url, prediction, signals, enrichment, datetime.now(UTC))

    if notice:
        try:
            notice.write_text(render_notice(report), encoding="utf-8")
        except OSError as exc:
            _fail(f"could not write notice: {exc}")

    if json_output:
        # ASCII escapes keep the JSON printable on any console encoding.
        typer.echo(json.dumps(report, indent=2))
    else:
        # Registry and certificate names can contain characters a legacy Windows
        # console cannot encode; replace them rather than crash mid-report.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        print_report(report, Console(soft_wrap=True))
    if notice:
        typer.echo(f"Notice draft written to {notice}", err=json_output)
