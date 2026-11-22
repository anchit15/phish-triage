"""Render a takedown notice draft from a report, using templates/notice.md."""

from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from phish_triage.report import format_probability

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "notice.md"

OPENINGS = {
    "hosting": (
        "We are writing to report a website that we believe is being used for phishing "
        "and that appears to be served through your network. We would be grateful if you "
        "could investigate and remove the content."
    ),
    "registrar": (
        "We are writing to report a domain registered through your service that we "
        "believe is being used for phishing. We would be grateful if you could "
        "investigate and take appropriate action, such as suspending the domain."
    ),
    None: (
        "We are writing to report a website that we believe is being used for phishing. "
        "We would be grateful if you could investigate."
    ),
}
REQUESTS = {
    "hosting": (
        "Please remove or disable access to the content at the URL above and let us know "
        "once action has been taken. If you need more information, please reply to this "
        "message."
    ),
    "registrar": (
        "Please review the domain and suspend it if it breaches your terms of service, and "
        "let us know once action has been taken. If you need more information, please "
        "reply to this message."
    ),
    None: (
        "Please take appropriate action and let us know once it has been taken. If you "
        "need more information, please reply to this message."
    ),
}


def _days(count: int) -> str:
    return "1 day" if count == 1 else f"{count} days"


def _banner(report: dict[str, Any]) -> str:
    lines = ["> **DRAFT for analyst review — not sent.** Check every fact below before sending."]
    verdict = report["verdict"]
    if verdict["label"] != "phishing":
        probability = format_probability(verdict["probability"])
        lines += [
            ">",
            f"> **Warning:** the model did NOT classify this URL as phishing (probability "
            f"{probability}, threshold {verdict['threshold']:.2f}). Send only "
            "if you have independent evidence.",
        ]
    roles = {contact["role"] for contact in report["abuse_contacts"]}
    enrichment = report["enrichment"]
    if "hosting" not in roles:
        reason = enrichment["host"]["errors"].get("host_abuse_email", "not found")
        lines += [">", f"> Note: no hosting abuse contact was found ({reason})."]
    if "registrar" not in roles:
        reason = enrichment["domain"]["errors"].get("registrar_abuse_email", "not found")
        lines += [">", f"> Note: no registrar abuse contact was found ({reason})."]
    return "\n".join(lines)


def _recipients(contacts: list[dict[str, Any]]) -> str:
    if not contacts:
        return "**To:** [no abuse contact found; add a recipient]"
    lines = [f"**To:** {contacts[0]['email']}"]
    if len(contacts) > 1:
        lines.append(f"**Cc:** {', '.join(contact['email'] for contact in contacts[1:])}")
    return "\n".join(lines)


def _fact_lines(report: dict[str, Any]) -> list[str]:
    domain, host, tls = (report["enrichment"][key] for key in ("domain", "host", "tls"))
    facts = []
    if domain["created"] or domain["registrar"]:
        line = f"Domain {domain['registered_domain']}"
        if domain["created"]:
            line += (
                f" registered {domain['created'][:10]} "
                f"({_days(domain['age_days'])} before this report)"
            )
        if domain["registrar"]:
            line += f" with {domain['registrar']}"
        facts.append(line + ".")
    if host["ip"]:
        details = []
        if host["asn"] is not None:
            details.append(" ".join(filter(None, [f"AS{host['asn']}", host["asn_name"]])))
        if host["hosting_org"]:
            details.append(host["hosting_org"])
        suffix = f" ({', '.join(details)})" if details else ""
        facts.append(f"Resolves to {host['ip']}{suffix}.")
    if tls["issuer"]:
        line = f"TLS certificate issued by {tls['issuer']}"
        if tls["not_before"]:
            line += f", valid from {tls['not_before'][:10]}"
        facts.append(line + ".")
    return facts or ["No registration, hosting or certificate details could be retrieved."]


def _structure_lines(flags: dict[str, bool]) -> list[str]:
    lines = []
    if flags["has_at_symbol"]:
        lines.append("The URL contains an @ before the host, which hides the real destination.")
    if flags["has_double_slash_redirect"]:
        lines.append("The URL path contains //, often used to embed a redirect.")
    return lines or ["No userinfo (@) or embedded-redirect (//) tricks found in the URL."]


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _signal_lines(signals: list[dict[str, Any]]) -> list[str]:
    # Signals pointing towards "legitimate" are useful to the analyst but are not
    # evidence of phishing, so the notice leaves them out.
    lines = [signal["text"] for signal in signals if signal["direction"] == "phishing"]
    return lines or ["None of the classifier's top signals pointed towards phishing."]


def _evidence(report: dict[str, Any]) -> str:
    # Verifiable registry facts come first: they are what an abuse desk can check.
    return "\n\n".join(
        [
            "Registration, hosting and certificate details (from RDAP, DNS and TLS):",
            _bullets(_fact_lines(report)),
            "URL characteristics identified by the automated classifier:",
            _bullets(_signal_lines(report["signals"])),
            "URL structure:",
            _bullets(_structure_lines(report["url_flags"])),
        ]
    )


def _hosting_note(report: dict[str, Any], recipient_role: str | None) -> str:
    if not report["cdn"]:
        return ""
    cdn = report["cdn"]["name"]
    if recipient_role == "hosting":
        action = (
            "We are contacting you as the provider able to identify the origin host and act "
            "on this report or pass it on."
        )
    else:
        action = f"{cdn} can identify the origin host and may also need to be notified."
    return (
        "\n## Note on hosting\n\n"
        f"This site is served through {cdn}, a content delivery network and reverse proxy, "
        f"so the origin server's address is hidden. {action}\n"
    )


def render_notice(report: dict[str, Any], template_path: Path = TEMPLATE_PATH) -> str:
    """Fill the notice template from a defanged report.

    Args:
        report: The output of ``phish_triage.report.build_report``.
        template_path: The Markdown template with ``$placeholders``.

    Returns:
        The notice as Markdown.
    """
    contacts = report["abuse_contacts"]
    recipient_role = contacts[0]["role"] if contacts else None
    verdict = report["verdict"]
    probability = format_probability(verdict["probability"])
    generated = datetime.strptime(report["generated_at"], "%Y-%m-%dT%H:%M:%SZ")
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.substitute(
        banner=_banner(report),
        recipients=_recipients(contacts),
        host=report["host"],
        opening=OPENINGS[recipient_role],
        url=report["url"],
        verdict=(
            f"**{verdict['label']}** (automated classification; probability "
            f"{probability}, decision threshold {verdict['threshold']:.2f})"
        ),
        timestamp=generated.strftime("%Y-%m-%d %H:%M UTC"),
        evidence=_evidence(report),
        hosting_note=_hosting_note(report, recipient_role),
        request=REQUESTS[recipient_role],
    )
