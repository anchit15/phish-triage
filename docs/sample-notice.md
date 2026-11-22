> **DRAFT for analyst review — not sent.** Check every fact below before sending.

**To:** abuse@cloudflare.com
**Cc:** abuse@registrar.eu
**Subject:** Phishing site report: allegrolokalnie[.]ogoszenie-8690384[.]sbs

Dear Abuse Team,

We are writing to report a website that we believe is being used for phishing and that appears to be served through your network. We would be grateful if you could investigate and remove the content.

## Reported URL

`hxxps://allegrolokalnie[.]ogoszenie-8690384[.]sbs/`

(Defanged for safety: replace `hxxp` with `http` and `[.]` with `.`)

## Assessment

- Verdict: **phishing** (automated classification; probability >0.99, decision threshold 0.80)
- Report time: 2026-11-22 20:32 UTC

## Evidence

Registration, hosting and certificate details (from RDAP, DNS and TLS):

- Domain ogoszenie-8690384[.]sbs registered 2026-11-20 (2 days before this report) with Hosting Concepts B.V. d/b/a Registrar.eu.
- Resolves to 172[.]67[.]211[.]81 (AS13335, Cloudflare, Inc.).

URL characteristics identified by the automated classifier:

- The host contains 7 digits; legitimate hosts in the training dataset have a median of 0.
- The host has 1 subdomain level in front of the registered domain, a common way to bury the real domain.
- The host is 37 characters long; legitimate hosts in the training dataset have a median of 15.
- The host contains 1 hyphen, common in brand-imitation domains such as secure-brand-login.

URL structure:

- No userinfo (@) or embedded-redirect (//) tricks found in the URL.

## Note on hosting

This site is served through Cloudflare, a content delivery network and reverse proxy, so the origin server's address is hidden. We are contacting you as the provider able to identify the origin host and act on this report or pass it on.

## Request

Please remove or disable access to the content at the URL above and let us know once action has been taken. If you need more information, please reply to this message.

Kind regards,

[Analyst name]
[Organisation]
[Contact details]
