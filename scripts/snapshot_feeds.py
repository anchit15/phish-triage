"""Save today's OpenPhish and phishunt.io feeds to data/feeds/YYYY-MM-DD/.

Only the published feed lists are downloaded; the URLs inside them are never visited.
OpenPhish data must be attributed and must not be redistributed. phishunt.io is CC0.
"""

import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

FEEDS = {
    "openphish": "https://openphish.com/feed.txt",
    "phishunt": "https://phishunt.io/feed.txt",
}
FEEDS_DIR = Path(__file__).resolve().parent.parent / "data" / "feeds"
USER_AGENT = "phish-triage/0.1 (+https://github.com/anchit15/phish-triage)"
DEFAULT_RETRY_AFTER = 30


def get_timeout() -> float:
    """Return the network timeout in seconds from PHISH_TRIAGE_TIMEOUT (default 10)."""
    return float(os.environ.get("PHISH_TRIAGE_TIMEOUT") or 10)


def fetch_feed(url: str) -> str:
    """Fetch a feed, waiting and retrying once if the server answers HTTP 429."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=get_timeout())
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "")
        wait = int(retry_after) if retry_after.isdigit() else DEFAULT_RETRY_AFTER
        print(f"Rate limited by {url}; retrying in {wait}s")
        time.sleep(wait)
        response = requests.get(url, headers=headers, timeout=get_timeout())
    response.raise_for_status()
    return response.text


def main() -> int:
    """Download every feed into today's snapshot directory."""
    out_dir = FEEDS_DIR / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name, url in FEEDS.items():
        try:
            text = fetch_feed(url)
        except requests.RequestException as exc:
            print(f"{name}: failed ({exc})", file=sys.stderr)
            failures += 1
            continue
        path = out_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        count = sum(1 for line in text.splitlines() if line.strip())
        print(f"{name}: saved {count} URLs to {path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
