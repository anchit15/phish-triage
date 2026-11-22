"""Download a Tranco top-sites list (top 100k by default) into data/tranco/.

Tranco: Le Pochat et al., "Tranco: A Research-Oriented Top Sites Ranking Hardened
Against Manipulation", NDSS 2019. https://tranco-list.eu

Without --list-id the latest daily list is used; its permanent ID is printed and saved
with the list so the evaluation can cite and reproduce it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://tranco-list.eu"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tranco"
USER_AGENT = "phish-triage/0.1 (+https://github.com/anchit15/phish-triage)"
MAX_ATTEMPTS = 3
MAX_RETRY_AFTER = 60


def get_timeout() -> float:
    """Return the network timeout in seconds from PHISH_TRIAGE_TIMEOUT (default 10)."""
    return float(os.environ.get("PHISH_TRIAGE_TIMEOUT") or 10)


def get(url: str) -> requests.Response:
    """GET from tranco-list.eu, waiting and retrying when the server answers HTTP 429."""
    for attempt in range(MAX_ATTEMPTS):
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=get_timeout())
        if response.status_code != 429:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After", "")
        wait = (
            min(int(retry_after), MAX_RETRY_AFTER) if retry_after.isdigit() else 10 * (attempt + 1)
        )
        print(f"Rate limited by tranco-list.eu; retrying in {wait}s")
        time.sleep(wait)
    raise requests.HTTPError(f"Still rate limited after {MAX_ATTEMPTS} attempts: {url}")


def main() -> int:
    """Download the requested (or latest) Tranco list unless it is already present."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list-id", help="Permanent Tranco list ID (default: latest).")
    parser.add_argument("--top", type=int, default=100_000, help="Number of domains.")
    args = parser.parse_args()

    try:
        list_id = args.list_id or get(f"{API}/top-1m-id").text.strip()
        dest = DATA_DIR / f"tranco_{list_id}_top{args.top}.csv"
        if dest.exists():
            print(f"Already present: {dest}")
            return 0

        metadata = get(f"{API}/api/lists/id/{list_id}").json()
        if not metadata.get("available"):
            print(f"Tranco list {list_id} is not available", file=sys.stderr)
            return 1
        csv_text = get(f"{API}/download/{list_id}/{args.top}").text
    except (requests.RequestException, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    partial.write_text(csv_text, encoding="utf-8")
    partial.replace(dest)
    dest.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = sum(1 for line in csv_text.splitlines() if line.strip())
    print(f"Saved Tranco list {list_id} ({rows:,} domains) to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
