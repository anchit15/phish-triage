"""Download the PhiUSIIL Phishing URL Dataset (UCI id 967, CC BY 4.0) into data/."""

import io
import os
import sys
import zipfile
from pathlib import Path

import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USER_AGENT = "phish-triage/0.1 (+https://github.com/anchit15/phish-triage)"


def get_timeout() -> float:
    """Return the network timeout in seconds from PHISH_TRIAGE_TIMEOUT (default 10)."""
    return float(os.environ.get("PHISH_TRIAGE_TIMEOUT") or 10)


def download_dataset(dest: Path) -> None:
    """Download the dataset zip and write its CSV to ``dest``.

    The CSV is written to a temporary ``.part`` file and renamed only when complete,
    so an interrupted run never leaves a partial file that later runs would skip.
    """
    response = requests.get(DATASET_URL, headers={"User-Agent": USER_AGENT}, timeout=get_timeout())
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_bytes = archive.read(CSV_NAME)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    partial.write_bytes(csv_bytes)
    partial.replace(dest)


def main() -> int:
    """Download the dataset unless it is already present."""
    dest = DATA_DIR / CSV_NAME
    if dest.exists():
        print(f"Already present: {dest}")
        return 0

    print(f"Downloading {DATASET_URL}")
    try:
        download_dataset(dest)
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    size_mb = dest.stat().st_size / 1_000_000
    print(f"Saved {dest} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
