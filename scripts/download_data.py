"""Download the Transfermarkt dataset tables used by this project into data/raw/.

Source: https://github.com/dcaribou/transfermarkt-datasets (CC0).
Note: upstream updates are paused; valuations run to 2026-06-12.
"""
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["players", "player_valuations", "appearances", "games", "clubs", "competitions"]
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        dest = RAW_DIR / f"{table}.csv.gz"
        if dest.exists():
            print(f"skip {dest.name} (already downloaded)")
            continue
        print(f"downloading {dest.name} ...")
        # Use certifi's CA bundle: python.org macOS builds often lack system certificates.
        context = ssl.create_default_context(cafile=certifi.where())
        # The host rejects Python's default User-Agent with 403.
        request = urllib.request.Request(f"{BASE_URL}/{table}.csv.gz", headers={"User-Agent": "transfer-market/0.1"})
        with urllib.request.urlopen(request, context=context) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    print("done")


if __name__ == "__main__":
    main()
