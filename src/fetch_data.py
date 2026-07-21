"""
fetch_data.py
=============
Downloads the four Kenneth French Data Library files used by this
replication package, unzips them into data/, and writes a SHA-256 checksum
manifest to data/CHECKSUMS.sha256.

Kenneth French updates these files monthly. A freshly downloaded file's
hash will therefore legitimately differ from the VINTAGE_SHA256 values
below once the library has moved past the vintage used in the paper -- a
mismatch is a WARNING, not a failure. The vintage hashes are the exact
bytes used to produce the numbers in the manuscript; see README.md for
the sample-period consequences of a later vintage (in particular,
Portfolios_Formed_on_ME.csv is ~2-5 months behind the factor files, and
the analysis sample is bounded by whichever file is shortest).

F-F_Research_Data_5_Factors_2x3.csv is a real dependency, not an unused
extra: src/make_figure3.py falls back to it for pre-1963 dates, and
src/robustnesstests.py's FF5 robustness test (t9_ff5) reads it directly.

Usage:
    python src/fetch_data.py                 # fetch + checksum everything
    python src/fetch_data.py --checksum-only  # just checksum what's already in data/
"""
import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (zip filename on the French site, csv filename inside the zip)
FILES = [
    ("F-F_Research_Data_Factors_CSV.zip", "F-F_Research_Data_Factors.csv"),
    ("F-F_Momentum_Factor_CSV.zip", "F-F_Momentum_Factor.csv"),
    ("Portfolios_Formed_on_ME_CSV.zip", "Portfolios_Formed_on_ME.csv"),
    ("F-F_Research_Data_5_Factors_2x3_CSV.zip", "F-F_Research_Data_5_Factors_2x3.csv"),
]

# SHA-256 of the exact file vintages used to produce the manuscript's tables
# and figures. Not all downloaded on the same date: Portfolios_Formed_on_ME,
# F-F_Research_Data_Factors, and F-F_Momentum_Factor were pulled 2026-06-09;
# F-F_Research_Data_5_Factors_2x3 is an older pull from 2026-01-25 that was
# not refreshed alongside the other three. See README.md "Data" section for
# the corresponding sample-period detail.
VINTAGE_SHA256 = {
    "F-F_Research_Data_Factors.csv": "b0673efb39d1180605f117cb17e7747648ddab041031b5a01073c65416431f58",
    "F-F_Momentum_Factor.csv": "cd2429809e081485a5a62860de64af090948c26b835b92085dcfbb0e08de8663",
    "Portfolios_Formed_on_ME.csv": "a9e9adac1544fb3b1bad750208f56ddd32a2768882673dcf96953d474ed8d7ec",
    "F-F_Research_Data_5_Factors_2x3.csv": "a04a3ee9e25f5b271340ad262b5b4effe8712d7a2fed06bae5e68c8a5a73fba0",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(zip_name, csv_name):
    url = f"{BASE_URL}/{zip_name}"
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(n for n in zf.namelist() if n.lower() == csv_name.lower())
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / csv_name
        with zf.open(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())

    digest = sha256_of(out_path)
    expected = VINTAGE_SHA256.get(csv_name)
    print(f"  saved {out_path}")
    print(f"  SHA-256: {digest}")
    if expected and digest != expected:
        print(f"  WARNING: hash differs from the vintage used in the manuscript "
              f"({expected}). This is expected if French has published a newer "
              f"vintage since 2026-06-09 -- see README.md.")
    elif expected:
        print("  matches manuscript vintage.")
    return digest


def write_checksums():
    manifest_path = DATA_DIR / "CHECKSUMS.sha256"
    lines = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        sha = sha256_of(path)
        lines.append(f"{sha}  {path.name}")
        print(f"{sha[:16]}...  {path.name}")

    if not lines:
        print("No CSV files found in data/ to checksum.", file=sys.stderr)
        return

    manifest_path.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksum-only", action="store_true",
                         help="Skip downloading; just checksum whatever is already in data/")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not args.checksum_only:
        for zip_name, csv_name in FILES:
            fetch_one(zip_name, csv_name)
            print()

    write_checksums()
    print("Done. All scripts under src/ read these files from data/.")


if __name__ == "__main__":
    main()
