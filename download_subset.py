"""
Download a reproducible subset of the RSNA Knee Abnormality Detection dataset.

Pulls only N studies' worth of DICOM files (plus all the metadata CSVs) instead
of the full multi-GB archive, so you can develop and test your pipeline locally
before scaling up on Kaggle/Colab.

Requires: the `kaggle` CLI installed and authenticated (kaggle.json or
access_token already set up in ~/.kaggle/).

Usage:
    python download_subset.py --n-studies 15 --split train
    python download_subset.py --n-studies 15 --split train --dry-run
    python download_subset.py --n-studies 5 --split test --output data/test_sample
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"

# Metadata files we always want regardless of which studies we sample
METADATA_FILES = [
    "train.csv",
    "train_series.csv",
    "train_report.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
]


def run(cmd: list[str]) -> str:
    """Run a subprocess command and return stdout, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return result.stdout


def list_competition_files() -> list[dict]:
    """
    Get the FULL file listing from Kaggle, looping through pages.

    The plain `kaggle competitions files` CLI command only returns a single
    page of results, which silently truncates huge listings (this dataset
    has hundreds of thousands of files). We use the Python API directly so
    we can follow next_page_token until it's exhausted.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    print("Listing competition files (this can take a while, dataset has a LOT of files)...")
    api = KaggleApi()
    api.authenticate()

    all_files = []
    page_token = None
    page_num = 0
    while True:
        page_num += 1
        result = api.competition_list_files(COMPETITION, page_token=page_token)

        # The result object's attribute names have varied across kaggle
        # package versions. Handle a couple of possible shapes defensively.
        files_attr = getattr(result, "files", None)
        if files_attr is None:
            print("Unexpected response shape from competition_list_files().")
            print(f"Available attributes: {dir(result)}")
            raise SystemExit(1)

        for f in files_attr:
            name = getattr(f, "name", None) or getattr(f, "ref", None)
            size = getattr(f, "totalBytes", None) or getattr(f, "size", 0)
            if name:
                all_files.append({"name": name, "size": size})

        print(f"  page {page_num}: {len(files_attr)} files (running total: {len(all_files)})")

        page_token = getattr(result, "nextPageToken", None) or getattr(result, "next_page_token", None)
        if not page_token:
            break

    return all_files


def download_file(remote_name: str, output_root: Path, force: bool = False) -> None:
    """Download a single competition file, preserving its directory structure."""
    local_path = output_root / remote_name
    if local_path.exists() and not force:
        return  # already have it, skip
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-f", remote_name,
        "-p", str(local_path.parent),
        "--force",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-studies", type=int, default=15, help="Number of studies to pull")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output", default="data/sample", help="Local output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded, don't download")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    all_files = list_competition_files()
    print(f"Found {len(all_files)} total files in the competition.")

    # Isolate the DICOM files for the split we care about
    prefix = f"{args.split}_series/"
    series_files = [f for f in all_files if f["name"].startswith(prefix)]

    if not series_files:
        print(f"No files found under '{prefix}'. Check the exact folder name with:")
        print(f"  kaggle competitions files -c {COMPETITION}")
        raise SystemExit(1)

    # StudyInstanceUID is the first path segment after the prefix
    study_ids = sorted({f["name"][len(prefix):].split("/")[0] for f in series_files})
    print(f"Found {len(study_ids)} unique studies under '{prefix}'.")

    random.seed(args.seed)
    n = min(args.n_studies, len(study_ids))
    selected_studies = set(random.sample(study_ids, n))
    print(f"Selected {n} studies (seed={args.seed}):")
    for sid in sorted(selected_studies):
        print(f"  {sid}")

    files_to_get = [
        f for f in series_files
        if f["name"][len(prefix):].split("/")[0] in selected_studies
    ]
    total_bytes = sum(int(f["size"]) for f in files_to_get)
    print(f"\n{len(files_to_get)} DICOM files to download (~{total_bytes / 1e6:.1f} MB)")
    print(f"Plus {len(METADATA_FILES)} metadata CSVs.")

    if args.dry_run:
        print("\nDry run, nothing downloaded. Drop --dry-run to actually pull the data.")
        return

    print("\nDownloading metadata files...")
    for meta_file in METADATA_FILES:
        if any(f["name"] == meta_file for f in all_files):
            download_file(meta_file, output_root)
            print(f"  got {meta_file}")

    print("\nDownloading DICOM subset...")
    for i, f in enumerate(files_to_get, 1):
        download_file(f["name"], output_root)
        if i % 10 == 0 or i == len(files_to_get):
            print(f"  {i}/{len(files_to_get)}")

    print(f"\nDone. Data is in: {output_root.resolve()}")


if __name__ == "__main__":
    main()