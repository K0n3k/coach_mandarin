"""
Prepare THCHS-30 dataset for training.

Downloads from OpenSLR, extracts. Already 16kHz mono WAV.
Official splits are preserved.

Usage:
    python prepare_data/prepare_thchs30.py [--skip-download]
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

DATASET_DIR = Path("/app/dataset/thchs30")
DOWNLOAD_DIR = Path("/app/dataset/_downloads")

# OpenSLR resources
URLS = {
    "data": "https://www.openslr.org/resources/18/data_thchs30.tgz",
}
# There's also test-noise.tgz and resource.tgz but we only need the clean data.


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return
    print(f"  downloading {url} ...")
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", str(dest), url],
        check=True,
    )


def extract(archive: Path, dest: Path) -> None:
    if dest.exists() and any(dest.rglob("*.wav")):
        n = len(list(dest.rglob("*.wav")))
        print(f"  already extracted: {n} wav files")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=str(dest))


def build_structure(dest: Path) -> None:
    """THCHS-30 extracts as data_thchs30/{train,dev,test}/*.wav + *.trn."""
    raw_root = dest / "data_thchs30"
    if not raw_root.exists():
        # Already restructured?
        if (dest / "train").exists():
            return
        print(f"  ERROR: {raw_root} not found")
        sys.exit(1)

    for split in ["train", "dev", "test"]:
        src = raw_root / split
        dst = dest / split
        if dst.exists():
            continue
        if src.exists():
            src.rename(dst)

    # Also move the data/ directory (contains wav references) if present
    data_dir = raw_root / "data"
    if data_dir.exists():
        dst_data = dest / "data"
        if not dst_data.exists():
            data_dir.rename(dst_data)

    # Clean up
    import shutil
    shutil.rmtree(raw_root, ignore_errors=True)


def verify(dest: Path) -> None:
    total = 0
    for split in ["train", "dev", "test"]:
        split_dir = dest / split
        if split_dir.exists():
            wavs = list(split_dir.rglob("*.wav"))
            trns = list(split_dir.rglob("*.trn"))
            print(f"  {split}: {len(wavs):,} wav, {len(trns):,} transcripts")
            total += len(wavs)

    # Also check data/ dir (THCHS-30 stores actual wavs there, splits have symlinks)
    data_dir = dest / "data"
    if data_dir.exists():
        data_wavs = list(data_dir.glob("*.wav"))
        print(f"  data/: {len(data_wavs):,} wav files")
        if total == 0:
            total = len(data_wavs)

    print(f"  total: {total:,} wav files")
    if total < 10_000:
        print("  WARNING: expected ~13k utterances for THCHS-30")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print("=== THCHS-30 preparation ===")

    archive = DOWNLOAD_DIR / "data_thchs30.tgz"

    if not args.skip_download:
        download(URLS["data"], archive)

    extract(archive, DATASET_DIR)
    build_structure(DATASET_DIR)
    verify(DATASET_DIR)

    print("=== THCHS-30 done ===")


if __name__ == "__main__":
    main()
