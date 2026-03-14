"""
Prepare iCALL dataset for training (Phase 3 — main dataset).

iCALL: L2 Mandarin speech by European speakers, phonetically/tonally annotated.
Available on OpenSLR for free commercial use.

Usage:
    python prepare_data/prepare_icall.py [--skip-download]
"""

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

DATASET_DIR = Path("/app/dataset/icall")
DOWNLOAD_DIR = Path("/app/dataset/_downloads")

# iCALL on OpenSLR — the exact resource ID may vary, check https://openslr.org/
# iCALL is typically distributed as a set of zip/tar files.
# Update this URL once the exact download link is confirmed.
ICALL_URL = "https://www.openslr.org/resources/117/iCALL.tar.gz"
ARCHIVE_NAME = "iCALL.tar.gz"


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

    if not archive.exists():
        print(f"  ERROR: archive not found at {archive}")
        print(f"  Try downloading from https://openslr.org/ (search for iCALL)")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} ...")

    if archive.suffix == ".gz" or archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(path=str(dest))
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(path=str(dest))
    else:
        print(f"  ERROR: unknown archive format: {archive.name}")
        sys.exit(1)


def find_wavs(dest: Path) -> list[Path]:
    """Recursively find all wav files."""
    return sorted(dest.rglob("*.wav"))


def find_annotations(dest: Path) -> dict[str, Path]:
    """Find annotation files (CSV, TSV, TextGrid, etc.)."""
    result = {}
    for ext in ["*.csv", "*.tsv", "*.txt", "*.TextGrid", "*.json"]:
        for f in dest.rglob(ext):
            result[f.name] = f
    return result


def verify(dest: Path) -> None:
    wavs = find_wavs(dest)
    annotations = find_annotations(dest)

    print(f"  wav files: {len(wavs):,}")
    print(f"  annotation files: {len(annotations):,}")

    if annotations:
        print(f"  annotation types: {', '.join(sorted(set(f.suffix for f in annotations.values())))}")

    # Check audio format of first file
    if wavs:
        import struct
        with open(wavs[0], "rb") as f:
            riff = f.read(44)
            if len(riff) >= 28:
                sr = struct.unpack("<I", riff[24:28])[0]
                ch = struct.unpack("<H", riff[22:24])[0]
                print(f"  sample audio: {wavs[0].name} — {sr}Hz, {ch}ch")

    # Try to count speakers
    speakers = set()
    for w in wavs:
        # Common patterns: speaker ID as directory name or filename prefix
        parts = w.relative_to(dest).parts
        if len(parts) > 1:
            speakers.add(parts[0])
        else:
            # Try filename prefix (e.g., SPK001_utt001.wav)
            prefix = w.stem.split("_")[0]
            speakers.add(prefix)

    if speakers:
        print(f"  estimated speakers: {len(speakers):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Path to manually downloaded archive")
    args = parser.parse_args()

    print("=== iCALL preparation ===")

    if args.archive:
        archive = args.archive
    else:
        archive = DOWNLOAD_DIR / ARCHIVE_NAME

    if not args.skip_download and not args.archive:
        download(ICALL_URL, archive)

    extract(archive, DATASET_DIR)
    verify(DATASET_DIR)

    print("=== iCALL done ===")


if __name__ == "__main__":
    main()
