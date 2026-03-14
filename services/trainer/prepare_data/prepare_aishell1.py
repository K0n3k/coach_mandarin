"""
Prepare AISHELL-1 dataset for training.

Downloads from OpenSLR, extracts nested per-speaker tar.gz files.
AISHELL-1 is 16kHz mono WAV — no resampling needed.

Official splits by speaker IDs (from aishell website/README):
  train: 340 speakers, dev: 40 speakers, test: 20 speakers

Usage:
    python prepare_data/prepare_aishell1.py [--skip-download]
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

DATASET_DIR = Path("/app/dataset/aishell1")
DOWNLOAD_DIR = Path("/app/dataset/_downloads")

URL = "https://www.openslr.org/resources/33/data_aishell.tgz"
ARCHIVE_NAME = "data_aishell.tgz"


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


def extract_aishell(archive: Path, dest: Path) -> None:
    """Extract the nested AISHELL archive: outer tgz → per-speaker tar.gz → wavs."""
    # Check if already fully extracted
    wav_count = len(list(dest.rglob("*.wav")))
    if wav_count > 100_000:
        print(f"  already extracted: {wav_count:,} wav files")
        return

    dest.mkdir(parents=True, exist_ok=True)

    # Step 1: extract outer archive → data_aishell/wav/S0NNN.tar.gz + transcript
    raw_root = dest / "data_aishell"
    if not raw_root.exists():
        print(f"  extracting outer archive {archive.name} ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=str(dest))

    # Step 2: extract each per-speaker tar.gz
    wav_dir = raw_root / "wav"
    if not wav_dir.exists():
        print(f"  ERROR: {wav_dir} not found after extraction")
        sys.exit(1)

    speaker_tars = sorted(wav_dir.glob("*.tar.gz"))
    print(f"  found {len(speaker_tars)} speaker archives to extract")

    for i, spk_tar in enumerate(speaker_tars, 1):
        spk_id = spk_tar.stem.replace(".tar", "")
        # Each speaker tar contains: data_aishell/wav/{train|dev|test}/S0NNN/*.wav
        with tarfile.open(spk_tar, "r:gz") as tar:
            tar.extractall(path=str(dest))
        spk_tar.unlink()  # remove inner tar
        if i % 50 == 0 or i == len(speaker_tars):
            print(f"    extracted {i}/{len(speaker_tars)} speakers")


def build_structure(dest: Path) -> None:
    """Move from data_aishell/wav/{split}/SNNNN/ to aishell1/{split}/SNNNN/."""
    raw_root = dest / "data_aishell"
    if not raw_root.exists():
        # Already restructured
        if (dest / "train").exists():
            return
        print("  no data_aishell/ dir — checking if already restructured")
        return

    wav_root = raw_root / "wav"
    for split in ["train", "dev", "test"]:
        src = wav_root / split
        dst = dest / split
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"  moved {split}/")

    # Copy transcript
    transcript_src = raw_root / "transcript" / "aishell_transcript_v0.8.txt"
    transcript_dst = dest / "transcript.txt"
    if transcript_src.exists() and not transcript_dst.exists():
        transcript_src.rename(transcript_dst)

    # Clean up raw structure
    import shutil
    if raw_root.exists():
        shutil.rmtree(raw_root, ignore_errors=True)


def verify(dest: Path) -> None:
    total = 0
    for split in ["train", "dev", "test"]:
        split_dir = dest / split
        if split_dir.exists():
            wavs = list(split_dir.rglob("*.wav"))
            spks = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"  {split}: {len(wavs):,} utterances, {len(spks)} speakers")
            total += len(wavs)
    print(f"  total: {total:,} utterances")

    transcript = dest / "transcript.txt"
    if transcript.exists():
        n = sum(1 for _ in open(transcript))
        print(f"  transcript: {n:,} lines")

    if total < 120_000:
        print("  WARNING: expected ~141k utterances for AISHELL-1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print("=== AISHELL-1 preparation ===")

    archive = DOWNLOAD_DIR / ARCHIVE_NAME

    if not args.skip_download:
        download(URL, archive)

    extract_aishell(archive, DATASET_DIR)
    build_structure(DATASET_DIR)
    verify(DATASET_DIR)

    print("=== AISHELL-1 done ===")


if __name__ == "__main__":
    main()
