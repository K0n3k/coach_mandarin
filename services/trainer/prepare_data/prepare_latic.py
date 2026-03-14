"""
Prepare LATIC dataset for training (Phase 3 — complement).

LATIC: Non-native pre-labelled Mandarin Chinese validation corpus.
4 speakers (L1: Korean, Chadian, Russian x2), ~4h.
Annotated for mispronunciation detection.

Structure after extraction:
  WAVA/SPEAKER0010.zip → WAVA/SPEAKER0010/*.wav
  WAVA/SPEAKER0011.zip → ...
  SCRIPT/{Original_Text_Script,Orignial_Notations,Testing_Notations,...}/*.TXT

Usage:
    python prepare_data/prepare_latic.py [--archive PATH]
"""

import argparse
import struct
import sys
import tarfile
import zipfile
from pathlib import Path

DATASET_DIR = Path("/app/dataset/latic")
DOWNLOAD_DIR = Path("/app/dataset/_downloads")


def extract(archive: Path, dest: Path) -> None:
    """Extract the outer archive, then nested per-speaker zips."""
    wav_count = len(list(dest.rglob("*.wav")))
    if wav_count > 100:
        print(f"  already extracted: {wav_count:,} wav files")
        return

    if not archive.exists():
        print(f"  ERROR: archive not found at {archive}")
        print(f"  Download manually from IEEE Dataport:")
        print(f"  https://ieee-dataport.org/open-access/latic-non-native-pre-labelled-mandarin-chinese-validation-corpus-automatic-speech")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    # Step 1: extract outer archive
    if not (dest / "WAVA").exists():
        print(f"  extracting {archive.name} ...")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(path=str(dest))
        elif archive.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive, "r:*") as tar:
                tar.extractall(path=str(dest))

    # Step 2: extract nested per-speaker zips in WAVA/
    wava_dir = dest / "WAVA"
    if wava_dir.exists():
        for spk_zip in sorted(wava_dir.glob("*.zip")):
            spk_name = spk_zip.stem  # SPEAKER0010
            spk_dir = wava_dir / spk_name
            if spk_dir.exists() and any(spk_dir.rglob("*.wav")):
                continue
            print(f"    extracting {spk_zip.name} ...")
            with zipfile.ZipFile(spk_zip, "r") as zf:
                zf.extractall(path=str(wava_dir))

    # Step 3: if there's a SCRIPT.zip, extract it too
    script_zip = dest / "SCRIPT.zip"
    if script_zip.exists() and not (dest / "SCRIPT").exists():
        print(f"    extracting SCRIPT.zip ...")
        with zipfile.ZipFile(script_zip, "r") as zf:
            zf.extractall(path=str(dest))


def verify(dest: Path) -> None:
    wava_dir = dest / "WAVA"

    total_wavs = 0
    speakers = []

    if wava_dir.exists():
        for spk_dir in sorted(wava_dir.iterdir()):
            if spk_dir.is_dir():
                wavs = list(spk_dir.rglob("*.wav"))
                if wavs:
                    speakers.append(spk_dir.name)
                    total_wavs += len(wavs)
                    # Check format of first wav
                    with open(wavs[0], "rb") as f:
                        h = f.read(44)
                        if len(h) >= 28 and h[:4] == b"RIFF":
                            sr = struct.unpack("<I", h[24:28])[0]
                            ch = struct.unpack("<H", h[22:24])[0]
                            print(f"  {spk_dir.name}: {len(wavs):,} wav — {sr}Hz {ch}ch")
                        else:
                            print(f"  {spk_dir.name}: {len(wavs):,} wav — unknown format")

    print(f"  total: {total_wavs:,} wav files, {len(speakers)} speakers")
    print(f"  speakers: {speakers}")

    # Check annotations
    script_dir = dest / "SCRIPT"
    if script_dir.exists():
        for sub in sorted(script_dir.iterdir()):
            if sub.is_dir():
                files = list(sub.glob("*.TXT"))
                print(f"  annotations/{sub.name}: {len(files)} files")

    # Check speaker metadata
    spk_file = dest / "Speaker.txt"
    if spk_file.exists():
        print(f"  speaker metadata: {spk_file.name}")

    if total_wavs < 1000:
        print(f"  WARNING: expected ~2579 wav files for LATIC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=False)
    args = parser.parse_args()

    print("=== LATIC preparation ===")

    archive = args.archive
    if archive is None:
        for name in ["latic.zip", "latic.tar.gz", "LATIC.zip", "LATIC Speech Corpus.zip"]:
            candidate = DOWNLOAD_DIR / name
            if candidate.exists():
                archive = candidate
                break

    if archive is None:
        # Check if already extracted
        if DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.wav")):
            print("  archive not found but data already extracted")
            verify(DATASET_DIR)
            print("=== LATIC done ===")
            return
        print("  ERROR: no archive specified and no default found")
        sys.exit(1)

    extract(archive, DATASET_DIR)
    verify(DATASET_DIR)

    print("=== LATIC done ===")


if __name__ == "__main__":
    main()
