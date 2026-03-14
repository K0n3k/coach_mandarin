"""
Prepare Common Voice zh-CN dataset for training.

Handles extraction and MP3→WAV 16kHz mono conversion.
Conversion is done in batches to control memory usage.

Usage:
    python prepare_data/prepare_cv_zh.py [--archive PATH] [--workers N]
"""

import argparse
import csv
import subprocess
import sys
import tarfile
from pathlib import Path

DATASET_DIR = Path("/app/dataset/cv_zh")
DOWNLOAD_DIR = Path("/app/dataset/_downloads")
DEFAULT_ARCHIVE = DOWNLOAD_DIR / "cv-corpus-zh-CN.tar.gz"


def extract(archive: Path, dest: Path) -> None:
    # Quick check: look for clips dir directly (known path)
    for d in dest.iterdir() if dest.exists() else []:
        if d.is_dir():
            clips = d / "zh-CN" / "clips"
            if clips.exists():
                print(f"  already extracted (clips dir exists)")
                return

    if not archive.exists():
        print(f"  ERROR: archive not found at {archive}")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} (streaming, this takes a while) ...")
    # Use tar command instead of Python tarfile to avoid OOM on large archives
    subprocess.run(
        ["tar", "xzf", str(archive), "-C", str(dest)],
        check=True,
    )


def find_clips_dir(dest: Path) -> Path:
    """Find clips/ dir using known Common Voice structure."""
    # Known pattern: cv-corpus-XX.X-YYYY-MM-DD/zh-CN/clips/
    for d in sorted(dest.iterdir()):
        if d.is_dir() and d.name.startswith("cv-corpus"):
            clips = d / "zh-CN" / "clips"
            if clips.exists():
                return clips
    # Fallback
    clips = dest / "clips"
    if clips.exists():
        return clips
    print("  ERROR: could not find clips/ directory")
    sys.exit(1)


def find_tsv(dest: Path, name: str) -> Path | None:
    """Find a specific TSV using known structure."""
    for d in sorted(dest.iterdir()):
        if d.is_dir() and d.name.startswith("cv-corpus"):
            tsv = d / "zh-CN" / f"{name}.tsv"
            if tsv.exists():
                return tsv
    return None


def convert_one(mp3: Path, wav: Path) -> bool:
    """Convert a single MP3 to 16kHz mono WAV."""
    if wav.exists():
        return False
    wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3),
             "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
             str(wav)],
            capture_output=True, check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def build_structure(dest: Path, workers: int) -> None:
    """Convert MP3s to WAV 16kHz, organized by split."""
    clips_dir = find_clips_dir(dest)
    print(f"  clips dir: {clips_dir}")

    wav_root = dest / "wav"
    wav_root.mkdir(exist_ok=True)

    # Build tasks from TSV files (memory-efficient: one split at a time)
    total_converted = 0
    total_skipped = 0

    for split_name in ["train", "dev", "test"]:
        tsv_path = find_tsv(dest, split_name)
        if tsv_path is None:
            print(f"  WARNING: {split_name}.tsv not found")
            continue

        # Count entries and collect tasks
        tasks: list[tuple[Path, Path]] = []
        skipped = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                mp3_name = row.get("path", "")
                if not mp3_name:
                    continue
                mp3_path = clips_dir / mp3_name
                wav_name = mp3_name.replace(".mp3", ".wav")
                wav_path = wav_root / split_name / wav_name
                if wav_path.exists():
                    skipped += 1
                elif mp3_path.exists():
                    tasks.append((mp3_path, wav_path))

        print(f"  {split_name}: {skipped:,} already done, {len(tasks):,} to convert")
        total_skipped += skipped

        if not tasks:
            continue

        # Convert in batches to limit memory
        BATCH = 500
        converted = 0
        for batch_start in range(0, len(tasks), BATCH):
            batch = tasks[batch_start:batch_start + BATCH]

            if workers > 1:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futs = {pool.submit(convert_one, m, w): i for i, (m, w) in enumerate(batch)}
                    for fut in as_completed(futs):
                        if fut.result():
                            converted += 1
            else:
                for mp3, wav in batch:
                    if convert_one(mp3, wav):
                        converted += 1

            done = batch_start + len(batch)
            if done % 5000 < BATCH or done == len(tasks):
                print(f"    {split_name}: {done:,}/{len(tasks):,} ({converted:,} converted)")

        total_converted += converted

    print(f"  total: {total_converted:,} converted, {total_skipped:,} skipped")


def verify(dest: Path) -> None:
    wav_root = dest / "wav"
    total = 0
    for split in ["train", "dev", "test"]:
        split_dir = wav_root / split
        if split_dir.exists():
            # Count without loading full list into memory
            n = sum(1 for _ in split_dir.rglob("*.wav"))
            print(f"  {split}: {n:,} wav files")
            total += n
    print(f"  total: {total:,} wav files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    print("=== Common Voice zh-CN preparation ===")

    archive = args.archive
    if archive is None:
        for candidate in sorted(DOWNLOAD_DIR.glob("cv-corpus*zh-CN*.tar.gz")):
            archive = candidate
            break
    if archive is None:
        archive = DEFAULT_ARCHIVE

    extract(archive, DATASET_DIR)
    build_structure(DATASET_DIR, args.workers)
    verify(DATASET_DIR)

    print("=== Common Voice zh-CN done ===")


if __name__ == "__main__":
    main()
