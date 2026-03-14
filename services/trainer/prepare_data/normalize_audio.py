"""
Normalize audio files to 16kHz mono PCM WAV.

Scans a dataset directory for audio files and resamples anything
that isn't already 16kHz mono. Skips files already conforming.

Supports: WAV (any SR), MP3, FLAC, OGG, M4A.

Usage:
    python prepare_data/normalize_audio.py <dataset_dir> [--dry-run] [--workers N]
"""

import argparse
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TARGET_SR = 16000
TARGET_CH = 1


def check_wav(path: Path) -> tuple[int, int]:
    """Read WAV header, return (sample_rate, channels). Returns (0,0) on error."""
    try:
        with open(path, "rb") as f:
            header = f.read(44)
            if len(header) < 28:
                return 0, 0
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return 0, 0
            ch = struct.unpack("<H", header[22:24])[0]
            sr = struct.unpack("<I", header[24:28])[0]
            return sr, ch
    except Exception:
        return 0, 0


def needs_conversion(path: Path) -> bool:
    """Return True if the file needs resampling/conversion."""
    if path.suffix.lower() != ".wav":
        return True
    sr, ch = check_wav(path)
    return sr != TARGET_SR or ch != TARGET_CH


def convert_file(path: Path) -> tuple[str, bool]:
    """Convert a single file to 16kHz mono WAV in-place."""
    temp = path.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-ac", str(TARGET_CH),
                "-ar", str(TARGET_SR),
                "-acodec", "pcm_s16le",
                str(temp),
            ],
            capture_output=True,
            check=True,
        )
        # Replace original
        temp.rename(path.with_suffix(".wav"))
        # Remove original if different extension
        if path.suffix.lower() != ".wav" and path.exists():
            path.unlink()
        return str(path), True
    except subprocess.CalledProcessError as e:
        if temp.exists():
            temp.unlink()
        return str(path), False


def scan_files(root: Path) -> list[Path]:
    """Find all audio files recursively."""
    files = []
    for ext in AUDIO_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Normalize audio to 16kHz mono WAV")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't convert")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.dataset_dir
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    print(f"=== Audio normalization: {root} ===")

    files = scan_files(root)
    print(f"  found {len(files):,} audio files")

    to_convert = [f for f in files if needs_conversion(f)]
    already_ok = len(files) - len(to_convert)
    print(f"  {already_ok:,} already 16kHz mono WAV")
    print(f"  {len(to_convert):,} need conversion")

    if args.dry_run or not to_convert:
        if to_convert and args.dry_run:
            for f in to_convert[:10]:
                sr, ch = check_wav(f) if f.suffix.lower() == ".wav" else (0, 0)
                print(f"    {f.name}: {sr}Hz {ch}ch" if sr else f"    {f.name}: non-WAV")
            if len(to_convert) > 10:
                print(f"    ... and {len(to_convert)-10} more")
        return

    converted = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_file, f): f for f in to_convert}
        for i, fut in enumerate(as_completed(futures), 1):
            name, ok = fut.result()
            if ok:
                converted += 1
            else:
                failed += 1
            if i % 500 == 0 or i == len(to_convert):
                print(f"    {i:,}/{len(to_convert):,} — {converted:,} ok, {failed:,} failed")

    print(f"  done: {converted:,} converted, {failed:,} failed")
    print(f"=== Normalization complete ===")


if __name__ == "__main__":
    main()
