"""
Generate training manifests (train/val/test splits) for all datasets.

Creates a JSON manifest per dataset per split with entries:
  { "path": "relative/to/dataset.wav", "speaker": "...", "text": "...", "tone": N }

Respects official splits where available (AISHELL-1, THCHS-30, CV zh-CN).
Speaker-independent splits for iCALL and LATIC.
Voice-based splits for TTS phase 1 (5 train / 1 val).

Usage:
    python prepare_data/gen_manifests.py [--dataset NAME] [--all]
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

DATASET_ROOT = Path("/app/dataset")
MANIFEST_DIR = DATASET_ROOT / "manifests"

random.seed(42)


def write_manifest(name: str, split: str, entries: list[dict]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{name}_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    print(f"  {name}/{split}: {len(entries):,} entries → {out.name}")


# ── TTS Phase 1 ─────────────────────────────────────────────

def gen_tts_phase1():
    """Split by voice: 5 train, 1 val (Yunyang held out)."""
    root = DATASET_ROOT / "tts_phase1"
    csv_path = root / "syllables.csv"

    if not csv_path.exists():
        print("  WARNING: tts_phase1/syllables.csv not found, skipping")
        return

    # Read syllable list
    syllables = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    hanzi, tone, pinyin = row[0], int(row[1]), row[2]
                except ValueError:
                    continue  # skip header
                syllables.append((hanzi, tone, pinyin))

    voices = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not voices:
        print("  WARNING: no voice directories in tts_phase1/")
        return

    print(f"  TTS phase 1: {len(syllables)} syllables × {len(voices)} voices")

    # Hold out Yunyang for val, rest for train
    val_voice = "Yunyang"
    train_voices = [v for v in voices if v != val_voice]

    train_entries = []
    val_entries = []

    for voice in voices:
        entries = []
        for hanzi, tone, pinyin in syllables:
            wav = root / voice / f"{pinyin}.wav"
            if wav.exists():
                entries.append({
                    "path": f"tts_phase1/{voice}/{pinyin}.wav",
                    "speaker": voice,
                    "text": hanzi,
                    "tone": tone,
                    "pinyin": pinyin,
                })

        if voice == val_voice:
            val_entries.extend(entries)
        else:
            train_entries.extend(entries)

    write_manifest("tts_phase1", "train", train_entries)
    write_manifest("tts_phase1", "val", val_entries)


# ── AISHELL-1 ────────────────────────────────────────────────

def gen_aishell1():
    """Use official splits (train/dev/test → train/val/test)."""
    root = DATASET_ROOT / "aishell1"

    # Read transcript
    transcript = {}
    transcript_path = root / "transcript.txt"
    if transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    transcript[parts[0]] = parts[1]

    split_map = {"train": "train", "dev": "val", "test": "test"}

    for src_name, dst_name in split_map.items():
        split_dir = root / src_name
        if not split_dir.exists():
            print(f"  WARNING: aishell1/{src_name} not found")
            continue

        entries = []
        for wav in sorted(split_dir.rglob("*.wav")):
            utt_id = wav.stem
            spk = wav.parent.name
            text = transcript.get(utt_id, "")
            entries.append({
                "path": f"aishell1/{src_name}/{spk}/{wav.name}",
                "speaker": spk,
                "text": text,
                "utt_id": utt_id,
            })

        write_manifest("aishell1", dst_name, entries)


# ── THCHS-30 ─────────────────────────────────────────────────

def gen_thchs30():
    """Use official splits."""
    root = DATASET_ROOT / "thchs30"

    split_map = {"train": "train", "dev": "val", "test": "test"}

    for src_name, dst_name in split_map.items():
        split_dir = root / src_name
        if not split_dir.exists():
            # Check if wavs are in data/ and splits are symlinks
            split_dir = root / src_name
            if not split_dir.exists():
                print(f"  WARNING: thchs30/{src_name} not found")
                continue

        entries = []
        for wav in sorted(split_dir.rglob("*.wav")):
            utt_id = wav.stem
            # Speaker from filename pattern: A11_0, B22_1, etc.
            spk = utt_id.split("_")[0] if "_" in utt_id else "unknown"

            # Try to find transcript (.trn file, same name)
            trn = wav.with_suffix(".wav.trn")
            text = ""
            if trn.exists():
                with open(trn, "r", encoding="utf-8") as f:
                    text = f.readline().strip()

            entries.append({
                "path": str(wav.relative_to(DATASET_ROOT)),
                "speaker": spk,
                "text": text,
                "utt_id": utt_id,
            })

        write_manifest("thchs30", dst_name, entries)


# ── Common Voice zh-CN ────────────────────────────────────────

def gen_cv_zh():
    """Use official splits from converted WAV files."""
    root = DATASET_ROOT / "cv_zh"
    wav_root = root / "wav"

    if not wav_root.exists():
        print("  WARNING: cv_zh/wav/ not found (run prepare_cv_zh.py first)")
        return

    split_map = {"train": "train", "dev": "val", "test": "test"}

    # Try to read TSV for text info
    tsvs = {}
    for name in ["train", "dev", "test"]:
        for tsv in root.rglob(f"{name}.tsv"):
            tsvs[name] = tsv

    # Build text lookup from TSVs
    text_map: dict[str, str] = {}
    speaker_map: dict[str, str] = {}
    for name, tsv_path in tsvs.items():
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                mp3 = row.get("path", "")
                wav_name = mp3.replace(".mp3", ".wav")
                text_map[wav_name] = row.get("sentence", "")
                speaker_map[wav_name] = row.get("client_id", "")[:8]

    for src_name, dst_name in split_map.items():
        split_dir = wav_root / src_name
        if not split_dir.exists():
            print(f"  WARNING: cv_zh/wav/{src_name} not found")
            continue

        entries = []
        for wav in sorted(split_dir.rglob("*.wav")):
            fname = wav.name
            entries.append({
                "path": f"cv_zh/wav/{src_name}/{fname}",
                "speaker": speaker_map.get(fname, "unknown"),
                "text": text_map.get(fname, ""),
            })

        write_manifest("cv_zh", dst_name, entries)


# ── iCALL ─────────────────────────────────────────────────────

def gen_icall():
    """Speaker-independent split: 80/10/10 by speaker."""
    root = DATASET_ROOT / "icall"
    if not root.exists():
        print("  WARNING: icall/ not found")
        return

    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        print("  WARNING: no wav files in icall/")
        return

    # Group by speaker (directory or filename prefix)
    speakers: dict[str, list[Path]] = {}
    for w in wavs:
        parts = w.relative_to(root).parts
        spk = parts[0] if len(parts) > 1 else w.stem.split("_")[0]
        speakers.setdefault(spk, []).append(w)

    spk_list = sorted(speakers.keys())
    random.shuffle(spk_list)

    n = len(spk_list)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)

    split_spk = {
        "test": spk_list[:n_test],
        "val": spk_list[n_test:n_test + n_val],
        "train": spk_list[n_test + n_val:],
    }

    print(f"  iCALL: {len(wavs):,} files, {n} speakers")
    print(f"    train: {len(split_spk['train'])} spk, val: {len(split_spk['val'])} spk, test: {len(split_spk['test'])} spk")

    for split_name, spks in split_spk.items():
        entries = []
        for spk in spks:
            for wav in speakers[spk]:
                entries.append({
                    "path": str(wav.relative_to(DATASET_ROOT)),
                    "speaker": spk,
                    "utt_id": wav.stem,
                })
        write_manifest("icall", split_name, entries)


# ── LATIC ─────────────────────────────────────────────────────

def gen_latic():
    """Speaker-independent split: 2 train, 1 val, 1 test (4 speakers)."""
    root = DATASET_ROOT / "latic"
    wava = root / "WAVA"
    if not wava.exists():
        print("  WARNING: latic/WAVA/ not found")
        return

    # Load text scripts for utterance text
    text_map: dict[str, str] = {}
    for script_dir_name in ["Original_Text_Script", "Testing_Text_Script"]:
        script_dir = root / "SCRIPT" / script_dir_name
        if script_dir.exists():
            for txt_file in script_dir.glob("*.TXT"):
                with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if "\t" in line:
                            utt_id, text = line.split("\t", 1)
                            text_map[utt_id] = text

    # Group wavs by speaker directory
    speakers: dict[str, list[Path]] = {}
    for spk_dir in sorted(wava.iterdir()):
        if not spk_dir.is_dir():
            continue
        wavs = sorted(spk_dir.rglob("*.wav"))
        if wavs:
            speakers[spk_dir.name] = wavs

    spk_list = sorted(speakers.keys())
    total = sum(len(v) for v in speakers.values())
    print(f"  LATIC: {total:,} files, {len(spk_list)} speakers: {spk_list}")

    # 4 speakers: 2 train, 1 val, 1 test — deterministic split
    # SPEAKER0010=Korean, 0011=Chadian, 0012=Russian, 0013=Russian
    if len(spk_list) >= 4:
        split_spk = {
            "train": [spk_list[0], spk_list[1]],  # 0010, 0011
            "val": [spk_list[2]],                   # 0012
            "test": [spk_list[3]],                  # 0013
        }
    else:
        split_spk = {"train": spk_list, "val": [], "test": []}

    for split_name, spks in split_spk.items():
        entries = []
        for spk in spks:
            for wav in speakers[spk]:
                utt_id = wav.stem
                entries.append({
                    "path": str(wav.relative_to(DATASET_ROOT)),
                    "speaker": spk,
                    "utt_id": utt_id,
                    "text": text_map.get(utt_id, ""),
                })
        write_manifest("latic", split_name, entries)


# ── Main ──────────────────────────────────────────────────────

GENERATORS = {
    "tts_phase1": gen_tts_phase1,
    "aishell1": gen_aishell1,
    "thchs30": gen_thchs30,
    "cv_zh": gen_cv_zh,
    "icall": gen_icall,
    "latic": gen_latic,
}


def main():
    parser = argparse.ArgumentParser(description="Generate training manifests")
    parser.add_argument("--dataset", type=str, choices=list(GENERATORS.keys()),
                        help="Generate manifest for a single dataset")
    parser.add_argument("--all", action="store_true", help="Generate all manifests")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("Specify --dataset NAME or --all")
        sys.exit(1)

    print(f"=== Manifest generation ===")

    if args.all:
        for name, gen in GENERATORS.items():
            print(f"\n--- {name} ---")
            gen()
    else:
        GENERATORS[args.dataset]()

    print(f"\n=== Done — manifests in {MANIFEST_DIR} ===")


if __name__ == "__main__":
    main()
