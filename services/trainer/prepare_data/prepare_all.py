#!/usr/bin/env python3
"""
Master script: prepare all datasets for the 3-phase training pipeline.

Runs each preparation step in order:
  1. Normalize TTS phase 1 audio (24kHz → 16kHz)
  2. Download + extract AISHELL-1
  3. Download + extract THCHS-30
  4. Extract + convert Common Voice zh-CN (manual download required)
  5. Download + extract iCALL
  6. Extract LATIC (manual download required)
  7. Normalize all audio to 16kHz mono
  8. Generate manifests for all datasets

Usage:
    python prepare_data/prepare_all.py [--skip-download] [--only STEP]

Steps: tts, aishell1, thchs30, cv_zh, icall, latic, normalize, manifests
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATASET_ROOT = Path("/app/dataset")


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download steps (use existing archives)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this step")
    args = parser.parse_args()

    py = sys.executable
    skip = ["--skip-download"] if args.skip_download else []

    steps = {
        "aishell1": [py, str(SCRIPT_DIR / "prepare_aishell1.py")] + skip,
        "thchs30": [py, str(SCRIPT_DIR / "prepare_thchs30.py")] + skip,
        "cv_zh": [py, str(SCRIPT_DIR / "prepare_cv_zh.py")],
        "icall": [py, str(SCRIPT_DIR / "prepare_icall.py")] + skip,
        "latic": [py, str(SCRIPT_DIR / "prepare_latic.py")],
        "normalize": [py, str(SCRIPT_DIR / "normalize_audio.py"), str(DATASET_ROOT)],
        "manifests": [py, str(SCRIPT_DIR / "gen_manifests.py"), "--all"],
    }

    if args.only:
        if args.only not in steps:
            print(f"Unknown step: {args.only}")
            print(f"Available: {', '.join(steps.keys())}")
            sys.exit(1)
        ok = run_step(args.only, steps[args.only])
        sys.exit(0 if ok else 1)

    # Run all steps
    failed = []
    for name, cmd in steps.items():
        ok = run_step(name, cmd)
        if not ok:
            failed.append(name)
            # Non-critical: continue with other datasets
            # (cv_zh and latic may fail if not manually downloaded)

    print(f"\n{'='*60}")
    if failed:
        print(f"  COMPLETED with {len(failed)} failures: {', '.join(failed)}")
        print(f"  (cv_zh and latic require manual download)")
    else:
        print(f"  ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
