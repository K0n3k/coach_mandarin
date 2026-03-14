"""
Standalone evaluation on test set.

Usage:
  python evaluate.py --checkpoint /checkpoints/best.pt --phase 1
  python evaluate.py --checkpoint /checkpoints/best.pt --phase 2
"""

import argparse
import os

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from dataset import MandarinDataset, build_phoneme_vocab, collate_fn
from metrics import TONE_NAMES, AccuracyTracker
from model import MandarinCoachModel
from train import get_manifests


def main():
    p = argparse.ArgumentParser(description="Evaluate Mandarin pronunciation model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--backbone", type=str, default="microsoft/wavlm-base")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    dataset_root = args.dataset_root or os.environ.get("DATASET_ROOT", "/app/dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp

    print(f"=== Evaluation Phase {args.phase} ===")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Phoneme vocab
    tts_train = get_manifests(dataset_root, 1, "train")
    phoneme_vocab = build_phoneme_vocab(tts_train)

    # Test dataset
    test_manifests = get_manifests(dataset_root, args.phase, "test")
    if not test_manifests:
        # Fall back to val if no test manifest
        test_manifests = get_manifests(dataset_root, args.phase, "val")
    print(f"  Test manifests: {test_manifests}")

    test_ds = MandarinDataset(
        test_manifests, dataset_root, args.phase, pinyin_to_phoneme_id=phoneme_vocab
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  Test samples: {len(test_ds):,}")

    # Load model
    model = MandarinCoachModel(
        backbone_name=args.backbone, num_phonemes=max(len(phoneme_vocab), 1)
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Evaluate
    tone_tracker = AccuracyTracker(5, TONE_NAMES)
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            lengths = batch["lengths"].to(device)
            tones = batch["tone"].to(device)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(input_values, lengths=lengths, phase=args.phase)

            if args.phase in (1, 2) and "tone_logits" in outputs:
                loss = criterion(outputs["tone_logits"], tones)
                total_loss += loss.item()
                tone_tracker.update(outputs["tone_logits"], tones)

    avg_loss = total_loss / max(len(test_loader), 1)

    print(f"\n=== Results ===")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {tone_tracker.accuracy:.1f}%")
    print(f"  Per-tone accuracy:")
    for name, acc in tone_tracker.per_class_accuracy().items():
        count = tone_tracker.per_class_total.get(
            TONE_NAMES.index(name) if name in TONE_NAMES else -1, 0
        )
        print(f"    {name}: {acc:.1f}% ({count} samples)")


if __name__ == "__main__":
    main()
