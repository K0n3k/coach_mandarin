"""
Main training script for Mandarin pronunciation model.

Supports all 3 phases:
  python train.py --phase 1
  python train.py --phase 2 --resume /checkpoints/wavlm-scorer-ph1-ep20-acc87.2.pt
  python train.py --phase 3 --resume /checkpoints/wavlm-scorer-ph2-ep30-acc76.1.pt
"""

import argparse
import math
import os
import time
import uuid
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from curriculum import (
    freeze_backbone,
    get_param_groups,
    print_trainable_summary,
    unfreeze_all,
    unfreeze_last_n_layers,
)
from dataset import MandarinDataset, build_phoneme_vocab, collate_fn
from metrics import TONE_NAMES, AccuracyTracker
from model import MandarinCoachModel
from watch_training import TrainingWatcher
from ws_reporter import WsReporter

# ── Phase defaults ──────────────────────────────────────────

PHASE_DEFAULTS = {
    1: {
        "epochs": 20,
        "batch_size": 32,
        "lr": 1e-3,
        "lr_backbone": None,
        "lr_head": None,
        "warmup": 200,
        "grad_clip": None,
        "patience": None,
    },
    2: {
        "epochs": 30,
        "batch_size": 32,
        "lr": None,
        "lr_backbone": 1e-4,
        "lr_head": 5e-4,
        "warmup": 500,
        "grad_clip": 1.0,
        "patience": 5,
    },
    3: {
        "epochs": 50,
        "batch_size": 16,
        "lr": 1e-5,
        "lr_backbone": None,
        "lr_head": None,
        "warmup": 0,
        "grad_clip": 0.5,
        "patience": 8,
    },
}


# ── GPU stats ───────────────────────────────────────────────


def get_gpu_stats() -> dict:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        return {
            "util_pct": util.gpu,
            "temp_c": temp,
            "vram_used_gb": round(mem.used / 1e9, 1),
            "vram_total_gb": round(mem.total / 1e9, 1),
            "power_w": int(power),
        }
    except Exception:
        return {
            "util_pct": 0,
            "temp_c": 0,
            "vram_used_gb": 0,
            "vram_total_gb": 0,
            "power_w": 0,
        }


# ── Manifests per phase ────────────────────────────────────


def get_manifests(dataset_root: str, phase: int, split: str) -> list[str]:
    root = Path(dataset_root) / "manifests"
    # Map val -> the actual manifest suffix
    split_map = {"train": "train", "val": "val"}
    s = split_map.get(split, split)

    if phase == 1:
        return [str(root / f"tts_phase1_{s}.json")]
    elif phase == 2:
        manifests = [str(root / f"tts_phase1_{s}.json")]
        for ds in ["aishell1", "thchs30", "cv_zh"]:
            p = root / f"{ds}_{s}.json"
            if p.exists():
                manifests.append(str(p))
        return manifests
    else:
        manifests = []
        for ds in ["latic", "tts_phase1", "aishell1"]:
            p = root / f"{ds}_{s}.json"
            if p.exists():
                manifests.append(str(p))
        return manifests


# ── Checkpoint ──────────────────────────────────────────────


def save_checkpoint(
    model, optimizer, epoch, phase, val_acc, val_loss, config, ckpt_dir
) -> str:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"wavlm-scorer-ph{phase}-ep{epoch}-acc{val_acc:.1f}.pt"
    path = ckpt_dir / name
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )
    print(f"  Checkpoint saved: {name}")
    return str(path)


def load_checkpoint(path, model, optimizer=None):
    print(f"  Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            print("  [warn] Optimizer state mismatch, starting fresh")
    return ckpt


# ── Scheduler ───────────────────────────────────────────────


def build_scheduler(optimizer, total_steps, warmup_steps, phase):
    if phase in (1, 2):

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"
    else:
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5
            ),
            "epoch",
        )


# ── Training loop ───────────────────────────────────────────


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    scheduler,
    scheduler_mode,
    criterion_tone,
    criterion_phoneme,
    phase,
    epoch,
    total_epochs,
    device,
    use_amp,
    grad_clip,
    reporter,
    report_every,
    tone_tracker,
):
    model.train()
    total_loss = 0.0
    loss_window = deque(maxlen=5)
    steps_per_epoch = len(loader)
    epoch_start = time.time()

    for step, batch in enumerate(loader):
        step_start = time.time()
        input_values = batch["input_values"].to(device)
        lengths = batch["lengths"].to(device)
        tones = batch["tone"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(input_values, lengths=lengths, phase=phase)

            if phase == 1:
                loss = criterion_tone(outputs["tone_logits"], tones)
            elif phase == 2:
                loss_t = criterion_tone(outputs["tone_logits"], tones)
                phoneme_ids = batch["phoneme_id"].to(device)
                loss_p = criterion_phoneme(outputs["phoneme_logits"], phoneme_ids)
                # Only count losses where we have valid labels
                has_tone = (tones >= 0).any()
                has_phoneme = (phoneme_ids >= 0).any()
                if has_tone and has_phoneme:
                    loss = 0.6 * loss_t + 0.4 * loss_p
                elif has_tone:
                    loss = loss_t
                elif has_phoneme:
                    loss = loss_p
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Phase 3 stub
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            ).item()
        else:
            grad_norm = 0.0

        scaler.step(optimizer)
        scaler.update()

        if scheduler_mode == "step":
            scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        loss_window.append(loss_val)

        # Track tone accuracy
        if phase in (1, 2) and "tone_logits" in outputs:
            tone_tracker.update(
                outputs["tone_logits"].detach(), tones
            )

        # Report step
        if reporter and (step + 1) % report_every == 0:
            elapsed = time.time() - step_start
            speed = batch["input_values"].shape[0] / max(elapsed, 0.001)
            epoch_elapsed = time.time() - epoch_start
            steps_done = step + 1
            steps_remaining_epoch = steps_per_epoch - steps_done
            time_per_step = epoch_elapsed / steps_done
            eta_epoch = steps_remaining_epoch * time_per_step
            epochs_remaining = total_epochs - epoch
            eta_global = eta_epoch + epochs_remaining * steps_per_epoch * time_per_step

            current_lr = optimizer.param_groups[-1]["lr"]

            reporter.send_step(
                epoch=epoch,
                step=step + 1,
                steps_per_epoch=steps_per_epoch,
                loss=loss_val,
                loss_ma5=sum(loss_window) / len(loss_window),
                speed_bps=speed,
                grad_norm=grad_norm,
                lr=current_lr,
                eta_epoch_s=eta_epoch,
                eta_global_s=eta_global,
                gpu=get_gpu_stats(),
            )

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion_tone, criterion_phoneme, phase, device, use_amp):
    model.eval()
    total_loss = 0.0
    tone_tracker = AccuracyTracker(5, TONE_NAMES)

    for batch in loader:
        input_values = batch["input_values"].to(device)
        lengths = batch["lengths"].to(device)
        tones = batch["tone"].to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(input_values, lengths=lengths, phase=phase)

            if phase == 1:
                loss = criterion_tone(outputs["tone_logits"], tones)
            elif phase == 2:
                loss_t = criterion_tone(outputs["tone_logits"], tones)
                phoneme_ids = batch["phoneme_id"].to(device)
                loss_p = criterion_phoneme(outputs["phoneme_logits"], phoneme_ids)
                has_tone = (tones >= 0).any()
                has_phoneme = (phoneme_ids >= 0).any()
                if has_tone and has_phoneme:
                    loss = 0.6 * loss_t + 0.4 * loss_p
                elif has_tone:
                    loss = loss_t
                elif has_phoneme:
                    loss = loss_p
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = torch.tensor(0.0, device=device)

        total_loss += loss.item()

        if phase in (1, 2) and "tone_logits" in outputs:
            tone_tracker.update(outputs["tone_logits"], tones)

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, tone_tracker.accuracy, tone_tracker.tone_accs_dict()


# ── Main ────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train Mandarin pronunciation model")
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-backbone", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--backbone", type=str, default="microsoft/wavlm-base")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--ws-url", type=str, default=None)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    phase = args.phase
    defaults = PHASE_DEFAULTS[phase]

    # Apply defaults
    epochs = args.epochs or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else defaults["warmup"]
    grad_clip = args.grad_clip if args.grad_clip is not None else defaults["grad_clip"]
    patience = args.patience if args.patience is not None else defaults["patience"]
    use_amp = not args.no_amp

    dataset_root = args.dataset_root or os.environ.get("DATASET_ROOT", "/app/dataset")
    ckpt_dir = args.checkpoint_dir or os.environ.get("CHECKPOINT_DIR", "/checkpoints")
    ws_url = args.ws_url or os.environ.get("WS_RELAY_URL")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase {phase} training ===")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, AMP: {use_amp}")

    # Build phoneme vocab from TTS manifests
    tts_train = get_manifests(dataset_root, 1, "train")
    phoneme_vocab = build_phoneme_vocab(tts_train)
    num_phonemes = max(len(phoneme_vocab), 1)
    print(f"  Phoneme vocab: {num_phonemes} classes")

    # Build datasets
    train_manifests = get_manifests(dataset_root, phase, "train")
    val_manifests = get_manifests(dataset_root, phase, "val")
    print(f"  Train manifests: {[Path(m).name for m in train_manifests]}")
    print(f"  Val manifests: {[Path(m).name for m in val_manifests]}")

    train_ds = MandarinDataset(
        train_manifests, dataset_root, phase, pinyin_to_phoneme_id=phoneme_vocab
    )
    val_ds = MandarinDataset(
        val_manifests, dataset_root, phase, pinyin_to_phoneme_id=phoneme_vocab
    )
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build model
    model = MandarinCoachModel(
        backbone_name=args.backbone,
        num_phonemes=num_phonemes,
    )

    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, model)
        start_epoch = ckpt.get("epoch", 0) + 1

    model = model.to(device)

    # Apply curriculum
    print(f"\n  Curriculum (Phase {phase}):")
    if phase == 1:
        freeze_backbone(model)
    elif phase == 2:
        unfreeze_last_n_layers(model, start=6)
    else:
        unfreeze_all(model)
    print_trainable_summary(model)

    # Optimizer
    if phase == 2:
        lr_bb = args.lr_backbone or defaults["lr_backbone"]
        lr_h = args.lr_head or defaults["lr_head"]
        param_groups = get_param_groups(model, lr_bb, lr_h)
        print(f"  LR backbone: {lr_bb}, LR head: {lr_h}")
    else:
        lr = args.lr or defaults["lr"]
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}
        ]
        print(f"  LR: {lr}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler, sched_mode = build_scheduler(optimizer, total_steps, warmup_steps, phase)

    # Loss functions
    criterion_tone = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_phoneme = nn.CrossEntropyLoss(ignore_index=-1)

    # AMP scaler
    scaler = GradScaler(enabled=use_amp)

    # WS reporter
    run_id = str(uuid.uuid4())[:8]
    reporter = WsReporter(url=ws_url, run_id=run_id)
    reporter.start()

    # Send config event
    config_data = {
        "phase": phase,
        "backbone": args.backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "total_samples": len(train_ds),
        "val_samples": len(val_ds),
        "datasets": [Path(m).stem for m in train_manifests],
        "num_phonemes": num_phonemes,
        "amp": use_amp,
    }
    reporter.send_config(config_data)

    # Training
    best_val_acc = 0.0
    no_improve = 0
    watcher = TrainingWatcher()

    print(f"\n  Starting training from epoch {start_epoch}...\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"--- Epoch {epoch}/{start_epoch + epochs - 1} ---")
        epoch_start = time.time()

        train_tone_tracker = AccuracyTracker(5, TONE_NAMES)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_mode=sched_mode,
            criterion_tone=criterion_tone,
            criterion_phoneme=criterion_phoneme,
            phase=phase,
            epoch=epoch,
            total_epochs=start_epoch + epochs - 1,
            device=device,
            use_amp=use_amp,
            grad_clip=grad_clip,
            reporter=reporter,
            report_every=args.report_every,
            tone_tracker=train_tone_tracker,
        )

        # Validation
        val_loss, val_acc, tone_accs = validate(
            model, val_loader, criterion_tone, criterion_phoneme, phase, device, use_amp
        )

        if sched_mode == "epoch":
            scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1

        train_acc = train_tone_tracker.accuracy
        print(
            f"  loss: {train_loss:.4f}/{val_loss:.4f}  "
            f"acc: {train_acc:.1f}/{val_acc:.1f}  "
            f"tones: {tone_accs}  "
            f"{'* BEST' if is_best else ''} ({elapsed:.0f}s)"
        )

        # Report epoch end
        reporter.send_epoch_end(
            epoch=epoch,
            loss_train=train_loss,
            loss_val=val_loss,
            acc_train=train_acc,
            acc_val=val_acc,
            tone_accs=tone_accs,
            is_best=is_best,
        )

        # Passive alerts
        watcher.check(
            epoch, phase,
            tone_accs=tone_accs,
        )

        # Save checkpoint
        ckpt_path = save_checkpoint(
            model, optimizer, epoch, phase, val_acc, val_loss, config_data, ckpt_dir
        )
        reporter.send_checkpoint(
            epoch=epoch, path=ckpt_path, val_acc=val_acc, is_best=is_best
        )

        # Early stopping
        if patience and no_improve >= patience:
            print(f"\n  Early stopping after {patience} epochs without improvement")
            break

    print(f"\n=== Training complete. Best val acc: {best_val_acc:.1f}% ===")
    reporter.stop()


if __name__ == "__main__":
    main()
