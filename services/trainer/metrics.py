"""
Training metrics: accuracy, per-tone accuracy.
"""

import torch
from collections import defaultdict

TONE_NAMES = ["T1", "T2", "T3", "T4", "T5"]


class AccuracyTracker:
    """Track overall and per-class accuracy across batches."""

    def __init__(self, num_classes: int = 5, class_names: list[str] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"C{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_correct = defaultdict(int)
        self.per_class_total = defaultdict(int)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: (B, C) raw logits
            targets: (B,) class indices. Entries with value -1 are ignored.
        """
        mask = targets >= 0
        if mask.sum() == 0:
            return

        logits = logits[mask]
        targets = targets[mask]

        preds = logits.argmax(dim=-1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.shape[0]

        for pred, target in zip(preds.cpu().tolist(), targets.cpu().tolist()):
            self.per_class_correct[target] += int(pred == target)
            self.per_class_total[target] += 1

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1) * 100

    def per_class_accuracy(self) -> dict[str, float]:
        result = {}
        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else f"C{i}"
            total = self.per_class_total[i]
            correct = self.per_class_correct[i]
            result[name] = (correct / max(total, 1)) * 100
        return result

    def tone_accs_dict(self) -> dict[str, float]:
        """Format for WS events: {T1: x, T2: y, T3: z, T4: w, T5: v}."""
        pca = self.per_class_accuracy()
        return {name: round(pca.get(name, 0), 1) for name in TONE_NAMES}
