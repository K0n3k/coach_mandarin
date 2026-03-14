"""
Passive training alerts — watches epoch_end events and prints warnings.

Integrated into the training loop (not a standalone process).
Alerts are printed to stdout; they never stop training.

Phase 1-2 alerts:
  - T3 accuracy stuck below 40% after epoch 5

Phase 3 alerts:
  - pcc_val < 0.30 after epoch 10
  - score_distribution.std < 5.0  (model averaging everything)
  - pcc_val drop > 0.05 between consecutive epochs
"""


class TrainingWatcher:
    """Accumulates epoch_end data and emits passive alerts."""

    def __init__(self):
        self._prev_pcc_val: float | None = None

    def check(
        self,
        epoch: int,
        phase: int,
        *,
        tone_accs: dict[str, float] | None = None,
        pcc_val: float | None = None,
        score_distribution: dict[str, float] | None = None,
    ) -> list[str]:
        """Return a list of alert strings (empty = all good)."""
        alerts: list[str] = []

        # ── Phase 1-2: T3 stuck ────────────────────────────
        if phase in (1, 2) and tone_accs and epoch >= 5:
            t3 = tone_accs.get("T3", 100.0)
            if t3 < 40.0:
                alerts.append(
                    f"[alert] T3 accuracy stuck at {t3:.1f}% (< 40%) after epoch {epoch}"
                )

        # ── Phase 3: PCC too low ───────────────────────────
        if phase == 3 and pcc_val is not None and epoch >= 10:
            if pcc_val < 0.30:
                alerts.append(
                    f"[alert] pcc_val = {pcc_val:.4f} (< 0.30) after epoch {epoch}"
                )

        # ── Phase 3: score std collapse ────────────────────
        if phase == 3 and score_distribution is not None:
            std = score_distribution.get("std", 999.0)
            if std < 5.0:
                alerts.append(
                    f"[alert] score_distribution.std = {std:.2f} (< 5.0) — "
                    f"model may be averaging all predictions"
                )

        # ── Phase 3: PCC drop ──────────────────────────────
        if phase == 3 and pcc_val is not None and self._prev_pcc_val is not None:
            drop = self._prev_pcc_val - pcc_val
            if drop > 0.05:
                alerts.append(
                    f"[alert] pcc_val dropped by {drop:.4f} "
                    f"({self._prev_pcc_val:.4f} → {pcc_val:.4f})"
                )

        # Update state for next call
        if pcc_val is not None:
            self._prev_pcc_val = pcc_val

        # Print alerts
        for a in alerts:
            print(a)

        return alerts
