"""
WebSocket reporter for training events.

Connects to Go API relay at WS_RELAY_URL with role=trainer.
Sends JSON events matching the frontend TypeScript interfaces.
Thread-safe: called from training loop without blocking.
"""

import json
import os
import queue
import threading
from typing import Any

import websockets.sync.client as ws_client


class WsReporter:
    """
    Non-blocking WS reporter. Sends events in a background thread.
    If connection fails, events are silently dropped (training continues).
    """

    def __init__(self, url: str | None = None, run_id: str = ""):
        self.url = url or os.environ.get("WS_RELAY_URL")
        self.run_id = run_id
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        if not self.url:
            print("[ws_reporter] No WS_RELAY_URL, reporting disabled")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def send(self, event: dict[str, Any]):
        """Queue an event for sending. Non-blocking, drops if queue full."""
        event.setdefault("run_id", self.run_id)
        try:
            self._queue.put_nowait(json.dumps(event))
        except queue.Full:
            pass

    def send_config(self, config: dict):
        self.send({"type": "config", "config": config})

    def send_step(
        self,
        epoch: int,
        step: int,
        steps_per_epoch: int,
        loss: float,
        loss_ma5: float,
        speed_bps: float,
        grad_norm: float,
        lr: float,
        eta_epoch_s: float,
        eta_global_s: float,
        gpu: dict,
    ):
        self.send({
            "type": "step",
            "epoch": epoch,
            "step": step,
            "steps_per_epoch": steps_per_epoch,
            "loss": round(loss, 4),
            "loss_ma5": round(loss_ma5, 4),
            "speed_bps": round(speed_bps, 1),
            "grad_norm": round(grad_norm, 3),
            "lr": lr,
            "eta_epoch_s": round(eta_epoch_s),
            "eta_global_s": round(eta_global_s),
            "gpu": gpu,
        })

    def send_epoch_end(
        self,
        epoch: int,
        loss_train: float,
        loss_val: float,
        acc_train: float,
        acc_val: float,
        tone_accs: dict,
        is_best: bool,
        *,
        pcc_val: float | None = None,
        pcc_train: float | None = None,
        mse_per_head: dict[str, float] | None = None,
        score_distribution: dict[str, float] | None = None,
    ):
        payload: dict[str, Any] = {
            "type": "epoch_end",
            "epoch": epoch,
            "loss_train": round(loss_train, 3),
            "loss_val": round(loss_val, 3),
            "acc_train": round(acc_train, 1),
            "acc_val": round(acc_val, 1),
            "tone_accs": tone_accs,
            "is_best": is_best,
        }
        # Phase 3 regression fields (only sent when present)
        if pcc_val is not None:
            payload["pcc_val"] = round(pcc_val, 4)
        if pcc_train is not None:
            payload["pcc_train"] = round(pcc_train, 4)
        if mse_per_head is not None:
            payload["mse_per_head"] = {k: round(v, 3) for k, v in mse_per_head.items()}
        if score_distribution is not None:
            payload["score_distribution"] = {
                k: round(v, 2) for k, v in score_distribution.items()
            }
        self.send(payload)

    def send_checkpoint(
        self, epoch: int, path: str, val_acc: float, is_best: bool
    ):
        self.send({
            "type": "checkpoint",
            "epoch": epoch,
            "path": path,
            "val_acc": round(val_acc, 1),
            "is_best": is_best,
        })

    def _run(self):
        """Background thread: connect and send queued messages."""
        url = f"{self.url}?role=trainer"
        while not self._stop.is_set():
            try:
                with ws_client.connect(url) as ws:
                    print(f"[ws_reporter] Connected to {url}")
                    while not self._stop.is_set():
                        try:
                            msg = self._queue.get(timeout=1.0)
                            ws.send(msg)
                        except queue.Empty:
                            continue
                        except Exception:
                            break
            except Exception as e:
                print(f"[ws_reporter] Connection failed: {e}, retrying in 5s...")
                self._stop.wait(5.0)
