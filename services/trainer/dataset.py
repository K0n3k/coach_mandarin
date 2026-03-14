"""
PyTorch Dataset for Mandarin pronunciation training.

Loads audio from manifests, returns raw waveforms + labels.
Handles padding/collation for batching variable-length audio.
"""

import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000
MAX_DURATION_S = 10.0
MAX_SAMPLES = int(MAX_DURATION_S * SAMPLE_RATE)


class MandarinDataset(Dataset):
    """
    Unified dataset for all phases.

    Phase 1: TTS phase 1 only. Labels: tone (0-4).
    Phase 2: TTS + AISHELL + THCHS + CV. Labels: tone (TTS only), phoneme_id (TTS only).
    Phase 3: LATIC + anchors. Labels: scores (stub).
    """

    def __init__(
        self,
        manifest_paths: list[str],
        dataset_root: str = "/app/dataset",
        phase: int = 1,
        max_samples: int = MAX_SAMPLES,
        pinyin_to_phoneme_id: dict[str, int] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.phase = phase
        self.max_samples = max_samples
        self.pinyin_to_phoneme_id = pinyin_to_phoneme_id or {}

        self.entries = []
        for mp in manifest_paths:
            with open(mp, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries.extend(data)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        audio_path = self.dataset_root / entry["path"]
        waveform, sr = torchaudio.load(str(audio_path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Clip to max duration
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[: self.max_samples]

        result = {
            "waveform": waveform,
            "length": waveform.shape[0],
        }

        # Tone label: available in TTS phase 1 manifests, -1 otherwise
        if "tone" in entry:
            # Manifest tone is 1-5, map to 0-4
            t = entry["tone"]
            result["tone"] = t - 1 if 1 <= t <= 5 else -1
        else:
            result["tone"] = -1

        # Phoneme ID: from pinyin field in TTS manifests
        pinyin = entry.get("pinyin", "")
        if pinyin and pinyin in self.pinyin_to_phoneme_id:
            result["phoneme_id"] = self.pinyin_to_phoneme_id[pinyin]
        else:
            result["phoneme_id"] = -1

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length waveforms into a padded batch."""
    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    tones = torch.tensor([item["tone"] for item in batch], dtype=torch.long)
    phoneme_ids = torch.tensor(
        [item.get("phoneme_id", -1) for item in batch], dtype=torch.long
    )

    padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)

    return {
        "input_values": padded,
        "lengths": lengths,
        "tone": tones,
        "phoneme_id": phoneme_ids,
    }


def build_phoneme_vocab(manifest_paths: list[str]) -> dict[str, int]:
    """Build phoneme vocabulary from manifests that have a 'pinyin' field."""
    vocab = set()
    for mp in manifest_paths:
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if "pinyin" in entry:
                    vocab.add(entry["pinyin"])
    return {p: i for i, p in enumerate(sorted(vocab))}
