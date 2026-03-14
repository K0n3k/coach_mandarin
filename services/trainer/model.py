"""
WavLM-based Mandarin pronunciation model.

Architecture:
  WavLMWithWeightedSum (backbone)
    -> ToneHead         (Phase 1+2: 5-class classification)
    -> PhonemeHead      (Phase 2:   N-class classification)
    -> ScoringHead      (Phase 3:   GOPT-inspired regression)
    -> FreeScorer       (Phase 3:   robust free-form regression)

MandarinCoachModel wraps everything and selects active heads per phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class WavLMWithWeightedSum(nn.Module):
    """
    WavLM backbone with learnable weighted sum of all hidden states.

    WavLM-base: 12 transformer layers + 1 CNN feature extractor = 13 hidden states.
    layer_weights are ALWAYS trainable (never frozen, even in Phase 1).
    """

    def __init__(self, model_name: str = "microsoft/wavlm-base"):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name)
        num_layers = self.backbone.config.num_hidden_layers + 1  # 12 + 1 = 13
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: (B, T_samples) raw waveform at 16kHz
        Returns:
            weighted: (B, T_frames, 768) weighted sum of hidden states
        """
        outputs = self.backbone(input_values, output_hidden_states=True)
        weights = torch.softmax(self.layer_weights, dim=0)  # (13,)
        # Accumulate weighted sum without materializing full (B, 13, T, 768) tensor
        weighted = torch.zeros_like(outputs.hidden_states[0])
        for i, hs in enumerate(outputs.hidden_states):
            weighted = weighted + weights[i] * hs
        return weighted


class ToneHead(nn.Module):
    """5-class tone classifier: T1, T2, T3, T4, neutral."""

    def __init__(self, hidden_size: int = 768, num_tones: int = 5):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_tones)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class PhonemeHead(nn.Module):
    """Phoneme classifier for Mandarin syllables."""

    def __init__(self, hidden_size: int = 768, num_phonemes: int = 400):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_phonemes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class ScoringHead(nn.Module):
    """
    GOPT-inspired scoring head for guided mode.

    Takes softmax tone probabilities + pooled features as input.
    Outputs 4 regression scores (0-100) + 1 classification (confusion type).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_tones: int = 5,
        proj_size: int = 256,
        num_confusion_types: int = 12,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_size + num_tones, proj_size)
        self.heads = nn.ModuleDict({
            "score_ton": nn.Linear(proj_size, 1),
            "score_initiale": nn.Linear(proj_size, 1),
            "score_finale": nn.Linear(proj_size, 1),
            "score_global": nn.Linear(proj_size, 1),
            "type_confusion": nn.Linear(proj_size, num_confusion_types),
        })

    def forward(
        self, tone_probs: torch.Tensor, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([tone_probs, features], dim=-1)  # (B, 773)
        x = F.relu(self.proj(x))  # (B, 256)
        out = {}
        for name, head in self.heads.items():
            val = head(x)
            if name != "type_confusion":
                out[name] = torch.sigmoid(val.squeeze(-1)) * 100
            else:
                out[name] = val  # raw logits for CE loss
        return out


class FreeScorer(nn.Module):
    """Free-form scorer for libre mode (no reference text)."""

    def __init__(self, hidden_size: int = 768, proj_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features).squeeze(-1)) * 100


class MandarinCoachModel(nn.Module):
    """
    Full model wrapping backbone + all heads.
    Active heads depend on the training phase.
    """

    def __init__(
        self,
        backbone_name: str = "microsoft/wavlm-base",
        num_tones: int = 5,
        num_phonemes: int = 400,
        num_confusion_types: int = 12,
    ):
        super().__init__()
        self.encoder = WavLMWithWeightedSum(backbone_name)
        self.tone_head = ToneHead(768, num_tones)
        self.phoneme_head = PhonemeHead(768, num_phonemes)
        self.scoring_head = ScoringHead(768, num_tones, 256, num_confusion_types)
        self.free_scorer = FreeScorer(768, 256)

    def _masked_mean_pool(
        self, frame_features: torch.Tensor, lengths: torch.Tensor | None
    ) -> torch.Tensor:
        """Mean pool over time, masking out padding frames."""
        if lengths is None:
            return frame_features.mean(dim=1)

        # WavLM stride = 320 samples per frame
        frame_lengths = (lengths / 320).long().clamp(min=1)
        max_frames = frame_features.shape[1]
        # (1, T_max) < (B, 1) -> (B, T_max) boolean mask
        mask = (
            torch.arange(max_frames, device=frame_features.device)[None, :]
            < frame_lengths[:, None]
        )
        # Masked sum / count
        masked = frame_features * mask.unsqueeze(-1).float()
        pooled = masked.sum(dim=1) / frame_lengths.unsqueeze(-1).float()
        return pooled

    def forward(
        self,
        input_values: torch.Tensor,
        lengths: torch.Tensor | None = None,
        phase: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_values: (B, T_samples) raw waveform at 16kHz
            lengths: (B,) original waveform lengths before padding
            phase: 1, 2, or 3
        Returns:
            dict of outputs depending on phase
        """
        frame_features = self.encoder(input_values)  # (B, T_frames, 768)
        pooled = self._masked_mean_pool(frame_features, lengths)  # (B, 768)

        out = {"pooled": pooled}

        if phase in (1, 2):
            out["tone_logits"] = self.tone_head(pooled)

        if phase == 2:
            out["phoneme_logits"] = self.phoneme_head(pooled)

        if phase == 3:
            with torch.no_grad():
                tone_probs = F.softmax(self.tone_head(pooled), dim=-1)
            out["scoring"] = self.scoring_head(tone_probs, pooled)
            out["free_score"] = self.free_scorer(pooled)

        return out
