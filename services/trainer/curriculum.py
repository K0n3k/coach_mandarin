"""
Curriculum learning: freeze/unfreeze strategies for 3-phase training.

Phase 1: Backbone frozen, only layer_weights + tone head trainable.
Phase 2: Layers 6-11 unfrozen + layer_weights + all heads.
Phase 3: Everything unfrozen.
"""

import torch.nn as nn


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: freeze entire WavLM backbone. layer_weights stays trainable."""
    for param in model.encoder.backbone.parameters():
        param.requires_grad = False
    model.encoder.layer_weights.requires_grad = True


def unfreeze_last_n_layers(model: nn.Module, start: int = 6) -> None:
    """Phase 2: unfreeze transformer layers [start:] (6-11). layer_weights always trainable."""
    # Keep CNN feature extractor frozen
    for param in model.encoder.backbone.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.encoder.backbone.feature_projection.parameters():
        param.requires_grad = False

    # Freeze layers 0..start-1, unfreeze layers start..11
    for i, layer in enumerate(model.encoder.backbone.encoder.layers):
        for param in layer.parameters():
            param.requires_grad = (i >= start)

    # Unfreeze encoder layer norm and pos conv if present
    if hasattr(model.encoder.backbone.encoder, "layer_norm"):
        for param in model.encoder.backbone.encoder.layer_norm.parameters():
            param.requires_grad = True
    if hasattr(model.encoder.backbone.encoder, "pos_conv_embed"):
        for param in model.encoder.backbone.encoder.pos_conv_embed.parameters():
            param.requires_grad = False  # keep frozen

    model.encoder.layer_weights.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Phase 3: unfreeze everything."""
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
) -> list[dict]:
    """
    Create optimizer parameter groups with separate LRs.

    Backbone group: WavLM encoder parameters that require grad.
    Head group: everything else (layer_weights, heads).
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    groups.append({"params": head_params, "lr": lr_head})
    return groups


def print_trainable_summary(model: nn.Module) -> dict:
    """Print and return summary of trainable vs frozen parameters."""
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if param.requires_grad:
            trainable += n
        else:
            frozen += n

    total = trainable + frozen
    print(f"  Trainable: {trainable:,} ({trainable / total * 100:.1f}%)")
    print(f"  Frozen:    {frozen:,} ({frozen / total * 100:.1f}%)")
    return {"trainable": trainable, "frozen": frozen}
