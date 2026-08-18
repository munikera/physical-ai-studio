# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for VLA-JEPA weight loading and dataset statistics."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_dataset_stats(
    action_mean: list[float],
    action_std: list[float],
    state_mean: list[float],
    state_std: list[float],
) -> dict[str, dict[str, Any]]:
    """Build a minimal dataset_stats dict from pre-computed statistics."""
    action_dim = len(action_mean)
    state_dim = len(state_mean)
    return {
        "action": {
            "mean": action_mean,
            "std": action_std,
            "shape": (action_dim,),
            "type": "continuous",
        },
        "observation.state": {
            "mean": state_mean,
            "std": state_std,
            "shape": (state_dim,),
            "type": "continuous",
        },
    }


# ---------------------------------------------------------------------------
# Key remapping: paper checkpoint → our module names
# ---------------------------------------------------------------------------

_KEY_REMAP = [
    # Paper wraps Qwen as qwen_vl_interface.model (ForConditionalGeneration),
    # whose sub-keys start with model.visual.* / model.language_model.* / lm_head.*
    # Our AutoModel.from_pretrained returns Qwen3VLModel (base model):
    #   self.qwen.visual.* / self.qwen.language_model.*
    # So strip two levels: qwen_vl_interface.model.model.* → qwen.*
    ("qwen_vl_interface.model.model.", "qwen."),
    # lm_head lives on ForCG but not on base model; rename to _unused so it's skipped
    ("qwen_vl_interface.model.lm_head.", "_qwen_lm_head_unused."),
    # Paper's action model → our self.action_head
    ("action_model.", "action_head."),
    # V-JEPA encoder/predictor — not part of inference model
    ("vj_encoder.", "_vj_encoder_unused."),
    ("vj_predictor.", "_vj_predictor_unused."),
]


def _remap_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename checkpoint keys to match our VlaJepaModel naming."""
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in _KEY_REMAP:
            if new_k.startswith(old_prefix):
                new_k = new_prefix + new_k[len(old_prefix):]
                break
        remapped[new_k] = v
    return remapped


def load_vla_jepa_pretrain(
    model: nn.Module,
    checkpoint_path: str,
) -> tuple[int, int]:
    """Load the paper's VLA-JEPA-pretrain.pt into our VlaJepaModel.

    Key mapping:
      qwen_vl_interface.model.* → qwen.*
      action_model.*            → action_head.*

    Shape mismatches are skipped so action_dim/state_dim differences
    (paper: action_dim=7, state_dim=8 vs ours: 6/6) don't block loading.

    Returns:
        (loaded_count, skipped_count)
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap common wrappers
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw:
                raw = raw[key]
                break

    # Strip DDP / framework prefixes
    stripped: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        k = k.removeprefix("module.")
        k = k.removeprefix("framework.")
        stripped[k] = v

    ckpt = _remap_keys(stripped)

    model_sd = model.state_dict()
    loaded = 0
    skipped = 0

    for k, v in ckpt.items():
        if k not in model_sd:
            logger.debug("Skipping unknown key: %s", k)
            skipped += 1
            continue

        if model_sd[k].shape != v.shape:
            logger.info(
                "Skipping shape mismatch: %s  ckpt=%s  model=%s",
                k, tuple(v.shape), tuple(model_sd[k].shape),
            )
            skipped += 1
            continue

        model_sd[k] = v.to(dtype=model_sd[k].dtype)
        loaded += 1

    model.load_state_dict(model_sd, strict=False)
    return loaded, skipped


def load_vla_jepa_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> None:
    """Load any VLA-JEPA checkpoint into `model` (legacy helper, non-strict by default)."""
    loaded, skipped = load_vla_jepa_pretrain(model, checkpoint_path)
    logger.info("Checkpoint load: %d loaded, %d skipped", loaded, skipped)
