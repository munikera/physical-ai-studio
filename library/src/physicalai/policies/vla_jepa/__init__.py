# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA Policy — Qwen3-VL-2B + V-JEPA 2 ViT-L + DiT-B action head."""

from .config import VlaJepaConfig
from .model import VlaJepaModel
from .policy import VlaJepa

__all__ = ["VlaJepa", "VlaJepaConfig", "VlaJepaModel"]
