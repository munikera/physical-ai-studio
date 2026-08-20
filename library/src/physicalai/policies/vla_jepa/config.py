# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for VLA-JEPA policy.

VLA-JEPA combines Qwen3-VL-2B + V-JEPA 2 ViT-L + DiT-B action head.
Paper: https://github.com/ginwind/VLA-JEPA (ECCV 2026)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physicalai.config import Config


@dataclass(frozen=True)
class VlaJepaConfig(Config):
    """Configuration for VLA-JEPA flow-matching VLA model.

    Attributes:
        chunk_size: Number of future actions to predict per forward pass.
        n_action_steps: Number of actions to execute from each predicted chunk.
        action_dim: Action space dimensionality (7 for SO-ARM 101: 6 joints + gripper).
        state_dim: Proprioceptive state dimensionality.

        qwen_model_name: HF repo or local path for Qwen3-VL-2B-Instruct.
        vjepa_encoder_name: HF repo or local path for V-JEPA 2 ViT-L encoder.
        num_frames: Number of video frames fed to V-JEPA encoder (model default is 64;
            8 fits safely in 32 GB XPU with other components loaded).
        video_resolution: Spatial resolution fed to V-JEPA encoder.

        freeze_vjepa_encoder: Freeze V-JEPA encoder weights during fine-tuning.
        freeze_qwen_vision: Freeze Qwen3-VL vision tower during fine-tuning.
        freeze_qwen_llm: Freeze Qwen3-VL language model backbone during fine-tuning.

        lr_backbone: Learning rate for VLM + V-JEPA components.
        lr_action_head: Learning rate for DiT action head and projectors.
        optimizer_betas: Adam betas.
        optimizer_eps: Adam epsilon.
        optimizer_weight_decay: Adam weight decay.
        optimizer_grad_clip_norm: Gradient clipping norm.
        scheduler_warmup_steps: LR warmup steps.
        scheduler_decay_steps: Steps to decay LR to scheduler_decay_lr.
        scheduler_decay_lr: Final LR after cosine decay.

        num_diffusion_steps_train: Diffusion timestep buckets during training.
        num_diffusion_steps_infer: Number of denoising steps at inference.

        image_resize: (H, W) to resize images before Qwen3-VL processor.
        tokenizer_max_length: Max token length for Qwen3-VL tokenizer.
        enable_gradient_checkpointing: Reduce activation memory at cost of recompute.
    """

    chunk_size: int = 50
    n_action_steps: int = 50

    action_dim: int = 6   # SO-ARM 101: 6 DOF (includes gripper as 6th joint)
    state_dim: int = 6

    # Gripper binarization (inference only)
    gripper_dim: int = 5          # 0-indexed; last joint for SO-ARM 101 (action_dim - 1)
    gripper_threshold: float = 0.5
    binarize_gripper_action: bool = False

    # Model checkpoints (set to local paths after downloading)
    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    vjepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"
    num_frames: int = 8
    video_resolution: int = 256

    # Fine-tuning freezing
    freeze_vjepa_encoder: bool = True
    freeze_qwen_vision: bool = False
    freeze_qwen_llm: bool = True

    # Optimizer — differential LRs
    lr_backbone: float = 1e-5
    lr_action_head: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0

    # LR schedule
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 10_000
    scheduler_decay_lr: float = 1e-6

    # DiT action head
    num_diffusion_steps_train: int = 1000
    num_diffusion_steps_infer: int = 4

    # Beta noise schedule (paper: α=1.5, β=1.0, s=0.999)
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999

    # Preprocessing
    image_resize: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 512
    enable_gradient_checkpointing: bool = True

    # Number of <|embodied_action|> tokens injected into Qwen prompt
    num_embodied_action_tokens: int = 32

    # Path to pretrained VLA-JEPA checkpoint (leave empty for random init)
    pretrained_checkpoint: str = "/checkpoints/vla-jepa-pretrain/Pretrain/checkpoints/VLA-JEPA-pretrain.pt"

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            msg = (
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
