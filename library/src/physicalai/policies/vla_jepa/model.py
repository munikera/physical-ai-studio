# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA model: Qwen3-VL-2B + FlowmatchingActionHead.

Architecture (VLA-JEPA paper, ECCV 2026):
  1. <|embodied_action|> × 32 tokens are injected into the Qwen3-VL user prompt.
  2. Qwen3-VL processes [image tokens | task text | <|embodied_action|>×32].
  3. The 32 <|embodied_action|> hidden states become the VL conditioning (vl_embs, B×32×2048).
  4. FlowmatchingActionHead denoises the action chunk conditioned on vl_embs and proprioceptive state.

V-JEPA is used for video-prediction pretraining (the JEPA objective), not in the inference forward pass.
The pretrained checkpoint (VLA-JEPA-pretrain.pt) contains Qwen and action_model weights only.

XPU note: attn_implementation="sdpa" is required (no flash_attention_2 on Intel XPU).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .config import VlaJepaConfig

logger = logging.getLogger(__name__)


def _normalize(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (actions - mean) / (std + 1e-8)


def _denormalize(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return actions * (std + 1e-8) + mean


class VlaJepaModel(nn.Module):
    """Core nn.Module for VLA-JEPA.

    Wraps Qwen3-VL-2B and a FlowmatchingActionHead following the paper architecture.
    The 32 <|embodied_action|> tokens injected in the Qwen prompt serve as the
    conditioning signal for the action head (no separate V-JEPA encoder at inference).
    """

    def __init__(
        self,
        config: VlaJepaConfig,
        dataset_stats: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self.config = config

        self._build_action_stats(dataset_stats)
        self._build_models()

        if config.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self._set_trainable_params()

    def _build_action_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        action_stats = dataset_stats.get("action", {})
        state_stats = dataset_stats.get("observation.state", {})

        self.register_buffer(
            "action_mean",
            torch.tensor(action_stats.get("mean", [0.0] * self.config.action_dim), dtype=torch.float32),
        )
        self.register_buffer(
            "action_std",
            torch.tensor(action_stats.get("std", [1.0] * self.config.action_dim), dtype=torch.float32),
        )
        self.register_buffer(
            "state_mean",
            torch.tensor(state_stats.get("mean", [0.0] * self.config.state_dim), dtype=torch.float32),
        )
        self.register_buffer(
            "state_std",
            torch.tensor(state_stats.get("std", [1.0] * self.config.state_dim), dtype=torch.float32),
        )

    def _build_models(self) -> None:
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        cfg = self.config

        # --- Qwen3-VL-2B ---
        logger.info("Loading Qwen3-VL from %s (attn=sdpa)", cfg.qwen_model_name)
        self.qwen = AutoModel.from_pretrained(
            cfg.qwen_model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        self.qwen_processor = AutoProcessor.from_pretrained(cfg.qwen_model_name)

        # Add <|embodied_action|> special token
        self.qwen_processor.tokenizer.add_tokens(["<|embodied_action|>"], special_tokens=True)
        self.qwen.resize_token_embeddings(len(self.qwen_processor.tokenizer))

        # Cache the action token id — used to find extraction positions
        self.action_token_id: int = self.qwen_processor.tokenizer.convert_tokens_to_ids(
            "<|embodied_action|>"
        )

        qwen_hidden = self.qwen.config.text_config.hidden_size  # 2048 for Qwen3-VL-2B

        # --- FlowmatchingActionHead ---
        from .flow_matching_head import FlowmatchingActionHead  # noqa: PLC0415

        self.action_head = FlowmatchingActionHead(
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            action_horizon=cfg.chunk_size,
            hidden_size=1024,
            input_embedding_dim=768,
            num_attention_heads=12,
            attention_head_dim=64,
            num_layers=16,
            cross_attention_dim=qwen_hidden,
            num_target_vision_tokens=cfg.num_embodied_action_tokens,
            num_timestep_buckets=cfg.num_diffusion_steps_train,
            num_inference_timesteps=cfg.num_diffusion_steps_infer,
            noise_beta_alpha=cfg.noise_beta_alpha,
            noise_beta_beta=cfg.noise_beta_beta,
            noise_s=cfg.noise_s,
        )

    def _enable_gradient_checkpointing(self) -> None:
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            self.qwen.gradient_checkpointing_enable()

    def _set_trainable_params(self) -> None:
        cfg = self.config

        if cfg.freeze_qwen_vision and hasattr(self.qwen, "visual"):
            for p in self.qwen.visual.parameters():
                p.requires_grad_(False)

        # Qwen3-VL base model uses .language_model; Qwen2-VL uses .model
        llm_attr = "language_model" if hasattr(self.qwen, "language_model") else "model"
        if cfg.freeze_qwen_llm and hasattr(self.qwen, llm_attr):
            for p in getattr(self.qwen, llm_attr).parameters():
                p.requires_grad_(False)

    def set_dataset_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        self._build_action_stats(dataset_stats)

    def _encode_with_embodied_tokens(
        self,
        images: list[Any],
        task: list[str],
    ) -> torch.Tensor:
        """Encode images + task + 32 <|embodied_action|> tokens with Qwen3-VL.

        The 32 <|embodied_action|> tokens are appended at the end of the user
        turn, so Qwen can attend to image + text when computing their hidden states.

        Returns:
            embodied_hidden: (B, num_embodied_action_tokens, qwen_hidden)
        """
        n_tokens = self.config.num_embodied_action_tokens
        action_suffix = " " + "<|embodied_action|>" * n_tokens

        messages_batch = []
        for img, t in zip(images, task):
            content = []
            for img_item in (img if isinstance(img, list) else [img]):
                content.append({"type": "image", "image": img_item})
            content.append({"type": "text", "text": t + action_suffix})
            messages_batch.append([{"role": "user", "content": content}])

        texts = [
            self.qwen_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        device = next(self.qwen.parameters()).device
        inputs = self.qwen_processor(
            text=texts,
            images=[
                [item["image"] for item in m[0]["content"] if item["type"] == "image"]
                for m in messages_batch
            ],
            padding=True,
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        ).to(device)

        # Cast float inputs to model dtype so bf16-true works (no autocast in that mode).
        qwen_dtype = next(self.qwen.parameters()).dtype
        inputs = {
            k: v.to(qwen_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in inputs.items()
        }

        grad_enabled = not (self.config.freeze_qwen_vision and self.config.freeze_qwen_llm)
        with torch.set_grad_enabled(grad_enabled):
            outputs = self.qwen(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # (B, L, qwen_hidden)
        input_ids = inputs["input_ids"]     # (B, L)

        # Extract hidden states at <|embodied_action|> positions
        B = hidden.shape[0]
        action_token_id = self.action_token_id
        embodied_list = []
        for b in range(B):
            positions = (input_ids[b] == action_token_id).nonzero(as_tuple=True)[0]
            if len(positions) < n_tokens:
                # Padding fallback: repeat last valid position or use zeros
                logger.warning(
                    "batch[%d]: found %d embodied tokens, expected %d — using zeros for missing",
                    b, len(positions), n_tokens,
                )
                emb = torch.zeros(n_tokens, hidden.shape[-1], device=device, dtype=hidden.dtype)
                if len(positions) > 0:
                    emb[:len(positions)] = hidden[b, positions]
            else:
                emb = hidden[b, positions[:n_tokens]]
            embodied_list.append(emb)

        return torch.stack(embodied_list, dim=0)  # (B, n_tokens, qwen_hidden)

    def forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training forward pass — compute flow-matching loss.

        Args:
            batch: dict with keys:
                "images" : list[PIL.Image] — single frame per batch item
                "task"   : list[str] — task instruction strings
                "state"  : (B, state_dim)
                "action" : (B, chunk_size, action_dim)

        Returns:
            (loss, loss_dict)
        """
        actions_gt = batch["action"]  # (B, T_a, action_dim)
        state = batch["state"]        # (B, state_dim)

        actions_norm = _normalize(actions_gt, self.action_mean, self.action_std)
        state_norm = _normalize(state, self.state_mean, self.state_std)

        # Get embodied_action_tokens from Qwen (B, 32, qwen_hidden)
        vl_embs = self._encode_with_embodied_tokens(batch["images"], batch["task"])

        # Cast to model dtype so bf16-true and bf16-mixed both work.
        # With bf16-mixed Lightning wraps in autocast; with bf16-true it doesn't,
        # so dataset float32 tensors would hit bf16 weight matrices without this cast.
        mdtype = vl_embs.dtype
        actions_norm = actions_norm.to(mdtype)
        state_input = state_norm.unsqueeze(1).to(mdtype)

        # Flow-matching loss
        loss = self.action_head(vl_embs, actions_norm, state_input)

        return loss, {"loss": loss}

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Inference: Euler ODE integration → denormalized action chunk.

        Returns:
            (B, chunk_size, action_dim) denormalized actions.
        """
        state = batch["state"]
        state_norm = _normalize(state, self.state_mean, self.state_std)

        vl_embs = self._encode_with_embodied_tokens(batch["images"], batch["task"])

        mdtype = vl_embs.dtype
        state_input = state_norm.unsqueeze(1).to(mdtype)
        actions_norm = self.action_head.predict_action(vl_embs, state_input)

        actions = _denormalize(actions_norm, self.action_mean, self.action_std)

        if self.config.binarize_gripper_action:
            g = self.config.gripper_dim
            actions[..., g] = torch.where(
                actions[..., g] > self.config.gripper_threshold,
                torch.ones_like(actions[..., g]),
                torch.zeros_like(actions[..., g]),
            )

        return actions
