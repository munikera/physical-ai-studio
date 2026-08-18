# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Architecture ported from ginwind/VLA-JEPA (ECCV 2026).
# Original flow matching head © 2025 NVIDIA Corp. (GR00T N1.5), Apache-2.0.
# Adapted for Intel XPU: pure PyTorch (no diffusers dependency), action_dim=6.
#
# Module naming mirrors VLA-JEPA-pretrain.pt checkpoint exactly so weights load
# without key remapping (except 4 shape-mismatched layers for action/state dim).

"""Flow-matching action head for VLA-JEPA.

Verified key overlap with VLA-JEPA-pretrain.pt: 248/248 keys matched,
4 shape mismatches (action_dim 7→6 and state_dim 8→6) skipped by pretrained_utils.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Timestep encoder — naming: timestep_encoder.timestep_embedder.linear_1/2
# ---------------------------------------------------------------------------

class _TimestepMLP(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = _TimestepMLP(embedding_dim)

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        return self.timestep_embedder(self._sinusoidal(t).to(dtype))


# ---------------------------------------------------------------------------
# AdaLayerNorm — naming: norm1.linear (matches checkpoint norm1.linear.weight)
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        out = self.linear(F.silu(temb))
        scale, shift = out.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


# ---------------------------------------------------------------------------
# Attention — naming: attn1.to_q/to_k/to_v/to_out.0 (matches checkpoint)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        inner_dim: int,
        num_heads: int,
        cross_attention_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert inner_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads

        kv_dim = cross_attention_dim if cross_attention_dim is not None else inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, inner_dim, bias=bias)])

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        ctx = context if context is not None else x
        M = ctx.shape[1]

        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(ctx).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(ctx).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v)  # sdpa for XPU compatibility
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out[0](out)


# ---------------------------------------------------------------------------
# FeedForward — naming: ff.net.0.proj + ff.net.2 (matches checkpoint)
# ---------------------------------------------------------------------------

class _GELUProj(nn.Module):
    """GELU activation wrapper with .proj attr to match ff.net.0.proj naming."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList([
            _GELUProj(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[1](self.net[0](x)))


# ---------------------------------------------------------------------------
# TransformerBlock — naming matches checkpoint block structure
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        inner_dim: int,
        num_heads: int,
        cross_attention_dim: Optional[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(inner_dim)
        self.attn1 = Attention(
            inner_dim=inner_dim,
            num_heads=num_heads,
            cross_attention_dim=cross_attention_dim,
            dropout=dropout,
        )
        self.ff = FeedForward(dim=inner_dim, dropout=dropout)
        self.cross_attention_dim = cross_attention_dim

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(x, temb)
        if self.cross_attention_dim is not None and encoder_hidden_states is not None:
            x = x + self.attn1(normed, context=encoder_hidden_states)
        else:
            x = x + self.attn1(normed)
        x = x + self.ff(x)
        return x


# ---------------------------------------------------------------------------
# FlowMatchingDiT — naming: transformer_blocks, timestep_encoder, proj_out_1/2
# ---------------------------------------------------------------------------

class FlowMatchingDiT(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        output_dim: int = 1024,
        num_layers: int = 16,
        dropout: float = 0.0,
        cross_attention_dim: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim

        self.timestep_encoder = TimestepEncoder(self.inner_dim)

        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            is_self_attn = (i % 2 == 1)
            self.transformer_blocks.append(
                TransformerBlock(
                    inner_dim=self.inner_dim,
                    num_heads=num_attention_heads,
                    cross_attention_dim=None if is_self_attn else cross_attention_dim,
                    dropout=dropout,
                )
            )

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(self.inner_dim, self.inner_dim * 2)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


# ---------------------------------------------------------------------------
# Action encoder — naming: layer1/2/3 (matches checkpoint)
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(a_emb.dtype)
        x = swish(self.layer2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.layer3(x)


# ---------------------------------------------------------------------------
# State encoder / action decoder — naming: layer1/2 (matches checkpoint)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.silu(self.layer1(x)))


class ActionDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.silu(self.layer1(x)))


# ---------------------------------------------------------------------------
# FlowmatchingActionHead — public class used by VlaJepaModel
# ---------------------------------------------------------------------------

class FlowmatchingActionHead(nn.Module):
    """VLA-JEPA flow-matching action head.

    Module naming matches VLA-JEPA-pretrain.pt (ginwind/VLA-JEPA) exactly.
    After key remapping (action_model.→action_head.), 244 of 248 keys load
    from the pretrain checkpoint; 4 are re-initialized (action_dim 7→6,
    state_dim 8→6).

    Interleaved 16-layer DiT: even layers cross-attend to Qwen embodied tokens,
    odd layers are self-attention. AdaLayerNorm, Beta noise schedule.
    """

    def __init__(
        self,
        action_dim: int = 6,
        state_dim: int = 6,
        action_horizon: int = 7,
        hidden_size: int = 1024,
        input_embedding_dim: int = 768,
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        num_layers: int = 16,
        cross_attention_dim: int = 2048,
        num_target_vision_tokens: int = 32,
        num_timestep_buckets: int = 1000,
        num_inference_timesteps: int = 4,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_s = noise_s

        self.model = FlowMatchingDiT(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            output_dim=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
        )

        self.action_encoder = ActionEncoder(action_dim=action_dim, hidden_size=input_embedding_dim)
        self.state_encoder = StateEncoder(state_dim=state_dim, hidden_dim=hidden_size, output_dim=input_embedding_dim)
        self.action_decoder = ActionDecoder(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=action_dim)

        self.future_tokens = nn.Embedding(num_target_vision_tokens, input_embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, input_embedding_dim)

        nn.init.normal_(self.future_tokens.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - t) / self.noise_s

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute flow-matching MSE loss."""
        device = vl_embs.device

        noise = torch.randn_like(actions)
        t = self._sample_time(actions.shape[0], device, actions.dtype)
        t_bcast = t[:, None, None]

        noisy = (1 - t_bcast) * noise + t_bcast * actions
        velocity = actions - noise

        t_disc = (t * self.num_timestep_buckets).long()

        dtype = next(self.model.parameters()).dtype
        action_features = self.action_encoder(noisy.to(dtype), t_disc)

        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + self.position_embedding(pos_ids).to(dtype)

        future = self.future_tokens.weight.to(dtype).unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

        if state is not None:
            state_feat = self.state_encoder(state.to(dtype))
            sa_embs = torch.cat([state_feat, future, action_features], dim=1)
        else:
            sa_embs = torch.cat([future, action_features], dim=1)

        out = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs.to(dtype),
            timestep=t_disc,
        )
        pred = self.action_decoder(out)
        pred_actions = pred[:, -self.action_horizon:]

        return F.mse_loss(pred_actions.float(), velocity.float())

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler ODE integration → (B, action_horizon, action_dim)."""
        B = vl_embs.shape[0]
        device = vl_embs.device
        dtype = next(self.model.parameters()).dtype

        actions = torch.randn(B, self.action_horizon, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / self.num_inference_timesteps
        future = self.future_tokens.weight.to(dtype).unsqueeze(0).expand(B, -1, -1)
        state_feat = self.state_encoder(state.to(dtype)) if state is not None else None

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((B,), t_disc, device=device, dtype=torch.long)

            action_features = self.action_encoder(actions, t_tensor)
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).to(dtype)

            if state_feat is not None:
                sa_embs = torch.cat([state_feat, future, action_features], dim=1)
            else:
                sa_embs = torch.cat([future, action_features], dim=1)

            out = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs.to(dtype),
                timestep=t_tensor,
            )
            pred = self.action_decoder(out)
            velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * velocity.to(dtype)

        return actions.float()
