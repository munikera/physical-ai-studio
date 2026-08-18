# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DiT-B action head matching VLA-JEPA paper checkpoint structure.

Module naming mirrors checkpoint keys exactly for direct weight loading:
  action_model.model.*              → self.model.*
  action_model.action_encoder.*     → self.action_encoder.*
  action_model.state_encoder.*      → self.state_encoder.*
  action_model.action_decoder.*     → self.action_decoder.*
  action_model.future_tokens        → self.future_tokens
  action_model.position_embedding   → self.position_embedding

Architecture (from VLA-JEPA-pretrain.pt inspection):
  - hidden_size = 768 (internal transformer dim)
  - 16 blocks alternating: cross-attn (even, context_dim=2048) / self-attn (odd)
  - AdaLayerNorm: Linear(768, 1536) → scale(768) + shift(768)
  - FeedForward: Linear(768, 3072) + GELU + Linear(3072, 768)
  - Timestep: sinusoidal(256) → Linear(256, 768) → SiLU → Linear(768, 768)
  - Action encoder: 3-layer MLP with timestep concatenation
  - State encoder: 2-layer MLP (state_dim → 1024 → 768)
  - Action decoder: 2-layer MLP (1024 → 1024 → action_dim)
  - Output: AdaLayerNorm + proj_out_2(768→1024) → decoder
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules matching checkpoint key structure
# ---------------------------------------------------------------------------

class _AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: conditions normalization on timestep embedding.

    Matches key: transformer_blocks.*.norm1 (weight [1536, 768])
    """

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(t_emb).chunk(2, dim=-1)  # each (B, dim)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _GELUProj(nn.Module):
    """GELU activation with linear projection.

    Matches key: ff.net.0 (weight [3072, 768])
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x))


class _FeedForward(nn.Module):
    """2-layer FFN: Linear(768→3072) + GELU + Linear(3072→768).

    Matches keys: ff.net.0.proj [3072, 768], ff.net.2 [768, 3072]
    """

    def __init__(self, dim: int = 768, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.ModuleList([
            _GELUProj(dim, inner),     # net.0.proj = Linear(dim, inner)
            nn.Dropout(dropout),        # net.1
            nn.Linear(inner, dim),      # net.2
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[1](self.net[0](x)))


class _Attention(nn.Module):
    """Multi-head attention (cross or self).

    Matches keys: attn1.to_q/k/v/to_out.0
    Cross blocks:  to_k/v weight [768, 2048] (context_dim=2048)
    Self blocks:   to_k/v weight [768, 768]  (context_dim=768)
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim), nn.Dropout(0.0)])

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Tq = query.shape[:2]
        Tc = context.shape[1]

        q = self.to_q(query).view(B, Tq, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, Tc, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, Tc, self.heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)  # sdpa for XPU
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.to_out[0](out)


class _DiTBlock(nn.Module):
    """Single DiT block: AdaLayerNorm → Attention → FFN.

    Even-indexed blocks: cross-attention to VLM context (2048-dim)
    Odd-indexed blocks:  self-attention
    """

    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = _AdaLayerNorm(hidden_size)
        self.attn1 = _Attention(hidden_size, context_dim, heads)
        self.ff = _FeedForward(hidden_size, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm1(x, t_emb)
        x = x + self.attn1(normed, ctx)
        x = x + self.ff(x)
        return x


class _TimestepEmbedder(nn.Module):
    """Sinusoidal + 2-layer MLP timestep embedding.

    Matches keys: timestep_encoder.timestep_embedder.linear_1/2
    """

    def __init__(self, freq_dim: int = 256, hidden_size: int = 768) -> None:
        super().__init__()
        self._freq_dim = freq_dim
        self.linear_1 = nn.Linear(freq_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self._freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = F.silu(self.linear_1(emb))
        return self.linear_2(emb)


class _TimestepEncoder(nn.Module):
    """Wraps _TimestepEmbedder.

    Matches key: model.timestep_encoder.timestep_embedder.*
    """

    def __init__(self, freq_dim: int = 256, hidden_size: int = 768) -> None:
        super().__init__()
        self.timestep_embedder = _TimestepEmbedder(freq_dim, hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(t)


class _DiTTransformer(nn.Module):
    """Inner DiT transformer matching action_model.model.* keys."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.timestep_encoder = _TimestepEncoder(256, hidden_size)

        self.transformer_blocks = nn.ModuleList([
            _DiTBlock(
                hidden_size=hidden_size,
                context_dim=cross_attention_dim if (i % 2 == 0) else hidden_size,
                heads=num_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Output AdaLayerNorm + projection
        self.proj_out_1 = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_out_2 = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        for i, block in enumerate(self.transformer_blocks):
            ctx_for_block = ctx if (i % 2 == 0) else x
            x = block(x, ctx_for_block, t_emb)

        # Final AdaLayerNorm
        scale, shift = self.proj_out_1(t_emb).chunk(2, dim=-1)
        x = F.layer_norm(x, x.shape[-1:]) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj_out_2(x)  # (B, T, 1024)


class _ActionEncoder(nn.Module):
    """3-layer action encoder with timestep concatenation.

    Matches keys: action_encoder.layer1/2/3
    layer1: (action_dim → 768)  ← reinitialized for action_dim=6
    layer2: (1536 → 768)        — compatible with checkpoint
    layer3: (768 → 768)         — compatible with checkpoint
    """

    def __init__(self, action_dim: int, hidden_size: int = 768) -> None:
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, actions: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        x = F.silu(self.layer1(actions))                          # (B, T, 768)
        t = t_emb.unsqueeze(1).expand(-1, T, -1)
        x = F.silu(self.layer2(torch.cat([x, t], dim=-1)))        # (B, T, 768)
        return self.layer3(x)


class _StateEncoder(nn.Module):
    """2-layer state encoder.

    Matches keys: state_encoder.layer1/2
    layer1: (state_dim → 1024)  ← reinitialized for state_dim=6
    layer2: (1024 → 768)        — compatible with checkpoint
    """

    def __init__(self, state_dim: int, hidden_size: int = 768) -> None:
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 1024)
        self.layer2 = nn.Linear(1024, hidden_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.silu(self.layer1(state)))


class _ActionDecoder(nn.Module):
    """2-layer action decoder.

    Matches keys: action_decoder.layer1/2
    layer1: (1024 → 1024)      — compatible with checkpoint
    layer2: (1024 → action_dim) ← reinitialized for action_dim=6
    """

    def __init__(self, action_dim: int, hidden_size: int = 1024) -> None:
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.silu(self.layer1(x)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DiTActionHead(nn.Module):
    """DiT-B flow-matching action head (VLA-JEPA paper architecture).

    Module naming matches VLA-JEPA-pretrain.pt checkpoint keys:
      self.model.*            ← action_model.model.*
      self.action_encoder.*   ← action_model.action_encoder.*
      self.state_encoder.*    ← action_model.state_encoder.*
      self.action_decoder.*   ← action_model.action_decoder.*
      self.future_tokens      ← action_model.future_tokens
      self.position_embedding ← action_model.position_embedding

    Weights that must be reinitialized (dimension mismatch vs checkpoint):
      action_encoder.layer1   checkpoint: (768, 7) → ours: (768, action_dim=6)
      action_decoder.layer2   checkpoint: (7, 1024) → ours: (action_dim=6, 1024)
      state_encoder.layer1    checkpoint: (1024, 8) → ours: (1024, state_dim=6)
    """

    def __init__(
        self,
        action_dim: int = 6,
        chunk_size: int = 7,
        state_dim: int = 6,
        hidden_size: int = 768,
        cross_attention_dim: int = 2048,
        num_layers: int = 16,
        num_heads: int = 12,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 4,
        max_seq_len: int = 1024,
        num_future_tokens: int = 32,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s

        # Inner transformer (matches action_model.model.*)
        self.model = _DiTTransformer(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Encoder / decoder (match action_model.{action,state}_encoder/decoder.*)
        self.action_encoder = _ActionEncoder(action_dim, hidden_size)
        self.state_encoder = _StateEncoder(state_dim, hidden_size)
        self.action_decoder = _ActionDecoder(action_dim)

        # Learnable tokens and position embeddings
        self.future_tokens = nn.Embedding(num_future_tokens, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        nn.init.zeros_(self.action_decoder.layer2.weight)
        nn.init.zeros_(self.action_decoder.layer2.bias)

    def _get_dtype(self) -> torch.dtype:
        return self.action_encoder.layer1.weight.dtype

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        return self.model.timestep_encoder(t).to(self._get_dtype())

    def _velocity(
        self,
        noisy_actions: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self._get_dtype()
        noisy_actions = noisy_actions.to(dtype)
        state = state.to(dtype)
        ctx = ctx.to(dtype)

        t_emb = self._embed_timestep(t)                    # (B, 768)

        state_tok = self.state_encoder(state).unsqueeze(1)  # (B, 1, 768)
        x = self.action_encoder(noisy_actions, t_emb)       # (B, T, 768)
        x = torch.cat([state_tok, x], dim=1)                # (B, 1+T, 768)

        T = x.size(1)
        x = x + self.position_embedding.weight[:T].to(dtype)

        x = self.model(x, ctx, t_emb)                       # (B, 1+T, 1024)
        x = x[:, 1:, :]                                     # (B, T, 1024) — skip state token

        return self.action_decoder(x)                        # (B, T, action_dim)

    def _sample_timesteps(self, B: int, device: torch.device) -> torch.Tensor:
        t = torch.from_numpy(
            np.random.beta(self.noise_beta_alpha, self.noise_beta_beta, B).astype(np.float32)
        ).to(device)
        return t * self.noise_s  # (B,) in [0, ~1)

    def compute_loss(
        self,
        actions_norm: torch.Tensor,
        ctx: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        B = actions_norm.size(0)
        device = actions_norm.device

        t = self._sample_timesteps(B, device)
        noise = torch.randn_like(actions_norm)
        t_b = t.view(B, 1, 1)
        x_t = t_b * actions_norm + (1 - t_b) * noise
        target_v = actions_norm - noise

        t_scaled = (t * self.num_train_timesteps).long().float()
        pred_v = self._velocity(x_t, state, t_scaled, ctx)
        return F.mse_loss(pred_v.float(), target_v.float())

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B, device = ctx.size(0), ctx.device
        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / self.num_infer_timesteps
        for i in range(self.num_infer_timesteps):
            t_val = i / self.num_infer_timesteps
            t = torch.full((B,), t_val * self.num_train_timesteps, device=device)
            v = self._velocity(x, state, t, ctx)
            x = (x + dt * v).float()
        return x
