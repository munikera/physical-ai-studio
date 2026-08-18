# VLA-JEPA Integration Plan for Physical AI Studio

## Overview

Integrate [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) (ECCV 2026) into `physical-ai-studio`
so you can fine-tune it on your 200-episode SO-ARM 101 dataset and run inference.

**VLA-JEPA components:**
- Qwen3-VL-2B-Instruct — vision-language backbone
- V-JEPA 2 ViT-L (fpc64-256) — video encoder / world model
- DiT-B — flow-matching action head (7-DOF, compatible with SO-ARM 101)

---

## Hardware

| Resource | Value |
|---|---|
| Device | Intel Arc Pro B60 / B70 (XPU) |
| VRAM per card | ~32 GB |
| Active card | `ZE_AFFINITY_MASK=0` (GPU 0) |
| Key XPU fix | `attn_implementation="sdpa"` (flash_attention_2 not supported on XPU) |

---

## File Layout

```
library/src/physicalai/policies/vla_jepa/
├── __init__.py           # exports VlaJepa, VlaJepaConfig, VlaJepaModel
├── config.py             # frozen dataclass with all hyperparams
├── model.py              # nn.Module wrapping Qwen3-VL + V-JEPA + DiT action head
├── policy.py             # Lightning module (training + inference)
└── pretrained_utils.py   # weight loading helpers

library/configs/physicalai/vla_jepa.yaml   # training config for SO-ARM 101
library/src/physicalai/policies/__init__.py  # add VlaJepa entry
```

---

## Phase 1 — Model Wiring ✅

**Goal:** Forward pass works on XPU; models load without OOM.

Key decisions:
- `attn_implementation="sdpa"` when loading Qwen3-VL-2B (flash_attention_2 crashes on XPU)
- V-JEPA encoder loaded from local checkpoint (HF format): `vjepa2-vitl-fpc64-256`
- `num_frames=8` for V-JEPA encoder (config default; lower than pretrained 64-frame training but fits 32 GB)
- Sequential CPU offloading if both Qwen3-VL + V-JEPA are too large to load simultaneously

Files: `config.py`, `model.py`

---

## Phase 2 — Fine-tuning ✅

**Goal:** `physicalai fit --config configs/physicalai/vla_jepa.yaml` trains on your LeRobot dataset.

Key decisions:
- Use `physicalai.data.lerobot.LeRobotDataModule` — already supports LeRobot format
- `embodiment_tag: NEW_EMBODIMENT` for SO-ARM 101 (not a known robot type)
- Freeze V-JEPA encoder and Qwen3-VL vision tower; train action head + projectors only
- Differential learning rates: VLM backbone 1e-5, action head 1e-4
- `action_dim: 7` (6 joints + gripper = SO-ARM 101 native DOF)
- `gradient_checkpointing: true` to reduce activation memory

Files: `policy.py`, `vla_jepa.yaml`

---

## Phase 3 — Inference

**Goal:** Load a checkpoint and call `predict_action_chunk(obs)` for real-time robot control.

Key decisions:
- `policy.predict_action_chunk(obs)` returns `(chunk_size, 7)` action tensor
- `n_action_steps=1` for reactive control; increase for smoother trajectories
- Normalize/denormalize actions using dataset statistics from training

---

## Quick-start

### 1. Download model checkpoints

```bash
# V-JEPA 2 ViT-L (HF format, ~1.3 GB)
huggingface-cli download facebook/vjepa2-vitl-fpc64-256 \
  --local-dir /home/devcloud/munikera/checkpoints/vjepa2-vitl-fpc64-256

# Qwen3-VL-2B (HF format, ~4.5 GB)
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /home/devcloud/munikera/checkpoints/Qwen3-VL-2B-Instruct
```

### 2. Prepare dataset

Your Physical AI Studio dataset is already in LeRobot format.
Set `repo_id` in `vla_jepa.yaml` to your local dataset path.

### 3. Fine-tune

```bash
cd /home/devcloud/munikera/physical-ai-studio/library
ZE_AFFINITY_MASK=0 physicalai fit --config configs/physicalai/vla_jepa.yaml
```

### 4. Inference

```python
from physicalai.policies import VlaJepa
policy = VlaJepa.load_from_checkpoint("checkpoints/vla_jepa/last.ckpt")
action = policy.predict_action_chunk(obs)  # shape: (chunk_size, 7)
```

---

## Known Issues / TODOs

- [ ] `attn_implementation="sdpa"` must be set; the default `flash_attention_2` crashes on XPU
- [ ] V-JEPA encoder uses `num_frames=8`; model was pretrained on 64 frames — may need to fine-tune for best results
- [ ] VLA-JEPA repo does not yet publish pretrained robot checkpoints; fine-tuning from scratch on your 200 episodes
- [ ] Export to OpenVINO IR not implemented in Phase 1 (can be added later following SmolVLA pattern)
