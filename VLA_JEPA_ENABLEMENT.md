# VLA-JEPA Enablement in Physical AI Studio

## References

- **Paper:** VLA-JEPA (ECCV 2026) — joint-embedding predictive architecture for visuomotor policy learning
- **Repo:** https://github.com/ginwind/VLA-JEPA
- **Checkpoint hub:** huggingface.co/ginwind/VLA-JEPA
  - `Pretrain/checkpoints/VLA-JEPA-pretrain.pt` — pretrained on DROID+SSv2 (what we use)
  - `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt` — fine-tuned on LIBERO
  - `Real-world/checkpoints/VLA-JEPA-Real-World.pt` — fine-tuned on Franka FR3

---

## Current Status (as of 2026-08-18)

**What we're trying now:** Training with unfrozen vision encoder (`freeze_qwen_vision=False`) on the `sorting-blocks-new-workspace-210` dataset (210 episodes). Gradient checkpointing was disabled to fix an XPU initialization hang (see Gotcha #11 below).

**Why:** Earlier runs with vision + LLM both frozen showed flat/slow loss on new datasets. The VLA-JEPA paper uses `freeze_modules: ''` (nothing frozen) for fine-tuning. With only 210 episodes we keep the LLM frozen (`freeze_qwen_llm=True`) but unfreeze vision.

**Current policy defaults:**
```python
freeze_qwen_vision: bool = False          # unfrozen — trains Qwen visual encoder
freeze_qwen_llm: bool = True              # frozen — saves memory/compute
enable_gradient_checkpointing: bool = False  # disabled — XPU hang with unfrozen vision
```

**Best completed run:** `vla-jepa-pretrain-b-3ep` — job `f4826c0d`, val/loss = **0.207** (3 epochs, `bcdf34ee` dataset, frozen vision+LLM)

---

## Overview

VLA-JEPA (ECCV 2026) integrates Qwen3-VL-2B as the vision-language backbone and a DiT-B flow-matching head for action generation, fine-tuned on SO-ARM 101 robot data. This document describes how it was integrated into Physical AI Studio on Intel XPU, and the full implementation history including the Option B pretrained checkpoint work.

**Current model (Option B — pretrained checkpoint):**
- Qwen3-VL-2B + 16-layer DiT-B action head (hidden_size=768)
- 244/248 action head weights loaded from `VLA-JEPA-pretrain.pt` (trained on DROID+SSv2)
- Vision encoder unfrozen, LLM frozen; action head + vision encoder trainable
- Dataset: 210-episode SO-ARM 101 sorting dataset (`sorting-blocks-new-workspace-210`)
- Training: **3 epochs, batch_size=4, bf16-mixed, Intel XPU (B70)**

---

## Architecture (Option B — Current)

```
Camera frame ─→ [resize 224×224] ─→ Qwen3-VL-2B (attn=sdpa)
                                            │
                   inject 32 <|embodied_action|> tokens at end of user prompt
                                            │
                          Qwen hidden states at those 32 positions
                                            │
                                   vl_embs (B, 32, 2048)
                                            │
state (B, 6) ──────────────→ FlowmatchingActionHead (DiT-B, 16 layers)
noisy_actions (B, 7, 6) ──→         cross-attend to vl_embs
                                            │
                                   action chunk (B, 7, 6)
```

**Key design choice:** V-JEPA is used for *pretraining* only (DROID+SSv2 video prediction). At inference, the conditioning signal comes from Qwen's `<|embodied_action|>` hidden states — no V-JEPA encoder needed at inference time. This simplifies the architecture significantly.

**Flow matching:** Beta noise schedule (α=1.5, β=1.0, s=0.999), linear interpolation `x_t = t·x₁ + (1-t)·noise`. 4-step Euler ODE at inference.

---

## Files (Host Source — Bind-Mounted into Container)

All policy source lives under `library/src/physicalai/policies/vla_jepa/`.

| File | Purpose |
|------|---------|
| `policy.py` | Lightning `Policy` subclass — training loop, data preprocessing, optimizer, checkpoint loading |
| `model.py` | `VlaJepaModel` — loads Qwen3-VL, injects embodied tokens, builds FlowmatchingActionHead |
| `config.py` | `VlaJepaConfig` — all hyperparameters including `pretrained_checkpoint` path |
| `flow_matching_head.py` | `FlowmatchingActionHead` — **checkpoint-compatible** DiT-B (244/248 keys load from paper ckpt) |
| `dit_action_head.py` | `DiTActionHead` + shared sub-modules imported by `flow_matching_head.py` |
| `pretrained_utils.py` | `load_vla_jepa_pretrain()` — key remapping + shape-mismatch-tolerant loader |
| `__init__.py` | Exports `VlaJepa` |

`library/src/physicalai/policies/__init__.py` — added `VlaJepa` to public exports.

---

## Files Modified (Container Writable Layer)

Must be re-patched if the container image is rebuilt.

| Container Path | Change |
|----------------|--------|
| `/app/application/backend/src/api/policies.py` | Added `VlaJepa` import and `"vla_jepa": VlaJepa` entry |
| `/app/application/backend/src/trainer/schemas.py` | Added `"vla_jepa"` to `_SUPPORTED_POLICIES` |
| `/app/application/backend/src/services/model_import_service.py` | Added `"vla_jepa"` to `_SUPPORTED_POLICIES` |
| `/app/application/backend/src/training/job.py` | Strip `compile_model` kwarg for vla_jepa (see Gotcha #1) |
| `/app/application/ui/dist/static/js/index.*.js` | Added VLA-JEPA to compiled `MODELS` array |

---

## Files Modified (Host Source)

| Path | Change |
|------|--------|
| `application/ui/src/routes/models/train-model-dialog.tsx` | Added VLA-JEPA to `MODELS` array |
| `application/docker/docker-compose.yaml` | Removed `:ro` from `library/src` bind mount |

---

## Checkpoint Paths (Inside Container)

| Model | Path |
|-------|------|
| Qwen3-VL-2B-Instruct | `/checkpoints/Qwen3-VL-2B-Instruct` |
| V-JEPA 2 ViT-L (not used at inference) | `/checkpoints/vjepa2-vitl-fpc64-256` |
| **VLA-JEPA pretrain** (DROID+SSv2) | `/checkpoints/vla-jepa-pretrain/Pretrain/checkpoints/VLA-JEPA-pretrain.pt` |

---

## Training Runs

### Run 1 — Option A (no pretrained checkpoint)
- **Job ID:** `547976cc-4ee3-4fef-8e9d-e3508b953dd4` (cancelled)
- Architecture: V-JEPA + Qwen + DiT (old, random init)
- Loss at step 2520: ~0.7 — but this was already into training, initial loss was ~3+
- Cancelled to implement Option B

### Run 2 — Option B (pretrained checkpoint)
- **Job ID:** `11ed2b24-2632-4f50-a381-c6751bf3c63c`
- **Dataset:** `bcdf34ee-97b1-4c89-8be7-c3045b24ae49` (so101_sorting_colored_blocks, 222 episodes)
- **Epochs:** 50
- **Batch size:** 4
- **Precision:** bf16-mixed
- **Pretrained checkpoint:** `VLA-JEPA-pretrain.pt` (trained on DROID 76k+ episodes + SSv2 220k clips)
- **Loaded weights:** 244/248 action head keys + ~625 Qwen keys
- **Loss trajectory:** step 1: 3.93 → step 100: 2.58 → step 200: 2.36 → step 300: 2.14 → step 400: 1.52
- **Freeze config:** `freeze_qwen_vision=True`, `freeze_qwen_llm=True` (only action head trainable)

### Run 3 — vla-jepa-pretrain-b-3ep (3 epochs, completed ✓)
- **Job ID:** `f4826c0d`
- **Dataset:** `bcdf34ee-97b1-4c89-8be7-c3045b24ae49` (222 episodes)
- **Epochs:** 3
- **Batch size:** 4
- **Precision:** bf16-mixed
- **Freeze config:** `freeze_qwen_vision=True`, `freeze_qwen_llm=True`
- **Result:** `val/loss = 0.207` — best completed run
- **Model export:** Manually exported to `/app/storage/models/3144ef3b-8895-4f62-b8f3-0e731f0b699c/exports/torch/vlajepa.pt` (4.6 GB) with `manifest.json`. Future runs export automatically (ExportablePolicyMixin added).

### Run 4 — Retrain attempt on new dataset (failed — flat loss)
- **Dataset:** New workspace dataset (different from `bcdf34ee`)
- **Issue:** Retrain from checkpoint had a bug: `setup()` was re-loading `VLA-JEPA-pretrain.pt` on top of fine-tuned weights, overwriting them. Fixed (see Gotcha #10).
- **Observation:** Loss appeared flat. Likely because both vision and LLM were frozen — the model had no way to adapt the feature extraction to the new dataset.

### Run 5 — sorting-blocks-new-workspace-210 (in progress)
- **Dataset:** `sorting-blocks-new-workspace-210` (210 episodes, 265 MB)
- **Epochs:** 3
- **Batch size:** 4
- **Precision:** bf16-mixed
- **Freeze config:** `freeze_qwen_vision=False`, `freeze_qwen_llm=True`
- **Gradient checkpointing:** `False` (disabled to fix XPU hang — see Gotcha #11)
- **Status:** Starting fresh after container restart

---

## SO-ARM 101 Configuration

- `action_dim = 6` (6 joints; gripper is joint 6, 0-indexed as dim 5)
- `state_dim = 6`
- `chunk_size = 7` (predict 7 future actions)
- `n_action_steps = 7` (execute all 7 predicted actions before re-planning — matches LeRobot/paper default)
- `binarize_gripper_action = True` (gripper dim 5 binarized to {0,1} post-denormalization — matches LeRobot)
- `gripper_dim = 5`, `gripper_threshold = 0.5`

---

## Recommended Training Settings

| Setting | Value | Notes |
|---------|-------|-------|
| `batch_size` | 4 | Higher may OOM on B70 |
| `max_epochs` | 50 | ~7600 steps on 222-episode dataset |
| `bf16` | true | **Required** |
| `compile_model` | false | VlaJepa does not support torch.compile |
| `pretrained_checkpoint` | `/checkpoints/vla-jepa-pretrain/Pretrain/checkpoints/VLA-JEPA-pretrain.pt` | Set in `policy_kwargs` |
| `freeze_qwen_vision` | false | Unfreezing helps on new datasets; keep LLM frozen |
| `freeze_qwen_llm` | true | Freeze LLM — saves memory, stable with <500 episodes |
| `enable_gradient_checkpointing` | false | **Must be false** when `freeze_qwen_vision=False` — XPU hang otherwise |

**Paper comparison:** VLA-JEPA paper freezes nothing for LIBERO fine-tune. With small datasets (~200 episodes) we freeze the LLM only (`freeze_qwen_llm=True`, `freeze_qwen_vision=False`).

---

## Pretrained Checkpoint — Key Remapping

`pretrained_utils.load_vla_jepa_pretrain()` remaps checkpoint keys to our module names:

| Checkpoint prefix | Our module |
|-------------------|-----------|
| `qwen_vl_interface.model.model.*` | `qwen.*` (note: two `.model.` levels stripped) |
| `qwen_vl_interface.model.lm_head.*` | skipped (ForCausalLM head, not in AutoModel) |
| `action_model.*` | `action_head.*` |
| `vj_encoder.*` | skipped (V-JEPA not used at inference) |

**Shape mismatches intentionally skipped** (reinitialized for SO-ARM 101):

| Key | Checkpoint shape | Our shape | Reason |
|-----|----------------|-----------|--------|
| `action_encoder.layer1.weight` | (768, 7) | (768, 6) | action_dim 7→6 |
| `action_decoder.layer2.weight` | (7, 1024) | (6, 1024) | action_dim 7→6 |
| `action_decoder.layer2.bias` | (7,) | (6,) | action_dim 7→6 |
| `state_encoder.layer1.weight` | (1024, 8) | (1024, 6) | state_dim 8→6 |

---

## DiT-B Architecture Details

`FlowmatchingActionHead` module naming matches `VLA-JEPA-pretrain.pt` exactly:

```
model.timestep_encoder.timestep_embedder.linear_1/2
model.transformer_blocks.{0..15}.norm1.{norm, linear}
model.transformer_blocks.{0..15}.attn1.{to_q, to_k, to_v, to_out.0}
model.transformer_blocks.{0..15}.ff.net.{0.proj, 2}
model.proj_out_1 / proj_out_2
action_encoder.layer{1,2,3}
state_encoder.layer{1,2}
action_decoder.layer{1,2}
future_tokens
position_embedding
```

**Sequence layout** fed to DiT: `[state_token(1) | future_tokens(32) | action_tokens(chunk_size=7)]` = 40 tokens. Cross-attention (even blocks) attends to `vl_embs` (Qwen embodied token hidden states). Self-attention (odd blocks) attends to sequence itself.

---

## Intel XPU Notes

- `attn_implementation="sdpa"` set when loading Qwen3-VL (`flash_attention_2` is CUDA-only)
- DiT attention uses `F.scaled_dot_product_attention` (replaced manual softmax-based attention)
- Training uses `SingleXPUStrategy` (auto-detected)
- Triton unavailable on XPU — flop counting disabled (harmless warning)

---

## Key Gotchas Resolved During Integration

### 0. Model export not shown in UI (ExportablePolicyMixin missing)

The UI shows available model formats (PyTorch, OpenVINO) only for policies that inherit `ExportablePolicyMixin`. Without it, `_export()` in `job.py` skips export entirely. Fixed by adding:

```python
from physicalai.export import ExportablePolicyMixin
from physicalai.export.backends import ExportBackend
from physicalai.inference.data import InferenceFeature
from physicalai.inference.data.features import InferenceFeatureDtype, InferenceFeatureType

class VlaJepa(ExportablePolicyMixin, Policy):
    @staticmethod
    def get_supported_export_backends() -> list[ExportBackend]:
        return [ExportBackend.TORCH]  # OpenVINO not possible (dynamic shapes)

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None: ...

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None: ...
```

Note: OpenVINO is NOT supported — VlaJepa uses dynamic tokenization and variable sequence lengths.

### 1. `compile_model` kwarg rejected by VlaJepa

The framework's `build_policy()` always passes `compile_model=False`. Fixed in `job.py`:

```python
if spec.policy.lower() == "vla_jepa":
    kwargs = {}  # VlaJepa does not accept compile_model
```

### 2. Lightning `save_hyperparameters` frame inspection failure

Fixed by passing an explicit dict:

```python
self.save_hyperparameters({"qwen_model_name": qwen_model_name, ...})
```

### 3. TrainingWorker module caching

The TrainingWorker is a **persistent long-running subprocess**. Code changes are not picked up until the container is restarted:

```bash
rm -f library/src/physicalai/policies/vla_jepa/__pycache__/*.pyc
docker restart physical-ai-studio-xpu
```

### 4. Qwen3VLConfig.hidden_size is nested

```python
qwen_hidden = self.qwen.config.text_config.hidden_size  # 2048 (NOT config.hidden_size)
```

### 5. Qwen image tokenization overflow

640×480 raw images → ~300 tokens, exceeding `tokenizer_max_length=256`. Fix: resize to 224×224 before Qwen, set `tokenizer_max_length=512`.

### 6. BFloat16 → NumPy conversion

NumPy doesn't support bfloat16. Cast before `.numpy()`:

```python
frame_np = frame.permute(1, 2, 0).cpu().float().numpy()
```

### 7. DiT dtype mismatches

Lightning bf16-mixed casts model weights but leaves intermediates (noise, sinusoidal embeddings) as float32. Cast all inputs at `_velocity` entry and compute MSE loss in float32:

```python
noisy = noisy.to(self._get_dtype())
return F.mse_loss(pred_v.float(), target_v.float())
```

### 8. Qwen processor `images` argument

`qwen_processor(images=...)` expects PIL images, not content dicts:

```python
images=[[item["image"] for item in m[0]["content"] if item["type"] == "image"]
        for m in messages_batch]
```

### 9. `bf16-true` dtype mismatches (three layers)

Training with `bf16-true` (as opposed to `bf16-mixed`) casts all weights to bf16 but does NOT wrap forward passes in `torch.autocast`. Dataset tensors arrive as float32 and hit bf16 weight matrices. Three separate fixes were needed:

**Fix 1 — Qwen inputs** (`model.py:_encode_with_embodied_tokens`):
```python
qwen_dtype = next(self.qwen.parameters()).dtype
inputs = {k: v.to(qwen_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
          for k, v in inputs.items()}
```

**Fix 2 — action/state tensors** (`model.py:forward`):
```python
mdtype = vl_embs.dtype
actions_norm = actions_norm.to(mdtype)
state_input = state_norm.unsqueeze(1).to(mdtype)
```

**Fix 3 — action decoder cast** (`flow_matching_head.py`):
```python
# Was: pred = self.action_decoder(out.float())
pred = self.action_decoder(out)  # remove .float() — let decoder use model dtype
```
MSE loss still uses `.float()` which is correct: `F.mse_loss(pred_actions.float(), velocity.float())`

### 10. Retrain overwrites fine-tuned weights

When retraining from a Lightning checkpoint via the UI, `setup()` was unconditionally re-loading `VLA-JEPA-pretrain.pt`, overwriting all fine-tuned weights.

**Root cause:** The framework calls `policy_class.load_from_checkpoint()` then `trainer.fit()` without `ckpt_path`, so `self.trainer.ckpt_path` is always `None` at `setup()` time — the first attempted fix was wrong.

**Correct fix:** Detect retrain by checking `self.model is not None` at the start of `setup()`. When `load_from_checkpoint()` restores `dataset_stats` into hparams, `_build_model` is called in `__init__`, so `self.model` is already built before `setup()` runs. Fresh training enters `setup()` with `self.model=None`.

```python
loaded_from_checkpoint = self.model is not None
if self.model is not None:
    self.model.set_dataset_stats(stats)
else:
    self._build_model(stats)
if self.config.pretrained_checkpoint and not loaded_from_checkpoint:
    self._load_pretrained_checkpoint()
```

### 11. XPU hang with unfrozen vision encoder + gradient checkpointing

When `freeze_qwen_vision=False` (vision encoder trainable), enabling gradient checkpointing (`gradient_checkpointing_enable()`) causes a silent hang on Intel XPU. Thread 1950 runs Python code for 12+ minutes with no log output and 0 GB XPU memory allocated. Root cause: Intel XPU appears to trigger long first-run kernel compilation when setting up autograd hooks for the visual encoder backward graph.

**Fix:** Set `enable_gradient_checkpointing=False` (default changed in `policy.py`). The B70 has 32 GB XPU memory — sufficient headroom to hold all activations without checkpointing.

### 12. Pretrained checkpoint Qwen key prefix

The checkpoint wraps Qwen as `AutoModelForCausalLM`, which adds an extra `.model.` level:

```
qwen_vl_interface.model.model.visual.*   →   qwen.visual.*
```
The initial remapping used `qwen_vl_interface.model.` (missing one level). Corrected to `qwen_vl_interface.model.model.`.

---

## Paper Training Regime (Reference)

The `VLA-JEPA-pretrain.pt` checkpoint was produced by:
1. **Stage 1 — Co-train:** DROID (76k+ episodes) + SSv2 (220k video clips), 50,000 steps on 8 GPUs
2. **Stage 2 — Fine-tune:** LIBERO (~2,100 episodes), 30,000 steps on 8 GPUs

Available checkpoints at `huggingface.co/ginwind/VLA-JEPA`:

| Checkpoint | Contents |
|------------|----------|
| `Pretrain/checkpoints/VLA-JEPA-pretrain.pt` | After stage 1 (DROID+SSv2) |
| `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt` | Fine-tuned on LIBERO |
| `Real-world/checkpoints/VLA-JEPA-Real-World.pt` | Fine-tuned on Franka FR3 |

We use the **pretrain** checkpoint as our starting point, then fine-tune on 222 SO-ARM 101 episodes. The paper freezes nothing (all params trainable), but with only 222 episodes we freeze Qwen vision + LLM.
