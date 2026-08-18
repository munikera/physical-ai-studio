# Physical Studio Pi0.5: What's In vs What's Missing

---

## What IS in Physical Studio's Pi0.5

### Architecture (matches Pi0.5 paper)
| Component | Detail |
|---|---|
| VLM backbone | PaliGemma 3B (SigLIP vision + Gemma 2B language) |
| Action expert | Gemma 300M with AdaRMS conditioning (time-step conditioning) |
| Action representation | Flow matching, 50-step chunks at 50 Hz |
| State encoding | Discretized into 256 bins, embedded in text: `"Task: X, State: 0 128 255...; Action:"` |
| Attention | Cross-attention between VLM prefix and action expert suffix |
| Normalization | QUANTILES (1st–99th percentile), robust to outliers |

### Physical Studio extras (not in original Pi0.5 paper)
| Feature | Config flag | What it does |
|---|---|---|
| **SnapFlow** | `snapflow_enabled` | Self-distillation for 1-step inference (10× faster). From arxiv 2604.05656 |
| **RTC (Real-Time Control)** | `enable_rtc` | Smooths action chunks across re-queries using KV cache prefix guidance |
| **n_action_steps < chunk_size** | `n_action_steps` | Execute fewer steps per chunk (e.g. 20) before re-querying model |
| **Gradient checkpointing** | `gradient_checkpointing` | Reduces VRAM at cost of recompute |
| **Freeze options** | `freeze_vision_encoder`, `train_expert_only` | Control which parts update during fine-tuning |

---

## What IS NOT in Physical Studio's Pi0.5

### 1. Subtask Prediction (Chain-of-Thought) — **Most critical for sorting task**

**What the paper does:**  
The model first autoregressively generates a text subtask (e.g., "pick up the block and place it in the left bin"), then uses that generated text as input context for the action expert's flow matching.

**What Physical Studio does:**  
The task text is provided as *input* from the dataset label — the model **never generates text**. It only takes `"Task: sort block by color, State: ..."` as a fixed string and goes directly to action denoising. If the task label doesn't say which bin, the model has no way to reason about it.

**Why this matters for your task:**  
Your episodes are labeled `"Pick up the block with blue X mark..."` — this gives the model the answer in the prompt. But the model cannot *figure out* which bin from the visual alone. A subtask-predicting model would look at the block, generate `"put in right bin"`, and then know which direction to move.

**No improvement number published** — Physical Intelligence treated this as a core architectural requirement, not an ablation.

---

### 2. Multi-Environment Training Data

**What the paper does:**  
Trains on data from many different homes, kitchens, and offices. This is the single biggest factor in the ablations.

**What Physical Studio does:**  
Fine-tunes on your specific workspace only. No built-in mechanism to mix in diverse environment data.

**Published improvement:** Removing this drops OOD success from **94% → 31%** (−63 percentage points).  
In-distribution also drops: **83% → 57%**.

---

### 3. Cross-Embodiment Data

**What the paper does:**  
Co-trains on data from multiple robot platforms.

**What Physical Studio does:**  
Single robot fine-tuning. The Pi0.5 base model weights do include cross-embodiment pretraining — this is inherited, not retrained.

**Published improvement:** Removing it drops OOD success from **94% → 49%** (−45 pp).

---

### 4. Web Data Co-Training

**What the paper does:**  
Mixes in internet-scale image captioning, VQA, and object detection data alongside robot demos.

**What Physical Studio does:**  
Training on robot demos only. Web data pretraining is baked into the PaliGemma base weights, so it is inherited at fine-tune time.

**Published improvement:** Removing it drops OOD success from **94% → 74%** (−20 pp). Mostly matters for novel object categories not seen in demos.

---

### 5. Verbal Instruction Co-Training

**What the paper does:**  
Includes demos where a human verbally coaches the robot step-by-step during execution (e.g., "now open the drawer"). These provide grounding between language commands and physical sub-actions.

**What Physical Studio does:**  
No special handling for this data type. Only standard teleoperated demos.

**Published improvement:** Not isolated in ablations.

---

## Summary Table

| Feature | In Physical Studio? | Impact if missing |
|---|---|---|
| PaliGemma 3B + Gemma 300M flow matching | Yes | — |
| AdaRMS time conditioning | Yes | — |
| SnapFlow 1-step inference | Yes (disabled by default) | — |
| RTC action smoothing | Yes | — |
| **Subtask prediction (chain-of-thought)** | **No** | Can't reason about which bin; no published number |
| Multi-environment training data | No (you have 1 workspace) | −63 pp OOD success |
| Cross-embodiment data | Inherited in weights | −45 pp OOD if fully absent |
| Web data co-training | Inherited in weights | −20 pp OOD |
| Verbal instruction data | No | Not measured |

---

## What This Means for Your Setup

The two things you can actually address:

**1. Subtask prediction (PI052)** — adds a text generation step before action denoising.  
For your task specifically: the model would learn to say "put in left bin" or "put in right bin" when it sees a red-X or blue-X block, then execute actions based on that text. This is the missing piece for reliable bin sorting.

**2. Data variation** — record demos in slightly different positions, lighting, and block placements. Even 10–20 more varied episodes per class would help generalization. This is low-effort and addresses the biggest measured gap.

Cross-embodiment and web data are already baked into the pretrained weights, so you have their benefit without doing anything.
