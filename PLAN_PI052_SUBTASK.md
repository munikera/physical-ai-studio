# Plan: PI052 Subtask Prediction for SO-ARM101 Sorting

Goal: Improve red-X vs blue-X bin sorting accuracy by adding subtask prediction
(PI052 architecture) to the Pi0.5 model trained in Physical Studio.

---

## Phase 1: Video Annotation with Qwen3-VL

Annotate each episode with subtask timestamps so the PI052 model can learn
to predict the current subtask alongside actions.

### 1a. Test Qwen3-VL-30B-A3B via vLLM (tensor parallel, both B70 cards)

vLLM 0.14.1 supports `Qwen3VLMoeForConditionalGeneration` (the MoE architecture
of Qwen3-VL-30B-A3B). Both B70 cards provide 2×32 GB = 64 GB total, enough for
this model in bf16 (~30B active params, MoE so actual weight size is ~30B but
active compute is ~3B).

**Steps:**
1. Check if `Qwen/Qwen3-VL-30B-A3B-Instruct` fits in 64 GB with TP=2
   - Total model weights: ~60 GB in bf16 — tight, may need INT8/INT4
   - With INT4 AWQ: ~15 GB — easily fits on one card
2. Run vLLM server with TP=2 (or TP=1 + INT4):
   ```bash
   conda run -n vllm-xpu python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen3-VL-30B-A3B-Instruct \
     --tensor-parallel-size 2 \
     --dtype bfloat16 \
     --max-model-len 8192 \
     --port 8001
   ```
3. Send one episode video (frames at 1 FPS) to the server and check output quality
4. Compare to Qwen2.5-Omni-7B results (block color detection, subtask granularity)

**Decision gate:** If Qwen3-VL-30B gives better block color detection and
more accurate subtask transitions than Qwen2.5-Omni-7B → use it for full annotation.

### 1b. Fallback: Manual annotation

If VLM annotation quality is insufficient (can't reliably identify red vs blue,
or subtask transitions are wrong), annotate manually:
- Watch each episode video, record frame ranges for each subtask
- Use the provided episode→color mapping (already done for 222 episodes)
- Subtask labels per episode: approach, grasp, lift, move-to-bin, release, return

### 1c. Full dataset annotation (222 episodes)

Once annotation quality is validated on 3–5 episodes:
- Run annotation script on all 222 episodes in `bcdf34ee` dataset
- Write subtask labels to `episodes/chunk-000/file-*.parquet` (new `subtask` column)
- Push annotated dataset to HuggingFace as `muniker/sorting-blocks-subtask-222`

---

## Phase 2: Train PI052 via lerobot CLI

PI052 is implemented in lerobot branch `codex/pi052-language-policy` (PR #4184).
Train directly with lerobot CLI, bypassing Physical Studio UI.

**Steps:**
1. Clone lerobot PI052 branch:
   ```bash
   git clone --branch codex/pi052-language-policy \
     https://github.com/huggingface/lerobot.git /home/devcloud/munikera/lerobot-pi052
   ```
2. Install in a new conda env (or reuse `vllm-xpu` with XPU torch):
   ```bash
   conda create -n lerobot-pi052 python=3.11
   conda activate lerobot-pi052
   pip install -e /home/devcloud/munikera/lerobot-pi052[xpu]
   ```
3. Run training with annotated dataset:
   ```bash
   python -m lerobot.scripts.train \
     --config-name=pi052 \
     dataset_repo_id=muniker/sorting-blocks-subtask-222 \
     training.num_steps=30000 \
     device=xpu
   ```
4. Evaluate on robot — does it pick correct bin?

**Success criterion:** Robot places red-X blocks in left bin and blue-X blocks
in right bin with >80% accuracy over 20 test trials.

---

## Phase 3: Integrate PI052 into Physical Studio

Make the PI052 model trainable and deployable from the Physical Studio UI.

### Changes needed in `pi05-training-on-arc-pro` branch:

1. **`library/src/physicalai/policies/pi05/config.py`**
   - Add `use_subtask_prediction: bool = False`
   - Add `subtask_vocab: list[str]` (list of subtask names)
   - Add `subtask_loss_weight: float = 0.1`

2. **`library/src/physicalai/policies/pi05/model.py`**
   - Add LM head for subtask classification (reuse Gemma 300M token embeddings)
   - Add cross-entropy loss on subtask token predictions
   - "Knowledge insulation" from PI052: stop-gradient between subtask head and action expert

3. **`library/src/physicalai/policies/pi05/policy.py`**
   - Accept `subtask` column in batch during training
   - During inference: predict subtask alongside actions (for logging/debugging)

4. **`library/configs/physicalai/pi05.yaml`**
   - Add `use_subtask_prediction: true` option

5. **Export (`manifest.json`):**
   - Add subtask prediction head to OpenVINO export
   - UI can display current predicted subtask in real time

### Deployment:
- Train via Physical Studio UI (will automatically use new config)
- Export to OpenVINO as usual
- Manifest will include subtask output for optional display

---

## Status

- [x] Qwen2.5-Omni-7B test annotation on episode 0 — ran, quality issues (color unknown, subtasks coarse)
- [ ] Phase 1a: Test Qwen3-VL-30B-A3B via vLLM TP=2
- [ ] Phase 1b: Manual annotation (fallback)
- [ ] Phase 1c: Full 222-episode annotation
- [ ] Phase 2: lerobot PI052 training
- [ ] Phase 3: Physical Studio integration
