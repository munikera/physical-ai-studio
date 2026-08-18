# Pi0.5 Strategies, Missing Components & Measured Improvements

Sources:
- Pi0.5 blog: https://www.pi.website/blog/pi05
- Pi0 blog: https://www.pi.website/blog/pi0
- Pi0.5 paper: https://arxiv.org/abs/2504.16054

---

## What Pi0.5 Is

Built on Pi0 (3B VLM + flow matching action expert), extended with a co-training
framework designed for generalization to entirely new homes and environments.
Evaluated on long-horizon tasks (make bed, put dishes in sink, fold clothes, clean
floor, wipe spills) in homes never seen during training.

---

## Strategies With Measured Improvements

### 1. Multi-Environment Data (biggest single factor)

Training on data from many different environments (homes, kitchens, offices) is the
largest contributor to generalization.

| Condition | OOD Follow Rate | OOD Success Rate | In-Dist Success |
|---|---|---|---|
| Pi0.5 (full) | **94%** | **94%** | **83%** |
| No multi-environment data | 33% | 31% | 57% |
| No cross-embodiment data | 67% | 49% | 67% |
| No web data | 80% | 74% | 82% |

**Impact of removing multi-environment data: −61 pp OOD success (94% → 31%)**

> The blog notes: "after only about 100 training environments, it actually
> approaches the performance of the baseline model that was trained on the test
> environment directly."

### 2. Cross-Embodiment Data

Data from multiple robot platforms (not just the deployment robot).

**Impact: −45 pp OOD success (94% → 49%), −16 pp in-distribution (83% → 67%)**

### 3. Web Data (image captioning, VQA, object detection)

Internet-scale vision-language data co-trained alongside robot demos.

**Impact: mostly OOD — −20 pp OOD success (94% → 74%), only −1 pp in-distribution**
Matters most for unseen object categories and novel environments.

### 4. Subtask Prediction / High-Level Chain-of-Thought

The model first outputs a natural-language subtask label (e.g., "pick up the sponge"),
then uses that as context for low-level flow-matching motor commands. This is the
"chain-of-thought" component mentioned in the paper.

**No isolated ablation number is published.** The architecture is described and
presented as necessary for long-horizon task success, but the blog/paper does not
report a "with vs. without subtask prediction" comparison. It is treated as a
core design choice rather than an optional add-on.

### 5. Verbal Instruction Co-Training

Human coaching data: a person verbally guides the robot step by step during demos
(e.g., "now close the drawer"). This is included in the training mixture.

**No isolated ablation number published.** Included as part of the multi-modal
training mixture; not broken out separately in ablations.

---

## Pi0 vs Pi0.5: Architecture Comparison

| Aspect | Pi0 | Pi0.5 |
|---|---|---|
| VLM backbone | PaliGemma 3B | PaliGemma 3B |
| Action expert | Gemma 300M flow matching | Same |
| Subtask prediction | No | Yes (text output before action) |
| Training data | Fixed multi-robot set | Co-training with web + multi-env + cross-embodiment |
| Evaluation | Near-training environments | Entirely new homes (OOD) |
| OOD success | Not reported separately | 94% |

---

## Pi0 Baseline Numbers (for context)

From the Pi0 blog, on standard manipulation benchmarks (normalized 0–1):

| Task | Pi0 | Pi0-small (no VLM) | OpenVLA | Octo |
|---|---|---|---|---|
| Bussing Easy (UR5e) | **0.971** | 0.443 | 0 | 0.043 |
| Bussing Hard (UR5e) | **0.875** | 0.333 | 0 | 0 |
| Shirt Folding | **1.000** | 0.500 | 0 | 0 |
| Grocery Bagging | **0.786** | 0.271 | 0 | 0 |
| Toast from Toaster | **0.750** | 0 | 0 | 0 |

**VLM pretraining (3B vs 470M) gives >2× improvement** over the smaller model.

---

## Relevance to the SO-ARM101 Sorting Task

| Strategy | Applicable to our setup? | Notes |
|---|---|---|
| Multi-environment data | Partially | We have 1 workspace; adding variation (lighting, block positions) would help |
| Cross-embodiment data | No — would require retraining from scratch | Not practical for fine-tuning |
| Web data | Already in Pi0.5 base | Inherited from pretraining |
| Subtask prediction | **Yes — this is PI052** | No published improvement number, but core to long-horizon sorting (red→left, blue→right) |
| Verbal instructions | Possible to add | Could add coaching demos to dataset |

### Why subtask prediction matters for bin sorting

The blog describes Pi0.5 using subtask text output as an intermediate step
before generating motor commands. For our task, the subtask would encode
"put block in left bin" vs "put block in right bin" — giving the action expert
explicit context about which bin is correct for the current block color.
Without this, the model must infer bin choice purely from visual features in
a single forward pass, which our results show is unreliable.

**Conclusion**: Subtask prediction is the most actionable missing piece for
improving sorting accuracy. Multi-environment data variation is the second.
