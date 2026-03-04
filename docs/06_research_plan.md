# Research Plan: Lightweight Pluggable ICL Module for Traffic Foundation Models

## Paper Framing

**Title (working)**: *Lightweight In-Context Learning for Pretrained Spatio-Temporal Foundation Models*

**Core Contribution**: A training-free residual correction framework enhanced by (1) embedding-space demo retrieval and (2) a lightweight learned aggregation module (~200K params), that plugs into any frozen traffic foundation model to enable in-context adaptation — no fine-tuning, no architecture modification.

**Format**: 8–10 page conference-style course project report.

---

## 1. Method Overview

### 1.1 Residual Correction Framework

Given a pretrained model $f$ (frozen), query history $x_q$, and K demonstration pairs $(x_k, y_k)$:

$$\hat{y}_q = f(x_q) + \text{Aggregate}\bigl(\{y_k - f(x_k)\}_{k=1}^K\bigr)$$

Each correction $c_k = y_k - f(x_k)$ captures the model's systematic error on a training sample similar to the query. The aggregation can be:
- **Naive average**: $(1/K)\sum_k c_k$
- **Learned**: $\text{Aggregator}(h_q, \{h_k\}, \{c_k\})$ where $h$ are encoder features

### 1.2 Two Optimization Axes

| Axis | Current | This Report | Future Directions (§6) |
|------|---------|------------|-------------------------|
| **Retrieval** | Raw-input L2 KNN | Encoder embedding-space KNN | Learned metric, temporal-aware, contrastive |
| **Aggregation** | Multi-head cross-attention (198K) | + Simple cosine (8K) | MLP, Gated, graph-aware, iterative |

---

## 2. Experiment Plan (aligned with paper sections)

> See `docs/07_paper_framework.md` for the full paper skeleton and experiment→section mapping.

### Phase 1: Baseline Completion → Paper §4.2 Table 1, §4.4 Table 3 (~5h GPU) **[P0]**

**Goal**: Get exact metrics for existing aggregator configurations.

| ID | Experiment | Paper target | GPU time | Status |
|----|-----------|-------------|----------|--------|
| 1a | Rerun Exp 8 test (attention, 198K, 10ep) | Table 3 "Attention (10ep)" | ~2h | **Done**: MAE=4.29 |
| 1a' | Retrain attention aggregator (3 epochs) for fair comparison with OpenCity few-shot | Table 1 "Ours", Table 3 "Attention (3ep)" | ~1h | PENDING |
| 1b | Train + test SimpleDemoAggregator (8K) | Table 3 "Cosine" | ~3h | PENDING |

> **Fair comparison note**: OpenCity few-shot trains for 3 epochs. Our current 10-epoch result (MAE=4.29) uses more training budget. Experiment 1a' (3-epoch) provides the fair comparison; 10-epoch result serves as an upper bound showing aggregator capacity.

### Phase 2: Embedding-Space Retrieval → Paper §4.3 Table 2 (~6h GPU) **[P0]**

**Goal**: Replace raw-input KNN with encoder-feature KNN (node-mean pooling).

**Implementation steps**:
1. One-time precompute: frozen `_forward_with_features()` over demo pool → `[5761, N, D]` (N=228, D=512)
2. Node-mean pooling → `[5761, 512]`, build KNN index (sklearn)
3. Add `demo_selection='embedding'` in `ict_data_process.py`

> **Why node-mean only**: Full embedding `[228×512]=117K` dims suffers from the curse of dimensionality — KNN quality may actually degrade. Node-mean (512-d) is a standard pooling approach with reasonable dimensionality.

**P0 experiments**:

| ID | Retrieval | Aggregation | K | Paper target |
|----|-----------|------------|---|-------------|
| 2a | Raw KNN | Naive avg | 3 | Table 2 row 2 (done: 4.66) |
| 2b | Enc-KNN (node-mean) | Naive avg | 3 | Table 2 row 3 |
| 2c | Best Enc-KNN | Attention agg | 3 | **Table 1 "Ours" row** |

**Retrieval directions to explore** (try after P0, ordered by priority):

| # | Direction | Idea | Difficulty | GPU cost |
|---|-----------|------|------------|----------|
| R1 | **Temporal-aware re-ranking** | After KNN retrieval, re-rank top-2K candidates with same-hour-of-day bonus, take top-K: $s' = s_{\text{embed}} + \lambda \cdot \mathbb{1}[\text{same\_hour}]$ | Low (modify scoring) | 0 (CPU only) |
| R2 | **Learned projection head** | Add `Linear(512→128)` on frozen encoder output, train with contrastive loss (positives = demo pairs with low correction MAE) | Medium (define pos/neg + train) | ~1h |
| R3 | **Dynamic K (threshold)** | Set similarity threshold τ, only select demos with $s(q,d)>\tau$; grid search τ on val set | Low (modify selection logic) | ~1h (multiple tests) |
| R4 | **PCA-reduced KNN** | PCA on encoder features to 64/128 dims before KNN — may outperform raw 512-d | Low (sklearn PCA) | 0 (CPU only) |

> **Recommended order**: R1 → R4 → R3 → R2. R1 and R4 have zero GPU cost and can be tried anytime; R2 requires the most effort but has highest potential.

### Phase 3: Aggregator Ablation → Paper §4.4 Table 3 (covered by Phase 1) **[P0]**

**Goal**: Compare 3 aggregation strategies (no learning / lightweight learning / full learning).

| Aggregator | Params | Design | Source |
|-----------|--------|--------|--------|
| Naive avg | 0 | $(1/K)\sum c_k$ | Built-in (existing Exp 7) |
| Simple cosine | 8K | cosine sim → softmax | Phase 1b |
| **Attention** | **198K** | multi-head cross-attn + scale | Phase 1a |

> These three cover the 0→8K→198K parameter gradient, sufficient for a course report.

**Aggregation directions to explore** (try after P0, ordered by priority):

| # | Direction | Idea | Extra params | Difficulty | GPU cost |
|---|-----------|------|-------------|------------|----------|
| A1 | **MLP aggregator** | `concat(h_q, h_k)→Linear(2D,D)→SiLU→Linear(D,1)→softmax over K` | ~70K | Low (1 new class) | ~3h (train+test) |
| A2 | **Gated aggregator** | `gate=σ(W(h_q⊙h_k))`; output=`α·learned + (1-α)·naive`, α learnable init=0.5 | ~130K | Low | ~3h |
| A3 | **Per-horizon weights** | Learn separate aggregation weights per prediction step t; short-term and long-term use different weights | +12 (T' scalars) | Low (add on existing aggregator) | ~1h (retrain) |
| A4 | **Graph-smoothed weights** | One-step GCN on attention weights: $\alpha'=\text{GCN}(\alpha, A_{adj})$, smoothing weights across neighboring nodes | ~5K | Medium | ~2h |
| A5 | **Temperature scheduling** | Linearly anneal softmax temperature from high→low during training, preventing early collapse to uniform weights | 0 | Low (modify training loop) | ~3h (retrain) |

> **Recommended order**: A3 → A5 → A1 → A2 → A4. A3 and A5 have near-zero implementation cost and can be applied directly to the existing attention aggregator; A1/A2 need new classes but minimal code; A4 requires passing the adj matrix into the aggregator.

### Phase 4: Multi-Dataset Validation → Paper §4.2 Table 1 extended (~2h GPU) **[P0]**

**Goal**: Validate generalization on other zero-shot datasets. A single dataset is insufficient to support the "pluggable" claim.

| Dataset | Nodes | Interval | Notes |
|---------|-------|----------|-------|
| **CHI_TAXI** | 77 | 30min | Small-scale, ~30min to run best config |
| PEMS_BAY | 325 | 5min | Medium-scale, ~2h |

> Prioritize CHI_TAXI (fewer nodes, fast). If results are positive, that is sufficient. PEMS_BAY as backup.

| ID | Experiment | Paper target |
|----|-----------|-------------|
| 4a | CHI_TAXI: zero-shot baseline | Table 1 extended |
| 4b | CHI_TAXI: best config (best retrieval + attention agg) | Table 1 extended |

### Phase 5: K Sweep → Paper §4.5 Figure 3 (~4h GPU) **[P1]**

| ID | Experiment | Paper target |
|----|-----------|-------------|
| 5a | K ∈ {1, 3, 5}, best aggregator + best retrieval | Figure 3 |

> K=3 data already available (Phase 1a), only need K=1 and K=5 — two inference runs (~2h each). K=7 requires 8 forward passes/batch, low cost-effectiveness, skipped for now.

### Phase 6: Analysis → Paper §4.6 Figures 4-5 (~1h GPU) **[P2]**

| ID | Analysis | Paper target |
|----|---------|-------------|
| 6a | Attention weight heatmap | Figure 4, §4.6.1 |
| 6b | Per-hour MAE curves (zero-shot vs ours) | Figure 5, §4.6.2 |
| 6c | Per-node MAE improvement (top/bottom nodes) | §4.6.3 |

---

## 3. Paper Outline

| Section | Pages | Content |
|---------|-------|---------|
| **Abstract** | 0.25 | Problem + method + key result (e.g., "reduces MAE by X% over zero-shot with only 198K trainable params") |
| **1. Introduction** | 1.0 | Gap in traffic FMs; ICL motivation; contribution bullets |
| **2. Related Work** | 1.0 | 2.1 Traffic foundation models; 2.2 In-context learning; 2.3 Test-time adaptation |
| **3. Method** | 2.5 | 3.1 Problem formulation; 3.2 Residual correction; 3.3 Demo retrieval strategies; 3.4 Learned aggregation architectures; 3.5 Training procedure |
| **4. Experiments** | 3.0 | 4.1 Setup; 4.2 Main results; 4.3 Retrieval ablation; 4.4 Aggregator ablation; 4.5 K sweep; 4.6 Analysis |
| **5. Discussion** | 0.5 | Limitations; compute cost; relation to true ICL |
| **6. Conclusion** | 0.25 | Summary + future work |
| **References** | 0.5 | ~20–30 citations |

---

## 4. Execution Priority & Compute Budget

Three tiers — each tier produces a usable paper version:

### Tier P0: Minimum Viable Paper (~13h GPU)

Fills Table 1 (main results + multi-dataset) + Table 2 (retrieval) + Table 3 (3 aggregators).

| Day | Tasks | GPU hrs |
|-----|-------|---------|
| 1 | 1a: Rerun Exp 8 test; 1b: SimpleDemoAggregator train+test | 5h |
| 2 | 2b: Embedding KNN implement + test; 2c: Enc-KNN + attention | 6h |
| 3 | 4a-4b: CHI_TAXI zero-shot + best config | 2h |

### Tier P1: K Sweep (+4h GPU)

Adds Figure 3 (K sensitivity), increases ablation completeness.

| Day | Tasks | GPU hrs |
|-----|-------|---------|
| 4 | 5a: K=1, K=5 inference (K=3 already done) | 4h |

### Tier P2: Analysis Depth (+1h GPU)

Figures 4-5, qualitative analysis. Mainly post-processing.

| Day | Tasks | GPU hrs |
|-----|-------|---------|
| 5 | 6a-6c: Attention viz, per-hour/per-node analysis | 1h |

**Total: P0=13h, P0+P1=17h, Full=18h** (down from 46h in original plan — 60% savings)

---

## 5. Implementation Checklist

### P0 — Minimum Viable Paper
- [x] 1a: Rerun Exp 8 test (10ep) → MAE=4.29, RMSE=7.71, MAPE=11.42%, CORR=0.7483
- [ ] 1a': Retrain attention aggregator (3 epochs) for fair comparison → Table 1, Table 3
- [ ] 1b: Train + test SimpleDemoAggregator → Table 3
- [ ] 2: Implement `demo_selection='embedding'` in `ict_data_process.py`
- [ ] 2: Precompute encoder features for demo pool
- [ ] 2b: Test enc-KNN (node-mean) + naive avg → Table 2
- [ ] 2c: Test best enc-KNN + attention aggregator → Table 1 "Ours"
- [ ] 4a: CHI_TAXI zero-shot baseline → Table 1
- [ ] 4b: CHI_TAXI best config → Table 1
- [ ] Add inference time measurement (wall-clock per sample) → Table 1 "Time" column
- [ ] Update `05_experiment_log.md` with all new results

### P1 — K Sweep
- [ ] 5a: K=1 inference (best retrieval + best aggregator) → Figure 3
- [ ] 5a: K=5 inference (best retrieval + best aggregator) → Figure 3

### P2 — Analysis
- [ ] 6a: Attention weight extraction + heatmap → Figure 4
- [ ] 6b: Per-hour MAE curves → Figure 5
- [ ] 6c: Per-node MAE top/bottom analysis → §4.6.3

### Paper Writing
- [ ] Draft §3 Method (can start after P0)
- [ ] Draft §4 Experiments (after P0+P1)
- [ ] Draft §1-2 Intro + Related Work
- [ ] Draft §5-6 Discussion + Conclusion
- [ ] Figures and formatting

---

## 6. Future Research Directions

> Directions beyond this course report scope, recorded for longer-term exploration along the Retrieval and Aggregation axes.

### 6.1 Retrieval

#### 6.1.1 Learned Retrieval Metric
Current embedding KNN uses frozen encoder features + L2 distance. The encoder was not designed for retrieval, so the distance metric may be suboptimal.

- **Option A: Metric adapter** — add a `Linear(D→D')` projection head on frozen encoder output, train with contrastive loss (positives = demo pairs with effective corrections, negatives = ineffective)
- **Option B: Siamese fine-tune** — LoRA fine-tune the last 1-2 encoder layers so similar query-demo pairs are closer in embedding space
- **Evaluation**: Compare retrieval precision@K (ranking of retrieved demos by correction quality)

#### 6.1.2 Cross-Node Retrieval
Current retrieval operates along the temporal axis (finding similar time windows); each demo covers all nodes.

- **Idea**: Allow composing demo corrections from different time windows for different nodes
- **Motivation**: For heterogeneous graphs (nodes with very different patterns), some nodes may find better matches in other time windows
- **Risk**: Increases retrieval complexity, may break spatial consistency

### 6.2 Aggregation

#### 6.2.1 Iterative Correction Refinement
Current correction is one-shot.

- **Idea**: Multi-round correction — round 1: apply correction → round 2: re-retrieve/re-aggregate based on corrected prediction
- **Motivation**: First-round correction may overshoot or drift; second round can fine-adjust
- **Risk**: Requires multiple forward passes, overhead may be excessive
