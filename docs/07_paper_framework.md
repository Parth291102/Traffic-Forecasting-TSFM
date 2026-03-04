# Paper Draft Framework: Lightweight In-Context Learning for Pretrained Spatio-Temporal Foundation Models

> Working document — paper skeleton with section-level content guidance and experiment-to-section mapping.

---

## Abstract (~200 words)

**Structure**: Problem → Gap → Method → Key Results → Significance

> Pretrained spatio-temporal foundation models (e.g., OpenCity, UniST) achieve strong zero-shot traffic forecasting, but lack the ability to adapt predictions using task-specific demonstrations at test time — a capability known as in-context learning (ICL). Naively extending input sequences to include demonstrations violates the model's fixed-length pretraining assumption, causing severe performance degradation. We propose **ICL-Traffic**, a lightweight, pluggable module that enables in-context adaptation for *any* frozen traffic foundation model through two components: (1) a **residual correction framework** that estimates model bias from demonstration prediction errors, and (2) a **learned aggregation module** (~200K trainable parameters) that attends over demonstration-query feature similarity to produce context-aware corrections. We further improve demonstration quality via **embedding-space retrieval**, replacing raw-input KNN with encoder-feature similarity matching. On PEMS07M, our method reduces MAE from [ZS baseline] to [best result], achieving [X%] improvement with only 0.2% additional parameters relative to the base model. The module requires no gradient flow through the backbone, trains in [Y] minutes from cached features, and transfers across datasets without re-architecture.

**Experiments needed**: Final best MAE number, training time, parameter ratio.

---

## 1. Introduction (1 page)

### Para 1: Traffic Foundation Models — Success and Limitation
- Pretrained ST models (OpenCity, UniST, STEP, TrafficBERT) achieve strong zero/few-shot forecasting
- But: predictions are *static* — same model, same behavior, regardless of test context
- In NLP/vision, foundation models improve at test time via in-context learning (ICL)
- **Gap**: No existing work enables ICL for frozen traffic foundation models

### Para 2: Challenge — Why ICL is Hard for Traffic Models
- NLP ICL: prepend demonstrations to input sequence, model attends across demo+query jointly
- Traffic models: pretrained on *fixed-length* sequences (e.g., 288 timesteps = 24 patches)
- Extending sequence → OOD attention patterns, positional encoding mismatch, precision issues
- Our Exp 2-4: naive concatenation degrades MAE by 9-28% (cite our ablations)

### Para 3: Our Approach — Residual Correction + Learned Aggregation
- Key insight: process query and demos *independently* through the frozen model, combine in *output space*
- Residual correction: each demo's prediction error estimates the model's systematic bias for similar inputs
- Two innovations: (a) embedding-space retrieval for higher-quality demos; (b) learned cross-attention aggregation to replace naive averaging
- Only ~200K new parameters, no backbone modification, trains from cached features in minutes

### Para 4: Contributions (3 bullet points)
1. A general residual correction framework for enabling ICL in frozen ST foundation models
2. A lightweight learned aggregation module with embedding-space demo retrieval
3. Systematic ablation study on retrieval strategies, aggregation architectures, and demonstration count K

### Figure 1: Method Overview Diagram
Left: standard zero-shot inference. Right: ICL-Traffic pipeline (query → frozen model → prediction; demos → frozen model → corrections; aggregator combines).

**Experiments needed**: None new for introduction (cite existing Exp 2-4 for "why naive doesn't work").

---

## 2. Related Work (1 page)

### 2.1 Spatio-Temporal Foundation Models
- OpenCity [ref], UniST [ref], STEP [ref], UrbanGPT [ref]
- Pretraining on heterogeneous traffic data, zero-shot transfer
- Limitation: no test-time adaptation beyond the prediction head

### 2.2 In-Context Learning
- Origin in NLP: GPT-3 [Brown et al.], prompt engineering
- Recent extension to vision: visual prompting [ref], in-context segmentation [ref]
- In time series: recent work on ICL for time-series forecasting [ref — check if any exist]
- **Gap**: no ICL framework for pretrained ST models with graph structure

### 2.3 Test-Time Adaptation
- TTT [Sun et al.], TENT [Wang et al.], TTA for time series [ref]
- Difference from our work: TTA modifies model parameters; we freeze the backbone entirely
- Our method is complementary — the aggregation module could combine with TTA

**Experiments needed**: None (literature review).

---

## 3. Method (2.5 pages)

### 3.1 Problem Formulation (0.3 page)

- Traffic forecasting: given historical observations $\mathbf{X} \in \mathbb{R}^{T \times N \times F}$ on a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$, predict future values $\mathbf{Y} \in \mathbb{R}^{T' \times N \times 1}$
- Pretrained model $f_\theta$: frozen parameters $\theta$, produces $\hat{\mathbf{Y}} = f_\theta(\mathbf{X})$
- Demonstration pool $\mathcal{D} = \{(\mathbf{X}_d, \mathbf{Y}_d)\}_{d=1}^{|\mathcal{D}|}$: training samples with known ground truth
- **Goal**: at test time, select K demonstrations and produce $\hat{\mathbf{Y}}_q$ that improves over $f_\theta(\mathbf{X}_q)$

### 3.2 Residual Correction Framework (0.5 page)

**Core formulation**:
$$\hat{\mathbf{Y}}_q = f_\theta(\mathbf{X}_q) + \text{Aggregate}\left(\left\{\mathbf{Y}_k - f_\theta(\mathbf{X}_k)\right\}_{k=1}^K\right)$$

**Intuition**: If the model makes a predictable error on training samples similar to the query, we can estimate and correct that error at test time.

**Properties**:
- Zero OOD risk: each input processed through the standard forward path
- When $K=0$, degenerates to zero-shot prediction
- When corrections are random noise, $\mathbb{E}[\text{Aggregate}] \approx 0$, recovering zero-shot (verified in Exp 5)

**Discussion**: Analogy to bias correction in ensemble methods, connection to kNN regression in output space.

### 3.3 Demonstration Retrieval (0.7 page)

**Problem**: The quality of corrections depends critically on demo-query similarity.

#### 3.3.1 Baseline: Raw-Input KNN
$$\mathcal{N}_K(\mathbf{X}_q) = \arg\min_{d \in \mathcal{D}} \|\text{vec}(\mathbf{X}_q) - \text{vec}(\mathbf{X}_d)\|_2$$

- Operates on raw input vectors: $\text{vec}(\mathbf{X}) \in \mathbb{R}^{T \times N \times F}$
- Simple and effective (MAE: 7.39 → 4.66 with K=3, vs random)
- Limitation: L2 in high-dimensional raw space may not capture semantic similarity

#### 3.3.2 Embedding-Space KNN (Proposed)
$$\mathcal{N}_K(\mathbf{X}_q) = \arg\min_{d \in \mathcal{D}} \|\mathbf{h}_q - \mathbf{h}_d\|_2, \quad \mathbf{h} = \text{pool}(\text{Encoder}(\mathbf{X}))$$

- Uses frozen encoder features as retrieval representation
- Captures learned traffic patterns rather than raw signal similarity
- One-time precompute: run encoder over demo pool → cache features

**Experiments needed**: Exp 2b, 2c (embedding KNN variants) → fills Table 3 in paper.

### 3.4 Learned Aggregation (1.0 page)

Replace naive $(1/K)\sum_k \mathbf{c}_k$ with a learned module:

$$\hat{\mathbf{C}} = \text{Aggregator}(\mathbf{h}_q, \{\mathbf{h}_k\}_{k=1}^K, \{\mathbf{c}_k\}_{k=1}^K)$$

where $\mathbf{h}$ are encoder features and $\mathbf{c}_k = \mathbf{Y}_k - f_\theta(\mathbf{X}_k)$.

#### Architecture Variants (present all, highlight best):

**(a) Cosine-Similarity Aggregator** (SimpleDemoAggregator, ~8K params)
- Project query/demo features → cosine similarity → temperature-scaled softmax → weighted sum
- Minimal parameters, interpretable weights

**(b) Cross-Attention Aggregator** (DemoAggregator, ~198K params) ← primary
- Multi-head cross-attention: Q = query features, K = demo features
- Per-head weighted corrections combined via learned head-combine layer
- Adaptive output scaling via softplus gate (zero-initialized for stable training)

**(c) MLP Aggregator** (~70K params)
- Concatenate query+demo features → 2-layer MLP → per-demo scalar weight → softmax
- Tests whether attention mechanism is necessary or simple feature interaction suffices

**(d) Gated Aggregator** (~130K params)
- Element-wise gating: $\sigma(\mathbf{W}(\mathbf{h}_q \odot \mathbf{h}_k))$ as per-demo gate
- Residual connection to naive average: $\alpha \cdot \text{learned} + (1-\alpha) \cdot \text{naive}$
- Tests the value of smooth interpolation between learned and naive

**Figure 2**: Architecture diagram showing the 4 aggregator variants side by side.

**Experiments needed**: Exp 3a-3e (aggregator comparison) → fills Table 2 in paper.

### 3.5 Training Procedure (0.3 page)

- Freeze all base model parameters
- Precompute and cache: query predictions, encoder features, demo corrections (fp16 on CPU)
- Train aggregator only: Adam optimizer, lr=1e-4, MAE loss, early stopping (patience 5)
- Training cost: ~minutes on cached data (no forward pass through base model)

---

## 4. Experiments (3 pages)

### 4.1 Experimental Setup (0.5 page)

**Dataset**: PEMS07M — 228 sensor nodes, 5-min interval, highway traffic flow
- Split: 50% train / 10% val / 40% test
- Demo pool: 5,761 sliding windows from training set

**Base Model**: OpenCity-plus (embed_dim=512, enc_depth=6, ~100M params)
- Pretrained on multi-city traffic data; PEMS07M is zero-shot (unseen during pretraining)

**Metrics**: MAE, RMSE, MAPE (%), Pearson Correlation
- Standard masked evaluation: exclude zero-flow points (mae_thresh=0, mape_thresh=0.001)

**Implementation**: PyTorch 2.4.1, NVIDIA L40S (46GB), fp32 inference

### 4.2 Main Results (0.5 page) → **Table 1**

| Method | Demo | K | MAE↓ | RMSE↓ | MAPE%↓ | CORR↑ | Δ vs ZS |
|--------|------|---|------|-------|--------|-------|---------|
| Zero-Shot | — | 0 | 4.50 | 8.21 | 12.20 | 0.736 | — |
| Residual + Random | rand | 3 | 6.14 | 9.53 | 14.84 | 0.634 | -36% |
| Residual + KNN | raw | 3 | 4.66 | 8.11 | 11.92 | 0.729 | -3.6% |
| Residual + Enc-KNN | emb | 3 | TBD | | | | |
| **Ours (3ep)** | raw | 3 | **TBD** | | | | |
| Ours (10ep) | raw | 3 | 4.29 | 7.71 | 11.42 | 0.748 | +4.7% |

**Narrative**: (1) Random demo selection hurts — confirms that demo *quality* matters. (2) KNN retrieval nearly recovers zero-shot. (3) Embedding retrieval + learned aggregation outperforms zero-shot, achieving ICL benefit.

**Experiments needed**: Exp 2b/2c + best aggregator with best retrieval.

### 4.3 Ablation: Demo Retrieval Strategy (0.5 page) → **Table 2**

| Retrieval | Feature Space | Dim | MAE | RMSE |
|-----------|--------------|-----|-----|------|
| Random | — | — | 6.14 | 9.53 |
| KNN (raw) | raw input | 196K | 4.66 | 8.11 |
| KNN (enc, mean) | encoder | 512 | TBD | TBD |
| KNN (enc, full) | encoder | 117K | TBD | TBD |

All with naive averaging, K=3. Shows that semantic similarity in embedding space improves over raw L2.

**Experiments needed**: Exp 2b, 2c.

### 4.4 Ablation: Aggregation Architecture (0.5 page) → **Table 3**

| Aggregator | Trainable Params | MAE | RMSE | MAPE% |
|-----------|-----------------|-----|------|-------|
| Naive average | 0 | TBD | TBD | TBD |
| Cosine similarity | 8K | TBD | TBD | TBD |
| MLP | 70K | TBD | TBD | TBD |
| Gated | 130K | TBD | TBD | TBD |
| **Cross-attention (3ep)** | **198K** | **TBD** | | |
| Cross-attention (10ep) | 198K | 4.29 | 7.71 | 11.42 |

All with best retrieval, K=3. Shows that learned aggregation meaningfully improves over naive averaging, and cross-attention captures richer query-demo interactions than simpler alternatives.

> **Note**: OpenCity few-shot trains prediction heads for 3 epochs. The 3-epoch row is the fair comparison; the 10-epoch row shows aggregator capacity with extended training.

### 4.5 Effect of Demonstration Count K (0.3 page) → **Figure 3**

Line plot: MAE vs K for {1, 3, 5, 7} under (a) naive avg and (b) best learned aggregator.

**Expected story**: (a) Naive avg improves monotonically with K but saturates. (b) Learned aggregator benefits more from each additional demo. (c) There exists diminishing returns beyond K=5-7.

**Experiments needed**: Exp 4a, 4b.

### 4.6 Analysis (0.7 page)

#### 4.6.1 Attention Weight Analysis → **Figure 4**

Visualize the learned cross-attention weights $\alpha_{k,n}$ across K demos and N nodes:
- (a) Heatmap for a representative query showing non-uniform demo weighting
- (b) Correlation between attention weight and demo-query feature cosine similarity
- **Story**: The aggregator learns to upweight demos that are more semantically similar to the query at a per-node level.

#### 4.6.2 Temporal Analysis → **Figure 5**

Per-hour-of-day MAE comparison: zero-shot vs ICL-Traffic.
- **Expected story**: ICL provides larger gains during transition periods (rush hour onset/offset) where model bias is highest.

#### 4.6.3 Spatial Analysis

Per-node MAE improvement map (if spatial coordinates available) or sorted bar chart.
- **Story**: Nodes with high zero-shot error benefit most from ICL correction.

#### 4.6.4 Correction Magnitude

Histogram of $\|\hat{\mathbf{C}}\|$ for naive avg vs learned aggregator.
- **Story**: Learned aggregator produces more moderate, targeted corrections (lower variance).

**Experiments needed**: Exp 5a-5e (analysis scripts).

---

## 5. Discussion (0.5 page)

### Limitations
1. **Compute overhead**: K+1 forward passes per sample (e.g., 4× for K=3). Partially mitigated by chunked batching.
2. **Not true ICL**: Demos do not enter the model's attention context — corrections are in output space, not representation space. This is a pragmatic constraint from fixed-length pretraining.
3. **Retrieval quality ceiling**: Embedding-space KNN improves over raw KNN, but the frozen encoder was not designed for retrieval. A learned retrieval metric could further improve.

### Connection to Related Paradigms
- **Test-time training (TTT)**: updates model params; ours freezes backbone, complementary
- **kNN regression**: our method can be viewed as kNN in correction space rather than prediction space
- **Prompt tuning**: soft prompts modify input representation; our corrections modify output. Future work: bridge these.

### Future Work
- Cross-example attention with continued pretraining (Plan 1 — needs retraining)
- Learned retrieval metric (jointly train retrieval + aggregation)
- Cross-domain demos (use corrections from different traffic domains)

---

## 6. Conclusion (0.25 page)

> We presented ICL-Traffic, a lightweight in-context learning module for pretrained traffic foundation models. By framing ICL as residual correction with learned aggregation, our approach enables test-time adaptation using historical demonstrations without modifying the frozen backbone. The module adds only ~200K parameters (<0.2% of the base model), trains in minutes from cached features, and consistently improves over zero-shot inference. Our systematic ablations reveal that demo retrieval quality and aggregation design are both critical, with embedding-space retrieval and cross-attention aggregation achieving the best results. ICL-Traffic is model-agnostic and can plug into any ST foundation model, opening a practical path toward adaptive traffic forecasting.

---

## Figures and Tables Summary

| Item | Content | Paper Section | Experiment Source |
|------|---------|--------------|-------------------|
| **Figure 1** | Method overview diagram | §3.2 | — (diagram) |
| **Figure 2** | Aggregator architecture variants | §3.4 | — (diagram) |
| **Figure 3** | MAE vs K (line plot) | §4.5 | Exp 4a, 4b |
| **Figure 4** | Attention weight heatmap | §4.6.1 | Exp 5a |
| **Figure 5** | Per-hour MAE comparison | §4.6.2 | Exp 5c |
| **Table 1** | Main results | §4.2 | Exp 1,5,7,8 + new |
| **Table 2** | Retrieval ablation | §4.3 | Exp 2b, 2c |
| **Table 3** | Aggregator ablation | §4.4 | Exp 3a-3e |

---

## Experiment → Paper Section Mapping

This mapping ensures every experiment directly serves a paper section:

| Exp ID | Description | Paper Section | Priority |
|--------|-----------|--------------|----------|
| 1a | Rerun Exp 8 (attention agg, exact metrics) | Table 1, Table 3 | **P0** |
| 1b | SimpleDemoAggregator (8K) | Table 3 | **P0** |
| 2b | Enc-KNN (node-mean) + naive avg | Table 2 | **P0** |
| 2c | Enc-KNN (full) + naive avg | Table 2 | **P1** |
| 2d | Best Enc-KNN + attention agg | Table 1 (Ours row) | **P0** |
| 3d | MLP aggregator | Table 3 | **P1** |
| 3e | Gated aggregator | Table 3 | **P1** |
| 4a | K sweep, naive avg | Figure 3 | **P1** |
| 4b | K sweep, best aggregator | Figure 3 | **P1** |
| 5a | Attention weight extraction | Figure 4 | **P2** |
| 5b | Per-node MAE | §4.6.3 | **P2** |
| 5c | Per-hour MAE | Figure 5 | **P2** |
| 5d | Correction magnitude stats | §4.6.4 | **P2** |
| 5e | Error case study | §4.6 (qualitative) | **P3** |

**P0** = must have for minimum viable paper (fills main tables)
**P1** = needed for complete ablation story
**P2** = analysis depth (differentiates from tech report)
**P3** = nice to have

### Minimum Viable Paper (P0 only, ~20h GPU)

With only P0 experiments, the paper has:
- Table 1: 5 rows (ZS, random, raw-KNN, enc-KNN, ours)
- Table 2: 4 rows (random, raw-KNN, enc-KNN mean, enc-KNN full)
- Table 3: 3 rows (naive, simple, attention)
- This is sufficient for a solid 8-page course report.

### Full Paper (P0+P1+P2, ~45h GPU)

Adds:
- Table 3: +2 rows (MLP, gated)
- Figure 3: K sweep plot
- Figures 4-5: attention viz, temporal analysis
- This makes a strong conference-quality submission.
