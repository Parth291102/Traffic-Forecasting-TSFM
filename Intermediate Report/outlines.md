# Detailed Outlines for Intermediate Report

> 请逐节阅读，标注你的修改意见，确认后我将展开撰写 LaTeX 正文。
>
> **核心文献（叙事主线围绕展开）**：
> 1. **OpenCity** [li2024opencity] — 我们的 base foundation model，分析其优势与局限
> 2. **ICT** [chen2022ict] — 我们的 idea 来源，meta-learning via in-context tuning
> 3. **TimesFM-ICF** [das2024icf] — ICL 在 time series 上的应用参考
>
> **其他文献**：仅简要提及作为佐证，不展开讨论。

---

## Abstract Outline (~200 words)

- **Problem + Gap**: Pretrained spatio-temporal foundation models (e.g., OpenCity [1]) achieve strong zero-shot traffic forecasting but still suffer performance degradation under cross-city distribution shifts. In NLP and time-series domains, in-context learning (ICL) has emerged as a powerful adaptation mechanism — enabling models to condition predictions on retrieved demonstrations at inference time. ICL has been formalized as a meta-learning objective via in-context tuning [2] and extended to time-series foundation models via in-context fine-tuning [3]. However, ICL remains entirely unexplored for spatio-temporal foundation models.
- **Challenge**: The standard ICL mechanism — feeding demonstrations into the model's context window for joint attention over demos and query — cannot be directly applied to spatio-temporal foundation models. ST foundation models are typically pretrained on fixed-length input sequences; OpenCity, for instance, uses 24 patches (288 timesteps). Concatenating K demonstrations extends the sequence beyond this fixed length, producing out-of-distribution self-attention patterns, positional encoding mismatches, and TC cross-attention failures. Our preliminary experiments confirm this: naive sequence extension degrades MAE by 9–28% versus zero-shot. Enabling native ICL would require architectural modification and continued pretraining — substantial cost that risks eroding the pretrained representations.
- **Method**: We propose a lightweight, pluggable ICL module for frozen ST foundation models. A small aggregation network (8K-198K params) is trained offline from cached features to learn a general correction weighting strategy. At inference time, for each test query, KNN retrieves the K most similar training examples; the frozen model's prediction errors on these demonstrations are aggregated into a query-specific correction — adapting each prediction to its local traffic context without any test-time weight updates.
- **Key Result**: On PEMS07M with OpenCity-plus, our best configuration reduces MAE from 4.50 to 4.29 (+4.7%) with only 198K trainable parameters (~0.8% of the 26M-param base model), approaching full-shot supervised baselines while the backbone remains entirely frozen.

---

## Section I: Introduction

### Para 1 — ST Foundation Models and the Distribution Shift Problem

- Traffic forecasting is critical for urban management. Traditional ST models (STGCN, GWN, etc.) achieve strong supervised performance but require per-dataset training. [1-2 sentences, cite briefly]
- **OpenCity** [li2024opencity] represents a new paradigm: a spatio-temporal foundation model pretrained on large-scale heterogeneous traffic data, achieving zero-shot transfer across cities, data categories, and time horizons. It integrates Transformer architecture with GNNs and demonstrates promising scaling laws.
- **However**, despite strong zero-shot capability, performance degrades under larger cross-city distribution shifts. For example, on SZ-DIDI and CD-DIDI, OpenCity's zero-shot MAE falls noticeably behind full-shot supervised baselines, exposing the limits of purely static zero-shot inference.
- **Current adaptation**: OpenCity provides gradient-based fine-tuning of the prediction head (3 epochs on target-domain data). This improves performance, but the question remains: are there alternative adaptation paradigms — specifically, demonstration-based adaptation at inference time — that could complement or rival fine-tuning for ST foundation models?

### Para 2 — ICL: A Proven Adaptation Paradigm, Missing in ST FMs

- In NLP, in-context learning (ICL) allows foundation models to adapt at test time by conditioning on demonstrations [brown2020gpt3, brief mention].
- **ICT** [chen2022ict] (our idea source): Formalizes ICL as a meta-learning objective. By fine-tuning LMs to predict target labels given concatenated instruction + in-context examples + query, ICT trains models to *learn from demonstrations* as a general capability. Key contributions:
  - Bridges prompting, fine-tuning, and meta-learning into a unified framework
  - Outperforms gradient-based meta-learning (MAML) by leveraging LM inductive bias for pattern matching
  - Reduces sensitivity to example ordering (6x) and selection (2x)
  - Core insight for our work: **models can be trained to extract useful patterns from contextual examples**
- **TimesFM-ICF** [das2024icf] (our application reference): Extends ICL to time-series foundation models. The model is trained to use related time-series in its context window (via separator tokens and cross-example attention) to forecast a target series. Key contributions:
  - Demonstrates ICL works beyond language — in numerical time-series forecasting
  - Achieves 7-25% improvement over the base FM, even rivaling per-dataset fine-tuning
  - Core insight for our work: **providing related examples at inference time helps the model adapt to target distributions**
- **Research gap**: ICL has been established in NLP (ICT) and time-series (TimesFM-ICF), but remains entirely unexplored for spatio-temporal foundation models. Can we bring ICL capability to frozen ST FMs like OpenCity?

### Para 3 — Challenges and Our Approach

- **Challenge — how to bring ICL to ST FMs**: Both ICT and TimesFM-ICF enable ICL by modifying the model (fine-tuning or continued pretraining with architectural changes). For spatio-temporal foundation models, such modification is non-trivial: ST FMs are typically pretrained on fixed-length input sequences with specific attention patterns and positional encodings — OpenCity, for instance, uses 24 patches (288 timesteps). Naively extending input sequences to include demonstrations causes OOD self-attention patterns, positional encoding mismatches, and TC cross-attention failures — our preliminary experiments show 9-28% MAE degradation versus zero-shot. This motivates an approach that never extends the model's input sequence: process query and demonstrations independently through the standard forward path, and combine their information only in output space.
- **Our insight**: Instead of modifying the model to accept demonstrations in its context window, we process query and demonstrations *independently* through the frozen model and combine their information in *output space* via residual correction. Each demo's prediction error estimates the model's systematic bias on similar inputs; a learned aggregator combines these errors into a query-specific correction at inference time.
- **Two axes of improvement**: (1) KNN retrieval ensures each query receives the most relevant demonstrations from a training-set demo pool; (2) a learned aggregation network (8K-198K params) replaces naive averaging with query-aware weighting — the aggregator learns a general correction strategy, while demo retrieval provides the per-query variation.
- The base model remains completely frozen; the aggregator is trained offline in seconds from cached features.

### Para 4 — Contributions

1. We are the first to bring in-context learning capability to spatio-temporal foundation models. Inspired by ICT [chen2022ict] and TimesFM-ICF [das2024icf], we propose a residual correction framework that enables demonstration-based adaptation for frozen ST FMs, without architectural modification or base model retraining.
2. We systematically study two axes of the framework: demonstration retrieval (random vs. KNN) and correction aggregation (naive average vs. cosine similarity vs. cross-attention), showing both are critical and jointly necessary to surpass zero-shot performance.
3. On PEMS07M with OpenCity-plus, our best configuration achieves +4.7% MAE improvement over zero-shot (4.50→4.29) with only 198K external parameters (~0.8% of the base model). We further provide a direct comparison against OpenCity's 3-epoch fast adaptation on SZ-DIDI and CD-DIDI under the same training budget (planned).

---

## Section II: Related Work

### 2.1 Spatio-Temporal Foundation Models

- **Brief background**: Traditional ST models (STGCN [yu2018stgcn], GWN [wu2019graphwavenet], etc.) require per-dataset training. Recent ST foundation models address this through large-scale pretraining. [1-2 sentences]
- **OpenCity** [li2024opencity] (detailed): Our base model. Integrates Transformer with GNNs; uses instance normalization for distribution shift handling and patch embedding for long-term prediction. Pretrained on heterogeneous multi-city traffic data covering flow, speed, and demand categories. Achieves zero-shot performance competitive with full-shot baselines across 6 datasets, demonstrates fast adaptation via 3-epoch prediction head fine-tuning, and exhibits promising scaling laws. However:
  - Zero-shot performance degrades under larger distribution shifts (CD-DIDI, SZ-DIDI)
  - Adaptation is limited to gradient-based fine-tuning
  - Inference-time adaptation (ICL) is unexplored
- **Other ST FMs** [brief, 1-2 sentences each]: UniST [yuan2024unist] (prompt-empowered), UrbanGPT [li2024urbangpt] (LLM backbone), FlashST [li2024flashst] (prompt-tuning). All adapt via fine-tuning or prompt-tuning; none explores ICL.

### 2.2 In-Context Learning

- **Brief NLP background**: GPT-3 [brown2020gpt3] demonstrates ICL via few-shot prompting. [1 sentence]
- **ICT** [chen2022ict] (detailed, our idea source):
  - Recasts few-shot learning as sequence prediction: concatenate instruction + labeled examples + target query → predict target label
  - Meta-trains LMs to learn from in-context examples across diverse tasks
  - At test time, adapts to new tasks via demonstrations without gradient updates
  - Outperforms MAML (gradient-based meta-learning) by leveraging LM pattern-matching inductive bias
  - Significantly reduces sensitivity to example ordering and selection
  - **Connection to our work**: We adopt the core idea — learning to extract useful patterns from contextual examples — but apply it to ST foundation models. Unlike ICT, our base model was not trained for ICL, so we achieve this through external output-space correction rather than internal attention-based ICL.
- **TimesFM-ICF** [das2024icf] (detailed, our application reference):
  - First work to enable ICL for time-series foundation models
  - Adapts TimesFM architecture: adds separator tokens between in-context examples, enables cross-example causal attention, removes positional encoding for length generalization
  - Continued pretraining from base checkpoint teaches the model to borrow patterns from in-context examples
  - Achieves 7-25% improvement over base FM; even surpasses per-dataset fine-tuning on Monash benchmark
  - **Connection to our work**: We share the goal of leveraging related examples for inference-time adaptation. However, TimesFM-ICF requires architectural modification and continued pretraining. We pursue a complementary approach: keep the ST model completely frozen and achieve adaptation through output-space residual correction.

### 2.3 Test-Time Adaptation

- **Brief background**: TTA methods (TTT [sun2020ttt], TENT [wang2021tent]) adapt models at test time by updating parameters using self-supervised objectives or entropy minimization. Recent work extends TTA to time-series forecasting (COSA, TAFAS, etc.). [2-3 sentences]
- **Positioning across adaptation paradigms** [concise]:

  | Paradigm | Training data | Base model | Per-query variation at inference |
  |----------|--------------|------------|-------------------------------|
  | Fine-tuning (OpenCity fast-adapt) | Target-domain train set | Modified | No (same model for all queries) |
  | TTA (TTT, TENT) | Test batch statistics | Modified at test time | Per-batch |
  | **Ours** | Target-domain train set (as demo pool) | **Frozen** | **Yes (query-specific demo retrieval)** |

  > Note: Fine-tuning and our method use the same target-domain training data; the difference is utilization — absorbed into weights vs. preserved as a retrievable demo pool.

  - TTA requires a self-supervised signal or statistics at test time; our method uses pre-computed training demonstrations, requiring no test-time gradient
  - Our method is complementary to both fine-tuning and TTA — the aggregator's per-query correction could be applied on top of a fine-tuned or TTA-adapted model

---

## Section III: Method

### 3.1 Problem Formulation (0.3 page)

- Traffic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$ with $N$ nodes
- Historical observations $\mathbf{X} \in \mathbb{R}^{T \times N \times F}$, predict future $\mathbf{Y} \in \mathbb{R}^{T' \times N \times 1}$
- Frozen pretrained model $f_\theta$: $\hat{\mathbf{Y}} = f_\theta(\mathbf{X})$
- Demonstration pool $\mathcal{D} = \{(\mathbf{X}_d, \mathbf{Y}_d)\}_{d=1}^{|\mathcal{D}|}$ from training set (ground truth available)
- **Goal**: At test time, select $K$ demonstrations and produce $\hat{\mathbf{Y}}_q$ that improves over $f_\theta(\mathbf{X}_q)$, without modifying $\theta$

### 3.2 Residual Correction Framework (0.5 page)

- Core equation:
$$\hat{\mathbf{Y}}_q = f_\theta(\mathbf{X}_q) + \text{Aggregate}\left(\left\{\mathbf{Y}_k - f_\theta(\mathbf{X}_k)\right\}_{k=1}^K\right)$$
- Each correction $\mathbf{c}_k = \mathbf{Y}_k - f_\theta(\mathbf{X}_k)$ captures model's systematic error on demo $k$
- **Properties**:
  - Zero OOD risk: each input processed through the standard forward path
  - When $K=0$, degenerates to zero-shot prediction
  - When corrections are random noise, $\mathbb{E}[\text{Aggregate}] \approx 0$, recovering zero-shot (verified experimentally)
- **Connection to ICT/TimesFM-ICF**: Like ICT, we use demonstrations to adapt model behavior at inference time. Like TimesFM-ICF, we leverage related examples to condition predictions on target distributions. Unlike both, we achieve this entirely in output space without modifying the base model's architecture or weights. At inference time, the aggregator (trained offline, fixed weights) provides a general correction weighting strategy, while KNN demo retrieval provides the per-query variation — each test query receives different demonstrations and therefore a different correction, even though the aggregator weights are the same for all queries.

### 3.3 Demonstration Retrieval (0.5 page)

- **Problem**: correction quality depends critically on demo-query similarity
- **Random baseline**: sample $K$ demos uniformly → high variance, no query correlation → MAE 6.14 (K=3), worse than zero-shot
- **Raw-input KNN**:
$$\mathcal{N}_K(\mathbf{X}_q) = \arg\min_{d \in \mathcal{D}} \|\text{vec}(\mathbf{X}_q) - \text{vec}(\mathbf{X}_d)\|_2$$
  - Simple and effective: MAE 6.14 → 4.66 (K=3)
  - KNN is deterministic → sampling rounds $S$ have no effect → confirms demo-query alignment is the key factor
- **Planned (TBD)**: Embedding-space KNN using frozen encoder features

### 3.4 Learned Aggregation (0.7 page)

- Replace naive $(1/K)\sum \mathbf{c}_k$ with learned module:
$$\hat{\mathbf{C}} = \text{Aggregator}(\mathbf{h}_q, \{\mathbf{h}_k\}_{k=1}^K, \{\mathbf{c}_k\}_{k=1}^K)$$
  where $\mathbf{h}$ are encoder features from the frozen model

#### (a) SimpleDemoAggregator (~8K params)
- Project query/demo encoder features → cosine similarity → temperature-scaled softmax → weighted sum of corrections
- Minimal parameters, interpretable weights → MAE 4.47

#### (b) Cross-Attention Aggregator (~198K params)
- Multi-head cross-attention: Q = query features, K/V = demo features
- Per-head weighted corrections + learned head-combine layer
- Adaptive output scaling via softplus gate (zero-initialized for stability) → MAE 4.29-4.30

#### Training Procedure
- **Phase 1** (one-time, ~3h GPU): Freeze base model; run forward passes over all training windows to precompute and cache predictions $f_\theta(\mathbf{X}_d)$, encoder features $\mathbf{h}_d$, and corrections $\mathbf{c}_d = \mathbf{Y}_d - f_\theta(\mathbf{X}_d)$
- **Phase 2** (aggregator training, ~5 seconds): Train aggregator from cached data only — no forward pass through the base model; Adam optimizer, lr=1e-4, MAE loss, early stopping (patience 5)
- **Efficiency**: The base model is touched only once (Phase 1 caching). Phase 2 operates entirely on cached data — no forward or backward pass through the base model during aggregator training or inference.

---

## Section IV: Experiments

### 4.1 Experimental Setup (0.4 page)

- **Dataset**: PEMS07M — 228 sensor nodes, 5-min intervals, highway traffic speed; 50/10/40 split; demo pool: 5,761 windows
- **Base Model**: OpenCity-plus [li2024opencity] (embed_dim=512, enc_depth=6, 16 heads, ~26M params); PEMS07M is zero-shot (unseen in pretraining)
- **Metrics**: MAE, RMSE, MAPE (%), Pearson Correlation
- **Implementation**: PyTorch 2.4.1, NVIDIA L40S (46GB), fp32

### 4.2 Main Results — Table I (0.5 page)

**Table Ia: PEMS07M Results**

| Method | Retrieval | Params | K | MAE↓ | RMSE↓ | MAPE%↓ | CORR↑ |
|--------|-----------|--------|---|------|-------|--------|-------|
| Zero-Shot | — | 0 | 0 | 4.50 | 8.21 | 12.20 | 0.736 |
| Residual + Random | random | 0 | 3 | 6.14 | 9.53 | 14.84 | 0.634 |
| Residual + KNN (raw) | raw-L2 | 0 | 3 | 4.66 | 8.11 | 11.92 | 0.729 |
| *Residual + Enc-KNN* | *emb* | *0* | *3* | *planned* | | | |
| Simple Agg + KNN | raw-L2 | 8K | 3 | 4.47 | 8.01 | 11.55 | 0.737 |
| Cross-Attn Agg + KNN (3ep) | raw-L2 | 198K | 3 | **4.30** | **7.68** | **11.37** | **0.748** |
| Cross-Attn Agg + KNN (10ep) | raw-L2 | 198K | 3 | 4.29 | 7.71 | 11.42 | 0.748 |

Reference (full-shot supervised baselines from OpenCity paper, PEMS07M):
- GWN: MAE 4.17 | ASTGCN: MAE 4.39 | STGCN: MAE 4.44

**Table Ib: Multi-Dataset — Fair Comparison with OpenCity Fast Adaptation (3 epochs, planned)**

| Dataset | OpenCity Zero-Shot | OpenCity Fast-Adapt (3ep) | Ours: Cross-Attn (3ep) |
|---------|--------------------|--------------------------|------------------------|
| SZ-DIDI | (from paper) | (from paper) | planned |
| CD-DIDI | (from paper) | (from paper) | planned |

> Both methods use 3 epochs of training on the same target-domain training data. OpenCity fast-adapt fine-tunes the prediction head. Our method trains an external aggregator with the base model frozen. This comparison isolates the effect of the adaptation mechanism while controlling for training budget and data.

**Narrative** (4-step progression):
1. Random demos severely degrade performance (4.50→6.14, −36%) — uncorrelated corrections introduce harmful noise; demo quality is critical.
2. KNN retrieval dramatically recovers over random (6.14→4.66) but still does not surpass zero-shot (4.50) — retrieval is *necessary but not sufficient*.
3. Learned aggregation finally crosses the zero-shot threshold: Simple Agg (4.47) and Cross-Attn (4.30) both surpass the 4.50 baseline — both axes (retrieval quality + aggregation) are jointly required.
4. Best result (MAE 4.29, +4.7%) approaches full-shot supervised baselines (GWN 4.17, ASTGCN 4.39, STGCN 4.44) while the backbone remains frozen and all adaptation is per-query at inference time.

### 4.3 Ablation: Demo Selection Strategy — Table II (0.4 page)

| Selection | K | S | MAE | RMSE | vs ZS |
|-----------|---|---|-----|------|-------|
| Random | 1 | 1 | 7.39 | 12.00 | -64% |
| Random | 3 | 1 | 6.14 | 9.53 | -36% |
| Random | 1 | 10 | 5.24 | 8.52 | -16% |
| KNN | 1 | 1 | 5.61 | 10.04 | -25% |
| KNN | 3 | 1 | 4.66 | 8.11 | -3.6% |

All with naive averaging. **Takeaway**: Demo-query alignment is the dominant factor.

### 4.4 Ablation: Aggregation Architecture — Table III (0.4 page)

| Aggregator | Params | MAE | RMSE | MAPE% | CORR |
|-----------|--------|-----|------|-------|------|
| Naive average | 0 | 4.66 | 8.11 | 11.92 | 0.729 |
| Cosine similarity (5ep) | 8K | 4.47 | 8.01 | 11.55 | 0.737 |
| Cross-attention (3ep) | 198K | **4.30** | **7.68** | **11.37** | **0.748** |
| Cross-attention (10ep) | 198K | 4.29 | 7.71 | 11.42 | 0.748 |

All with KNN, K=3. **Takeaway**: Learned aggregation improves over averaging; cross-attention captures richer interactions; 3ep sufficient (fair comparison with OpenCity few-shot protocol).

### 4.5 Analysis (0.3 page)

- **Why random demos fail**: Corrections are effectively noise; averaging S→∞ rounds converges back to zero-shot (confirms random demos carry no systematic signal)
- **Why KNN is necessary but not sufficient**: Similar inputs yield correlated errors, making corrections informative; but naive uniform averaging dilutes signal with residual noise — learned weighting is needed to fully exploit these correlated corrections
- **Why learned aggregation crosses the threshold**: The aggregator learns query-aware weights that upweight the most relevant corrections and suppress outliers; even 8K params (SimpleDemoAggregator) is sufficient to surpass zero-shot
- **Convergence**: Aggregator converges by epoch 3-4 from cached features (~5 seconds total training time); 3-epoch result (MAE 4.30) nearly matches 10-epoch (MAE 4.29), confirming rapid convergence — and providing the fair-comparison epoch budget with OpenCity fast adaptation
- **Failed approach (brief)**: Naive sequence extension abandoned — OOD attention patterns and positional encoding mismatch cause 9-28% MAE degradation; motivates our output-space approach

---

## Section V: Discussion and Future Work

### Limitations
1. **Compute overhead**: K+1 forward passes per sample (4× for K=3); partially mitigated by the fact that demo forward passes can be batched and cached during Phase 1
2. **Output-space correction, not true ICL**: Demonstrations do not enter the model's attention context — corrections are applied in output space only. This is a pragmatic constraint imposed by OpenCity's fixed-length pretraining. True ICL (where the model attends over query and demonstrations jointly, as in ICT and TimesFM-ICF) would require architectural modification and continued pretraining.
3. **Our method has trainable parameters**: Unlike purely training-free approaches, our aggregation module requires an offline training phase (~5 seconds from cached features). Like fine-tuning, we use target-domain training data. The structural difference is in how that data is utilized: fine-tuning absorbs training data into model weights (producing a single globally-adapted model); our method preserves training data as a retrievable demo pool, and the aggregator learns a general strategy for weighting corrections from retrieved demonstrations. At inference time, per-query variation comes from query-specific demo retrieval, not from the aggregator weights (which are fixed for all queries).
4. **Retrieval ceiling**: Raw-input L2 KNN may miss semantic similarity in high-dimensional feature space; planned embedding-space KNN is expected to improve this.

### Planned Experiments (TBD)
- **Enc-KNN retrieval**: embedding-space KNN using frozen encoder features (node-mean pooling, 512-d); expected to improve over raw-input L2 KNN
- **K sweep**: K ∈ {1, 3, 5} for both naive avg and Cross-Attn aggregator
- **Multi-dataset validation** (SZ-DIDI, CD-DIDI): direct fair comparison against OpenCity fast adaptation (3-epoch prediction head fine-tuning) — same training budget, different adaptation mechanism
- **Analysis**: per-hour MAE curves, attention weight visualization, per-node improvement map

### Broader Connections
- **ICL progression**: ICT [chen2022ict] trains models to perform ICL via a meta-learning objective → TimesFM-ICF [das2024icf] extends ICL to time-series via continued pretraining → our work shows ICL-like demonstration-based adaptation is achievable for frozen ST models via output-space correction, without any model modification
- **Comparison with fine-tuning**: Both fine-tuning and our method use the same target-domain training data. The difference is utilization: fine-tuning absorbs all training examples into model weights (global adaptation); our method preserves them as a retrievable pool and lets each query dynamically select its most relevant subset. This query-specific demo retrieval is the source of per-query variation. We provide a direct experimental comparison with OpenCity's fast adaptation on SZ-DIDI and CD-DIDI under the same 3-epoch budget (planned).
- **Why per-query retrieval helps** (analysis point): Fine-tuning aggregates gradient signals from all training examples equally into weight updates. Our method allows training examples highly similar to a specific test query to exert disproportionate influence on that query's prediction — through KNN retrieval and learned aggregation. This may be particularly beneficial under distribution shifts where the demo pool contains local structure that a global weight update would dilute.
- **Future bridge**: The ideal endpoint is a spatio-temporal foundation model natively capable of ICL (analogous to TimesFM-ICF) — this would require continued pretraining of OpenCity with cross-example attention. Our output-space correction provides a practical path today and establishes the empirical case for investing in that direction.

---

## Section VI: Conclusion

- Proposed a lightweight, pluggable ICL module for frozen ST foundation models, inspired by ICT [chen2022ict] and TimesFM-ICF [das2024icf]
- Key findings:
  1. Demo retrieval quality is the dominant factor
  2. Learned aggregation (cross-attention) significantly improves over naive averaging
  3. +4.7% MAE improvement with only 198K params (<1% of OpenCity)
- Positions within a broader ICL spectrum: ICT (train for ICL) → TimesFM-ICF (continued pretraining for ICL) → ours (frozen model ICL via output-space correction)
- Future: embedding retrieval, multi-dataset validation, bridging to true in-context attention
