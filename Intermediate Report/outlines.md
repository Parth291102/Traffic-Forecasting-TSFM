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

- **Problem**: Pretrained spatio-temporal foundation models (e.g., OpenCity [1]) achieve strong zero-shot traffic forecasting, but performance degrades under cross-city distribution shifts. The only available adaptation path — gradient-based fine-tuning of the prediction head — operates at the dataset level: it adjusts model weights once for the entire target domain, providing no mechanism to tailor predictions to individual test queries. Furthermore, fine-tuning modifies base model parameters, which risks eroding the broadly generalizable representations learned during pretraining.
- **Opportunity**: In-context learning (ICL) offers an alternative: at inference time, the model conditions each prediction on a small set of retrieved demonstrations, enabling per-query adaptation without any weight updates. ICL has been formalized as a meta-learning objective in NLP via in-context tuning [2] and demonstrated effective for time-series foundation models via in-context fine-tuning [3], yet remains entirely unexplored for spatio-temporal foundation models.
- **Method**: Inspired by [2] and [3], we propose a lightweight, pluggable ICL module that keeps the base ST foundation model completely frozen. Two phases: (1) offline — a small aggregation network (8K-198K params) is trained in seconds from cached features to learn *how to weight corrections*, not what to predict; (2) online — for each test query, KNN retrieves similar demonstrations, the frozen model estimates their prediction errors, and the trained aggregator applies a per-query correction at inference time with zero gradient updates.
- **Key Result**: On PEMS07M with OpenCity-plus, our best configuration reduces MAE from 4.50 to 4.29 (+4.7%) with only 198K trainable parameters (~0.8% of the 26M-param base model), approaching full-shot supervised baselines while the backbone remains entirely frozen.

---

## Section I: Introduction

### Para 1 — OpenCity: Success, Limitations, and Opportunity

- Traffic forecasting is critical for urban management. Traditional ST models (STGCN, GWN, etc.) achieve strong supervised performance but require per-dataset training. [1-2 sentences, cite briefly]
- **OpenCity** [li2024opencity] represents a new paradigm: a spatio-temporal foundation model pretrained on large-scale heterogeneous traffic data, achieving zero-shot transfer across cities, data categories, and time horizons. It integrates Transformer architecture with GNNs and demonstrates promising scaling laws.
- **Limitation 1**: Despite strong zero-shot capability, performance degrades under larger cross-city distribution shifts. For example, on SZ-DIDI and CD-DIDI, OpenCity's zero-shot MAE falls noticeably behind full-shot supervised baselines, exposing the limits of purely static zero-shot inference.
- **Limitation 2**: The only available adaptation mechanism is gradient-based fine-tuning — updating the prediction head for 3 epochs on target-domain data. This approach has two fundamental limitations: (a) it modifies base model parameters, which risks overwriting generalizable representations acquired during pretraining; (b) it operates at the *dataset level* — once fine-tuned, the model produces the same prediction behavior for every test query, with no mechanism to leverage individual query-relevant historical examples at inference time.
- **Opportunity**: These limitations point to a missing capability: *per-query inference-time adaptation*, i.e., the ability to condition each prediction on retrieved demonstrations without touching model weights. OpenCity's strong frozen representations make it a prime candidate for such an approach.

### Para 2 — ICL: From Meta-Learning in NLP to Time Series

- In NLP, in-context learning (ICL) allows foundation models to adapt at test time by conditioning on demonstrations, without weight updates [brown2020gpt3, brief mention].
- **ICT** [chen2022ict] (our idea source): Formalizes ICL as a meta-learning objective. By fine-tuning LMs to predict target labels given concatenated instruction + in-context examples + query, ICT trains models to *learn from demonstrations* as a general capability. Key contributions:
  - Bridges prompting, fine-tuning, and meta-learning into a unified framework
  - Outperforms gradient-based meta-learning (MAML) by leveraging LM inductive bias for pattern matching
  - Reduces sensitivity to example ordering (6x) and selection (2x)
  - Core insight for our work: **models can be trained to extract useful patterns from contextual examples without modifying weights at test time**
- **TimesFM-ICF** [das2024icf] (our application reference): Extends ICL to time-series foundation models. The model is trained to use related time-series in its context window (via separator tokens and cross-example attention) to forecast a target series. Key contributions:
  - Demonstrates ICL works beyond language — in numerical time-series forecasting
  - Achieves 7-25% improvement over the base FM, even rivaling per-dataset fine-tuning
  - Core insight for our work: **providing related examples at inference time helps the model adapt to target distributions without gradient updates**
- **Our question**: Can we bring ICL capability to frozen spatio-temporal foundation models like OpenCity?

### Para 3 — Challenges and Our Approach

- **Challenge — how to bring ICL to OpenCity**: Both ICT and TimesFM-ICF enable ICL by modifying the model (fine-tuning or continued pretraining with architectural changes). For ST foundation models like OpenCity, such modification is non-trivial: the model is pretrained on fixed-length sequences (288 timesteps = 24 patches) with specific attention patterns and positional encodings. Naively extending input sequences to include demonstrations causes OOD behavior and 9-28% MAE degradation in our preliminary experiments. A different approach is needed.
- **Our insight**: Instead of modifying the model to accept demonstrations in its context window, we process query and demonstrations *independently* through the frozen model and combine their information in *output space* via residual correction. Each demo's prediction error estimates the model's systematic bias on similar inputs; aggregating these errors yields a per-query correction applied at inference time.
- **How this differs from fine-tuning**: OpenCity's fast adaptation (3-epoch fine-tuning) trains the prediction head on target-domain data — it adjusts *what the model predicts* and produces the same adapted behavior for every subsequent query. Our aggregator, by contrast, is trained to learn *how to weight corrections from retrieved demonstrations* — a correction strategy, not a domain-specific predictor. Once trained (seconds, from cached features), it enables each test query to be individually adapted at inference time via its own retrieved demonstrations. The base model remains frozen throughout; domain adaptation comes from demonstrations, not from parameter updates.
- **Two axes of improvement**: (1) KNN retrieval ensures each query receives relevant demonstrations; (2) learned aggregation (8K-198K params) replaces naive averaging with query-aware weighting.
- **Direct experimental comparison**: On SZ-DIDI and CD-DIDI — the datasets where OpenCity reports its 3-epoch fast adaptation results — we run our 3-epoch aggregator under the same training budget. This provides a direct, fair comparison: same epochs, same training data, different adaptation mechanism (dataset-level weight update vs. per-query inference-time correction with frozen backbone).

### Para 4 — Contributions

1. Inspired by ICT [chen2022ict] and TimesFM-ICF [das2024icf], we propose a residual correction framework that enables **per-query inference-time adaptation** for frozen ST foundation models — without architectural modification, without base model gradients, and without retraining for each new dataset.
2. We systematically study two adaptation axes: demonstration retrieval (random vs. KNN) and correction aggregation (naive average vs. cosine similarity vs. cross-attention), showing both are critical and jointly necessary to surpass zero-shot performance.
3. On PEMS07M with OpenCity-plus, our best configuration achieves +4.7% MAE improvement over zero-shot (4.50→4.29) with only 198K external parameters (~0.8% of the base model). Under the same 3-epoch training budget as OpenCity's fast adaptation, our frozen-backbone approach offers a direct comparison point on SZ-DIDI and CD-DIDI (planned).

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

  | Paradigm | When adapted | Base model | Granularity |
  |----------|-------------|------------|-------------|
  | Fine-tuning (OpenCity fast-adapt) | Once per dataset | Modified | Dataset-level |
  | TTA (TTT, TENT) | Per test batch, at test time | Modified | Batch-level |
  | **Ours** | Aggregator trained offline; correction applied per query | **Frozen** | **Per-query** |

  - TTA requires a self-supervised signal or statistics at test time; our method uses pre-computed training demonstrations, requiring no test-time gradient
  - TTA modifies base model parameters on every batch; our aggregator is trained once and then applied at inference with no gradient updates anywhere
  - Our method is complementary to TTA — the aggregator's per-query correction could be further refined by TTA on top

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
- **Connection to ICT/TimesFM-ICF**: Like ICT, we use demonstrations to adapt model behavior at inference time. Like TimesFM-ICF, we leverage related examples to condition predictions on target distributions. Unlike both — and unlike fine-tuning — we achieve this entirely in output space without touching any model parameters at inference time. The aggregator is trained offline to learn *a correction strategy* (how to combine residual errors), not to memorize domain-specific predictions; once trained, domain adaptation comes from the retrieved demonstrations at inference time, not from learned weights.

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
- **Key distinction from fine-tuning**: Fine-tuning requires repeated forward and backward passes through the full base model on target-domain data. Our Phase 2 operates entirely on pre-cached features and corrections — the base model is touched only once in Phase 1, and never during aggregator training or inference-time correction.

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

> Both methods use 3 epochs of adaptation. OpenCity fast-adapt modifies the prediction head (dataset-level). Our method trains a frozen-backbone aggregator (per-query at inference time).

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
3. **Our method has trainable parameters**: Unlike purely training-free approaches, our aggregation module requires an offline training phase. The key distinction from fine-tuning is not the absence of trainable parameters, but rather: (a) the base model remains completely frozen; (b) what is learned is a *correction strategy* (how to weight residuals), not domain-specific predictions; (c) once trained, adaptation is delivered per-query at inference time through demonstrations, not through learned weights.
4. **Retrieval ceiling**: Raw-input L2 KNN may miss semantic similarity in high-dimensional feature space; planned embedding-space KNN is expected to improve this.

### Planned Experiments (TBD)
- **Enc-KNN retrieval**: embedding-space KNN using frozen encoder features (node-mean pooling, 512-d); expected to improve over raw-input L2 KNN
- **K sweep**: K ∈ {1, 3, 5} for both naive avg and Cross-Attn aggregator
- **Multi-dataset validation** (SZ-DIDI, CD-DIDI): direct fair comparison against OpenCity fast adaptation (3-epoch prediction head fine-tuning) — same training budget, different adaptation mechanism
- **Analysis**: per-hour MAE curves, attention weight visualization, per-node improvement map

### Broader Connections
- **Adaptation paradigm spectrum**: Fine-tuning (dataset-level, modifies model) → TTA (batch-level, modifies model at test time) → ICT/TimesFM-ICF (ICL-capable model, per-query, modifies model) → **Ours** (per-query at inference, frozen backbone, external module)
- **ICL progression**: ICT [chen2022ict] trains models to perform ICL via a meta-learning objective → TimesFM-ICF [das2024icf] extends ICL to time-series via continued pretraining → our work shows ICL-like per-query adaptation is achievable for frozen ST models via output-space correction, without any model modification
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
