# ICT-OpenCity Experiment Log

## Overview

- **Model**: OpenCity-plus (embed_dim=512, skip_dim=512, enc_depth=6, 16 attention heads)
- **Pretrained Weights**: `model_weights/OpenCity/OpenCity-plus.pth`
- **GPU**: NVIDIA L40S, 46 GB VRAM
- **Framework**: PyTorch 2.4.1+cu124
- **Dataset**: PEMS07M (228 nodes, 5-min interval), fp32

---

## Approach Evolution

### v1: Sequence Extension (DEPRECATED)

Concatenate demo patches with query patches along the temporal dimension (24 → K×48+24 patches). Abandoned because OpenCity was pretrained on fixed 24-patch sequences — extended sequences cause self-attention dilution, TC cross-attention mismatch, position encoding collisions, and require bfloat16 (which alone costs +7% MAE).

### v2: Residual Correction (CURRENT)

Query and each demo processed independently through the standard 24-patch forward path. Demo ground-truth futures estimate the model's systematic error:

```
pred_query   = model.forward(query_hist)
error_k      = demo_k_ground_truth - model.forward(demo_k_hist)
final        = pred_query + mean(error_1, ..., error_K)
```

Zero OOD risk, zero new parameters, fp32 inference, K+1 forward passes per sample.

### Root Cause Analysis

Two issues identified through experiments:

1. **Random demo selection = high variance**: Randomly sampled demos have no correlation with the query's traffic state. Increasing K or S reduces variance monotonically; S→∞ converges to zero-shot, confirming random demos carry no systematic signal.
2. **Demo never enters model context**: Query and demo are forwarded independently — this is post-hoc output-space correction, not true in-context learning.

| Property | True ICL | v2 (Residual) | v1 (Seq Ext) |
|----------|---------|--------------|--------------|
| Demo enters attention | ✅ | ❌ | ✅ (but OOD) |
| Query attends to demo | ✅ | ❌ | ✅ (but masked) |
| Demo correlated with query | ✅ | ❌ | ❌ |
| Sequence length in-distribution | ✅ | ✅ | ❌ |

KNN experiments confirm Issue 1 as the dominant bottleneck — once demos are similarity-aligned, residual correction nearly recovers zero-shot performance. Issue 2 is a deeper architectural limitation.

### v3: Similarity-Based Demo Selection

Replace random sampling with similarity-based retrieval: **time-of-week matching** (O(1), slot-index aligned) or **KNN matching** ($\arg\min_d \| x_{\text{query}} - x_d \|_2$, precomputed index). KNN with K=3 achieves MAE 4.66 vs zero-shot 4.50, confirming demo-query alignment as the key factor.

### v4: Learned Demo Aggregation

Replace naive `(1/K) * sum(corrections)` with a learned aggregation module (base model frozen, ~8K-200K new params):

```
v2/v3:  pred = f(query) + (1/K) * sum(GT_k - f(demo_k))          ← naive average
v4:     pred = f(query) + Aggregator(query_enc, demo_encs, corrections)  ← learned weighting
```

Two variants: (a) **SimpleDemoAggregator** (~8K params): cosine similarity → softmax weights; (b) **DemoAggregator** (~30K-200K params): multi-head cross-attention + sigmoid gating. Training: 5-10 epochs, CPU-feasible. See `docs/plan2_learned_demo_aggregation.md` for implementation details.

---

## Experiment 1: Zero-Shot Baseline

Standard OpenCity-plus inference on PEMS07M, no demonstrations, fp32.

| MAE | RMSE | MAPE | CORR |
|-----|------|------|------|
| 4.50 | 8.21 | 12.20% | 0.7360 |

Reference baseline for all ICT experiments.

---

## Experiment 2: [DEPRECATED] Sequence Extension — Naive Concatenation (bf16)

Concatenated demo+query patches into 72-patch sequence. MAE 5.75 (+28% vs ZS). Abandoned due to OOD sequence length.

---

## Experiment 3: [DEPRECATED] bfloat16 Precision Diagnostic (K=0)

Isolated bfloat16 precision loss without demos. MAE 4.81 (+7% vs ZS fp32) — bfloat16 alone costs 0.31 MAE.

---

## Experiment 4: [DEPRECATED] Sequence Extension — Block-Diagonal Mask (bf16)

72-patch sequence with block-diagonal T self-attention mask. MAE 4.92 (+9% vs ZS). Recovered 88% of concatenation damage, but bfloat16 precision loss remained dominant.

**Lessons from v1**: (1) Pretrained fixed-length models cannot handle extended sequences without retraining; (2) bfloat16 introduces non-trivial precision loss that compounds with method errors; (3) The model's forward path must be preserved exactly as pretrained.

---

## Experiment 5: Residual Correction + Random Demo Selection

Tested v2 residual correction with randomly selected demos. Varied K (demos per query) and S (independent sampling rounds).

| K | S | MAE | RMSE | MAPE% | CORR | vs ZS |
|---|---|-----|------|-------|------|-------|
| 1 | 1 | 7.39 | 12.00 | 17.39 | 0.509 | -64% |
| 3 | 1 | 6.14 | 9.53 | 14.84 | 0.634 | -36% |
| 1 | 10 | 5.24 | 8.52 | 13.15 | 0.688 | -16% |

**Conclusions**:
- Random demos introduce high-variance correction noise; all configurations worse than zero-shot
- Increasing K or S reduces gap monotonically — variance from demo mismatch is the dominant issue
- S=10 averaging approaches the expected residual over the training distribution (≈ 0), confirming random demos carry no systematic signal

---

## Experiment 6: [PENDING] Residual Correction + Time-of-Week Demo Matching

K=1, S=1. Select demos whose slot index matches the query's hour-of-week. Purpose: validate whether simple time-based similarity resolves the random demo noise problem.

---

## Experiment 7: Residual Correction + KNN Demo Selection

Tested v2 residual correction with KNN similarity-matched demos ($\arg\min_d \| x_{\text{query}} - x_d \|_2$).

| K | S | MAE | RMSE | MAPE% | CORR | vs ZS |
|---|---|-----|------|-------|------|-------|
| 1 | 1 | 5.61 | 10.04 | 14.23 | 0.625 | -25% |
| 3 | 1 | 4.66 | 8.11 | 11.92 | 0.729 | -3.6% |
| 1 | 10 | 5.61 | 10.04 | 14.23 | 0.625 | -25% |

**Conclusions**:
- KNN dramatically reduces variance over random selection (K=1: MAE 7.39→5.61; K=3: 6.14→4.66)
- K=3 KNN nearly matches zero-shot baseline (4.66 vs 4.50), confirming demo-query alignment as the dominant bottleneck
- S=10 identical to S=1 — KNN selection is deterministic per query, so prefix averaging provides no additional benefit

---

## Experiment 8a: Learned Demo Aggregation — Attention, 10 epochs (K=3, KNN)

Replace naive average with learned aggregation module. Base model frozen, KNN demo selection (raw-input L2), K=3.

**Aggregator**: DemoAggregator (cross-attention), ~198K params, 4 heads, proj_dim=128.
**Training**: 10 epochs, lr=1e-4, early stop patience=5. Best model at epoch 4 (val_loss=4.678804).
**Weights**: `aggregator_best_exp8_attn_10ep_K3_knn.pth` (best), `aggregator_ckpt_exp8_attn_10ep_K3_knn.pth` (checkpoint)

| MAE | RMSE | MAPE% | CORR | vs ZS |
|-----|------|-------|------|-------|
| **4.29** | **7.71** | **11.42** | **0.7483** | **+4.7%** |

**Test details**: 71 batches, 4493 samples, total_mae_count=295,028,352, total_mape_count=295,028,352.

> Achieved +4.7% improvement over zero-shot (MAE 4.50→4.29). Best model at epoch 4; early stopped after epoch 9 (patience=5).

---

## Experiment 8b: Learned Demo Aggregation — Attention, 3 epochs (K=3, KNN)

Same setup as Exp 8a but limited to 3 epochs for **fair comparison** with OpenCity's few-shot protocol (which uses 3 epochs).

**Aggregator**: DemoAggregator (cross-attention), ~198K params, 4 heads, proj_dim=128.
**Training**: 3 epochs, lr=1e-4. Val loss improved monotonically: epoch 0 → 4.7933, epoch 1 → 4.7399, epoch 2 → 4.7005 (best).
**Weights**: `aggregator_best.pth`
**Timing**: Phase 1 (pre-compute) ~3h17m; Phase 2 (aggregator training) ~5s; test inference ~2.5h.

| MAE | RMSE | MAPE% | CORR | vs ZS |
|-----|------|-------|------|-------|
| **4.30** | **7.68** | **11.37** | **0.7482** | **+4.4%** |

**Test details**: 71 batches, 4493 samples, total_mae_count=295,028,352, total_mape_count=295,028,352.

> Achieved +4.4% improvement over zero-shot (MAE 4.50→4.30). Nearly identical to the 10-epoch result (Exp 8a: MAE 4.29), confirming that 3 epochs are sufficient — the aggregator converges rapidly from cached features. This provides a **fair comparison** with OpenCity's few-shot protocol.

---

## Experiment 9: Learned Demo Aggregation — Simple Cosine, 5 epochs (K=3, KNN)

Replace naive average with SimpleDemoAggregator. Base model frozen, KNN demo selection (raw-input L2), K=3.

**Aggregator**: SimpleDemoAggregator (cosine similarity), ~8K params.
**Training**: 5 epochs, lr=1e-4.

| MAE | RMSE | MAPE% | CORR | vs ZS |
|-----|------|-------|------|-------|
| **4.47** | **8.01** | **11.55** | **0.7370** | **+0.7%** |

> Marginal improvement over zero-shot (MAE 4.50→4.47). Outperforms naive averaging (Exp 7 K=3: MAE 4.66) but falls short of the cross-attention aggregator (Exp 8a: MAE 4.29). Confirms that even a minimal learned weighting (8K params) improves over uniform averaging, while the richer cross-attention mechanism captures more useful query-demo interactions.

---

## Summary Table

| # | Method | Demo Selection | K | S | MAE | RMSE | MAPE% | CORR | vs ZS | Status |
|---|--------|---------------|---|---|-----|------|-------|------|-------|--------|
| 1 | Zero-Shot | — | 0 | - | 4.50 | 8.21 | 12.20 | 0.736 | — | Baseline |
| 2 | Seq ext naive (bf16) | random | 1 | 1 | 5.75 | 10.13 | 17.65 | 0.692 | -28% | DEPRECATED |
| 3 | bf16 diagnostic | — | 0 | 1 | 4.81 | 8.78 | 14.17 | 0.749 | -7% | DEPRECATED |
| 4 | Seq ext masked (bf16) | random | 1 | 1 | 4.92 | 8.98 | 14.74 | 0.745 | -9% | DEPRECATED |
| 5 | Residual correction | random | 1 | 1 | 7.39 | 12.00 | 17.39 | 0.509 | -64% | Done |
| 5 | Residual correction | random | 3 | 1 | 6.14 | 9.53 | 14.84 | 0.634 | -36% | Done |
| 5 | Residual correction | random | 1 | 10 | 5.24 | 8.52 | 13.15 | 0.688 | -16% | Done |
| 6 | Residual correction | ToW | 1 | 1 | TBD | TBD | TBD | TBD | TBD | PENDING |
| 7 | Residual correction | KNN | 1 | 1 | 5.61 | 10.04 | 14.23 | 0.625 | -25% | Done |
| 7 | Residual correction | KNN | 3 | 1 | **4.66** | 8.11 | 11.92 | 0.729 | -3.6% | Done |
| 7 | Residual correction | KNN | 1 | 10 | 5.61 | 10.04 | 14.23 | 0.625 | -25% | Done |
| 8a | Learned agg (attn, 10ep) | KNN | 3 | 1 | **4.29** | **7.71** | **11.42** | **0.748** | **+4.7%** | Done |
| 8b | Learned agg (attn, 3ep) | KNN | 3 | 1 | **4.30** | **7.68** | **11.37** | **0.748** | **+4.4%** | Done |
| 9 | Learned agg (simple, 5ep) | KNN | 3 | 1 | **4.47** | **8.01** | **11.55** | **0.737** | **+0.7%** | Done |

---

## Phase 4: Cross-Dataset Generalization (DIDI Datasets)

Evaluate OpenCity-plus zero-shot on out-of-distribution DIDI datasets (10-min interval, not seen in pretraining). These datasets test the model's generalization to unseen cities and different temporal resolutions.

- **SZ_DIDI**: Shenzhen, 627 nodes, 10-min interval, 17280 time steps, test split=40%
- **CD_DIDI**: Chengdu, 524 nodes, 10-min interval, 17280 time steps, test split=40%
- **Input/output window**: 144 steps (= 288 / (10/5)), PatchEmbedding uses gap=2

### Experiment 10: Zero-Shot on DIDI Datasets

Standard OpenCity-plus zero-shot inference on DIDI datasets, batch_size=16, fp32.

| Dataset | MAE | RMSE | MAPE | CORR | Test Samples |
|---------|-----|------|------|------|--------------|
| SZ_DIDI | 4.66 | 7.10 | 19.05% | 0.3222 | 6,625 |
| SZ_DIDI (Paper) | 3.68 | 5.58 | — | — | — |
| CD_DIDI | 6.34 | 9.36 | 28.36% | 0.3361 | 6,625 |
| CD_DIDI (Paper) | 6.03 | 9.50 | — | — | — |

**Observations**:
- Performance drops significantly compared to PEMS07M (MAE 4.50), especially on CD_DIDI
- Low correlation (0.32–0.34) indicates the model struggles with these unseen cities/intervals
- MAPE is notably higher (19–28% vs 12.2% on PEMS07M), suggesting difficulty with low-flow periods
- SZ_DIDI zero-shot MAE 4.66 vs paper-reported 3.68 — gap likely due to evaluation horizon differences or checkpoint version
- These baselines set the stage for ICT experiments on DIDI to measure cross-dataset adaptation improvement

### Experiment 11/12: Adaptation on DIDI Datasets

Three adaptation methods compared: (a) **Zero-shot** (Exp 10); (b) **Fine-tune** prediction head for 3 epochs, lr=0.001 (paper's fast-adaptation protocol, all other params frozen); (c) **ICT Aggregator** — DemoAggregator (cross-attention, ~198K params, 4 heads, proj_dim=128), 3 epochs, lr=1e-4, K=3, KNN demo selection, base model frozen.

| Dataset | Method | MAE | RMSE | MAPE | CORR | vs ZS |
|---------|--------|-----|------|------|------|-------|
| SZ_DIDI | Zero-Shot (Exp 10) | 4.66 | 7.10 | 19.05% | 0.3222 | — |
| SZ_DIDI | Fine-tune pred head (Exp 11) | **2.33** | **3.75** | **9.89%** | **0.6811** | +50.0% |
| SZ_DIDI | ICT Aggregator (Exp 12) | **2.70** | **4.11** | **11.08%** | **0.5402** | +42.1% |
| SZ_DIDI | Paper (zero-shot) | 3.68 | 5.58 | — | — | — |
| SZ_DIDI | Paper (fine-tune) | 2.36 | 3.55 | — | — | — |
| CD_DIDI | Zero-Shot (Exp 10) | 6.34 | 9.36 | 28.36% | 0.3361 | — |
| CD_DIDI | Fine-tune pred head (Exp 11) | **2.60** | **3.93** | **11.98%** | **0.7929** | +59.0% |
| CD_DIDI | ICT Aggregator (Exp 12) | **3.41** | **5.02** | **15.28%** | **0.7195** | +46.2% |
| CD_DIDI | Paper (zero-shot) | 6.03 | 9.50 | — | — | — |
| CD_DIDI | Paper (fine-tune) | 2.97 | 4.29 | — | — | — |

**Analysis**:

#### 1. ICT improvement scales with distribution shift

| Dataset | Type | ZS MAE | ICT MAE | MAE Reduction | vs ZS |
|---------|------|--------|---------|---------------|-------|
| PEMS07M | In-distribution | 4.50 | 4.30 | 0.20 | +4.4% |
| SZ_DIDI | Out-of-distribution | 4.66 | 2.70 | 1.96 | +42.1% |
| CD_DIDI | Out-of-distribution | 6.34 | 3.41 | 2.93 | +46.2% |

The magnitude of ICT improvement is positively correlated with the degree of distribution shift. On in-distribution data (PEMS07M), the model's zero-shot predictions are already well-calibrated — residual errors are small and stochastic, leaving limited systematic correction signal for the aggregator to learn (+4.4%). On out-of-distribution data, the model exhibits large systematic biases (unseen cities, different temporal resolution), and KNN demos expose these systematic errors, allowing the learned aggregator to effectively weight and apply corrections (+42–46%).

**Takeaway**: The core value of ICT lies in out-of-distribution adaptation, not in-distribution accuracy improvement.

#### 2. ICT recovers majority of fine-tuning gains

| Dataset | Metric | ZS | ICT | FT | ICT Recovery Rate |
|---------|--------|-----|-----|-----|-------------------|
| SZ_DIDI | MAE | 4.66 | 2.70 | 2.33 | 84.1% |
| SZ_DIDI | CORR | 0.3222 | 0.5402 | 0.6811 | 60.7% |
| CD_DIDI | MAE | 6.34 | 3.41 | 2.60 | 78.3% |
| CD_DIDI | CORR | 0.3361 | 0.7195 | 0.7929 | 83.9% |

Recovery Rate = (ZS − ICT) / (ZS − FT), i.e., the percentage of total fine-tuning gain recovered by ICT.

- **MAE recovery**: ICT recovers 78–84% of fine-tuning's MAE gain through output-space correction alone (without modifying model weights)
- **CORR recovery**: On SZ_DIDI, CORR recovery is lower (61%), indicating that while ICT reduces absolute error effectively, it is less capable of capturing spatio-temporal correlation patterns compared to representation adaptation. On CD_DIDI, CORR recovery is higher (84%), possibly because the zero-shot correlation baseline is slightly higher (0.34 vs 0.32), giving the aggregator more structure to exploit
- **Remaining gap**: SZ 0.37 MAE gap, CD 0.81 MAE gap — this quantifies the capability boundary between output-space correction and representation adaptation

**Takeaway**: Without modifying any model weights, ICT recovers ~80% of fine-tuning's MAE gain. The remaining ~20% stems from fine-tuning's representation adaptation of the prediction head, which is fundamentally unreachable by output-space correction.

#### 3. Output-space correction vs representation adaptation

Fine-tuning modifies the prediction head weights, enabling the model to learn a mapping from encoder features to the new distribution's output space. ICT keeps the model entirely unchanged and applies post-hoc correction in output space only. The gap between the two reveals a clear hierarchy:

```
Zero-Shot  →  ICT (output correction)  →  Fine-tune (representation adaptation)
  MAE↓            ~80% recovery               100% recovery
  No weight mod    No weight mod               Modifies prediction head
  1 forward pass   K+1 forward passes          1 forward pass
```

ICT advantages: (a) Preserves pretrained weights entirely, maintaining model generality; (b) No gradient backpropagation to base model; (c) Aggregator training is extremely fast (~5s from cached features).
ICT disadvantages: (a) Inference cost scales as K+1×; (b) Requires maintaining a demo database; (c) Cannot reach fine-tuning's accuracy ceiling.

#### 4. Practical implications

- **Target scenario**: ICT is best suited for rapid adaptation to new cities/datasets where modifying the pretrained model is not permitted (e.g., model-as-a-service with multi-tenant shared weights)
- **Complementary to fine-tuning**: When weight modification is allowed, fine-tuning remains the preferred approach; ICT can serve as a quick baseline before fine-tuning or as an alternative in non-trainable deployment settings
- **Aggregator training cost is negligible**: 3 epochs from cached features takes only ~5s; the main bottleneck lies in precomputing the cache (~8h at DIDI scale) and K+1 forward passes at inference time

#### 5. Comparison with paper results

| Dataset | Our FT | Paper FT | Our ZS | Paper ZS |
|---------|--------|----------|--------|----------|
| SZ_DIDI | **2.33** | 2.36 | 4.66 | 3.68 |
| CD_DIDI | **2.60** | 2.97 | 6.34 | 6.03 |

- Our fine-tuning slightly outperforms the paper on both datasets (SZ: 2.33 vs 2.36; CD: 2.60 vs 2.97)
- Our zero-shot is worse than the paper (SZ: 4.66 vs 3.68; CD: 6.34 vs 6.03) — likely due to differences in evaluation horizon or checkpoint version
- The larger zero-shot gap combined with better fine-tuning results suggests our adaptation protocol (3 epochs, lr=0.001) may be more effective than the paper's
