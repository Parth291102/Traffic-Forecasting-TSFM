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
