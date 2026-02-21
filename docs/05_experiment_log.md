# ICT-OpenCity Experiment Log

## Overview

This document tracks all experiments for the ICT (In-Context Tuning) integration into OpenCity.

- **Model**: OpenCity-plus (embed_dim=512, skip_dim=512, enc_depth=6, 16 attention heads)
- **Pretrained Weights**: `model_weights/OpenCity/OpenCity-plus.pth`
- **GPU**: NVIDIA L40S, 46 GB VRAM
- **Framework**: PyTorch 2.4.1+cu124
- **Branch**: `xzhou38-ic-opencity-baseline`

---

## Approach Evolution

### v1: Sequence Extension (DEPRECATED)

Original plan from integration doc: concatenate demo patches with query patches along the temporal dimension, extending the sequence from 24 to K×48+24 patches. All patches jointly processed through encoder blocks.

**Why abandoned**: OpenCity was pretrained on fixed 24-patch sequences. Extended sequences are OOD — causing self-attention dilution, TC cross-attention mismatch, and position encoding collisions. Even with block-diagonal masking (Exp 4), bfloat16 was required for VRAM, introducing further precision loss. Fundamental mismatch between fixed-length pretraining and variable-length ICT inference.

### v2: Residual Correction (CURRENT)

Each input (query and each demo) is processed independently through the standard 24-patch `forward()` path — identical to pretraining. Demo ground-truth futures estimate the model's systematic error, which corrects the query prediction:

```
pred_query   = model.forward(query_hist)         # standard 24-patch inference
pred_demo_k  = model.forward(demo_k_hist)        # same standard inference
error_k      = demo_k_ground_truth - pred_demo_k # model's systematic error on demo k
correction   = mean(error_1, ..., error_K)        # averaged correction
final        = pred_query + correction            # corrected prediction
```

**Advantages**:
- Zero OOD: every forward call uses exactly 24 patches
- Zero new parameters, zero architecture changes
- fp32 inference: no VRAM increase (same memory as zero-shot per call)
- Physically interpretable: estimates and corrects systematic model bias
- K+1 sequential forward passes (K=1 → 2× compute, trivially parallelizable)

### v2 Root Cause Analysis (after Exp 5–7)

Experiments 5, 6, and 7 reveal a consistent picture. Results summary:

| Exp | K | S | MAE | vs ZS |
|-----|---|---|-----|-------|
| 5 | 1 | 1 | 7.39 | -64% |
| 6 | 3 | 1 | 6.14 | -36% |
| 7 | 1 | 10 | 5.24 | -16% |

Increasing K (more demos per query) or S (more independent sampling rounds) consistently reduces the gap to zero-shot. This monotone convergence pattern confirms the root cause is **variance from random demo selection**, not a fundamental flaw in the residual correction idea.

**Issue 1: No correlation between demo and query (random sampling)**

The demo pool comes from the training split (time range 0–50%), while test queries are in time range 60–100%. Randomly sampled demos have no relationship to the traffic state at query time — different congestion patterns, different positions in the daily/weekly cycle. `demo_gt - demo_pred` reflects the model's prediction error at that specific demo timestep, which has no necessary alignment with the model's error at query time.

Exp 7 (S=10) is decisive: averaging 10 independent random-demo corrections approaches the **expected residual over the training distribution, which is ≈ 0**. The method degrades to zero-shot as S→∞ with random selection, confirming random demos carry no systematic signal. The correction variance, not a wrong direction, is what hurts performance at low S/K.

**Issue 2: Demo never enters the model's context (not truly In-Context)**

In v2, query and demo are forwarded completely independently with no interaction between the two passes. Demo information never enters the model's attention computation — the query's token representation is identical to zero-shot inference. This is **post-hoc output-space correction**, not in-context learning. True ICL requires query tokens to "see" the demo (x, y) pairs through attention and extract a task representation from them.

| Property | True ICL | v2 (Residual) | v1 (Seq Ext) |
|----------|---------|--------------|--------------|
| Demo enters attention | ✅ | ❌ | ✅ (but OOD sequence) |
| Query attends to demo | ✅ | ❌ | ✅ (but blocked by mask) |
| Demo correlated with query | ✅ | ❌ | ❌ |
| Sequence length in-distribution | ✅ | ✅ | ❌ |

**Conclusion**: Issue 1 is the actionable bottleneck. If demos are selected to match the query's temporal pattern, their residuals should point in a consistent, useful direction — the expected correction would be non-zero and predictive. Exp 8 (time-of-week matching) directly tests this hypothesis. Issue 2 is a deeper architectural limitation that requires demos to participate in the forward pass.

### v3 Direction: Similarity-Based Demo Selection

Replace random sampling with similarity-based retrieval so that the demo residual direction is aligned with the model's true error at query time:

- **Time-of-week matching** (simple, O(1)): select demos whose slot index matches the query's hour-of-week from the training pool
- **KNN matching** (more accurate, requires precomputed index): $\text{demo}^* = \arg\min_{d \in \text{pool}} \| x_{\text{query}} - x_d \|_2$, retrieve the K training windows whose history is closest to the query history

Traffic data has strong periodicity (daily period 288 steps, weekly period 2016 steps at 5-min intervals). Time-of-week matching is expected to substantially reduce residual variance.

---

## Experiment 1: Zero-Shot Baseline (mode=test)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Dataset** | PEMS07M (228 nodes, 5-min interval) |
| **Mode** | `test` (standard zero-shot inference) |
| **Precision** | float32 |
| **Batch Size** | 1 |
| **Command** | `uv run python Run.py -mode test -model OpenCity -load_pretrain_path OpenCity-plus.pth -batch_size 1 --embed_dim 512 --skip_dim 512 --enc_depth 6` |

### Results

| MAE | RMSE | MAPE | CORR |
|-----|------|------|------|
| 4.50 | 8.21 | 12.1980% | 0.7360 |

### Notes
- Standard OpenCity inference with 24 patches, no demonstrations.
- This is the reference baseline all ICT experiments are compared against.

---

## Experiment 2: [DEPRECATED] Sequence Extension — Naive Concatenation (bf16)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Approach** | v1 — concatenate demo+query patches into 72-patch sequence |
| **Precision** | bfloat16 |
| **Result** | MAE 5.75 (+28% vs ZS) |
| **Status** | **DEPRECATED** — approach abandoned due to OOD sequence length |

---

## Experiment 3: [DEPRECATED] bfloat16 Precision Diagnostic (K=0)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Purpose** | Isolate bfloat16 precision loss from demo concatenation damage |
| **Result** | MAE 4.81 (+7% vs ZS fp32) — bfloat16 alone costs 0.31 MAE |
| **Status** | **DEPRECATED** — bfloat16 no longer used in v2 approach |

---

## Experiment 4: [DEPRECATED] Sequence Extension — Block-Diagonal Mask (bf16)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Approach** | v1 — 72-patch sequence with block-diagonal T self-attention mask |
| **Precision** | bfloat16 |
| **Result** | MAE 4.92 (+9% vs ZS). Recovered 88% of concatenation damage, but bfloat16 precision loss (0.31 MAE) remained dominant. |
| **Status** | **DEPRECATED** — approach abandoned; v2 residual correction eliminates both issues |

### Lessons Learned from v1 Experiments

1. Pretrained fixed-length models cannot handle extended sequences without retraining
2. Block-diagonal masking partially mitigates attention dilution but cannot fix position encoding conflicts
3. bfloat16 introduces non-trivial precision loss (7% MAE) that compounds with method errors
4. The model's forward path must be preserved exactly as pretrained — any deviation degrades performance

---

## Experiment 5: Residual Correction K=1 (fp32)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Dataset** | PEMS07M (228 nodes, 5-min interval) |
| **Mode** | `ict` — residual correction approach (v2) |
| **Precision** | float32 (no bfloat16 needed) |
| **Batch Size** | 1 |
| **K (demonstrations)** | 1 |
| **S (prefix selections)** | 1 |
| **Demo Selection** | random |
| **Forward Calls** | 2 per sample (1 query + 1 demo), each exactly 24 patches |
| **Command** | `uv run python Run.py -mode ict -model OpenCity -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1 -batch_size 1 --embed_dim 512 --skip_dim 512 --enc_depth 6` |

### Results

| MAE | RMSE | MAPE | CORR |
|-----|------|------|------|
| 7.39 | 12.00 | 17.3880% | 0.5093 |

### Notes
- Worse than zero-shot (+64% MAE). Random single demo introduces high-variance correction noise.
- Single demo's residual is a noisy estimate of model bias — signal-to-noise ratio too low.
- Root cause: no temporal or pattern correlation between demo and query (see v2 Root Cause Analysis).

---

## Experiment 6: Residual Correction K=3 (fp32)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Dataset** | PEMS07M (228 nodes, 5-min interval) |
| **K** | 3 |
| **S** | 1 |
| **Batch Size** | 32 |
| **Forward Calls** | 4 per sample (1 query + 3 demos) |
| **Command** | `uv run python Run.py -mode ict -model OpenCity -load_pretrain_path OpenCity-plus.pth -num_demonstrations 3 -batch_size 32 --embed_dim 512 --skip_dim 512 --enc_depth 6` |

### Results

| MAE | RMSE | MAPE | CORR |
|-----|------|------|------|
| 6.14 | 9.53 | 14.8416% | 0.6344 |

### Notes
- Better than K=1 by 17% (MAE 7.39 → 6.14), but still +36% worse than zero-shot baseline (4.50).
- Averaging 3 residuals reduces variance but systematic bias from random demo mismatch remains.
- The K=1→K=3 trend indicates variance is the dominant issue rather than systematic bias; similarity-based matching should reduce both simultaneously.

---

## Experiment 7: Residual Correction K=1, S=10 (fp32)

| Field | Value |
|-------|-------|
| **Date** | 2026-02-20 |
| **Dataset** | PEMS07M (228 nodes, 5-min interval) |
| **K** | 1, **S** = 10 |
| **Batch Size** | 32 |
| **Purpose** | Variance reduction — average predictions over 10 independent demo selections |
| **Forward Calls** | 20 per sample (10 × (1 query + 1 demo)) |
| **Command** | `uv run python Run.py -mode ict -model OpenCity -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1 -num_prefix_selections 10 -batch_size 32 --embed_dim 512 --skip_dim 512 --enc_depth 6` |

### Results

| MAE | RMSE | MAPE | CORR |
|-----|------|------|------|
| 5.24 | 8.52 | 13.1476% | 0.6877 |

### Notes
- Better than K=1 S=1 by 29% (MAE 7.39 → 5.24), confirming variance is the dominant issue.
- Still +16% worse than zero-shot (4.50), but gap is closing with more selections.
- Averaging 10 random draws approximates the expected residual across the training distribution, which is near-zero — confirming random demos carry no systematic signal for the query.

---

## Experiment 8: [PENDING] Residual Correction K=1, Time-of-Week Demo Matching (fp32)

| Field | Value |
|-------|-------|
| **Date** | TBD |
| **K** | 1, **S** = 1 |
| **Demo Selection** | time-of-week matching — demo slot index matches query's hour-of-week |
| **Purpose** | Validate whether similarity-based selection resolves the random demo noise problem |
| **Forward Calls** | 2 per sample (1 query + 1 demo) |

---

## Experiment 9: [PENDING] Residual Correction K=3, KNN Demo Matching (fp32)

| Field | Value |
|-------|-------|
| **Date** | TBD |
| **K** | 3, **S** = 1 |
| **Demo Selection** | KNN — retrieve K nearest training windows by L2 distance on query history |
| **Purpose** | Compare KNN matching vs time-of-week matching vs random sampling |
| **Forward Calls** | 4 per sample (1 query + 3 demos) |

---

## Summary Table

| # | Method | K | S | Precision | MAE | RMSE | MAPE% | CORR | vs ZS | Status |
|---|--------|---|---|-----------|-----|------|-------|------|-------|--------|
| 1 | Zero-Shot | 0 | - | fp32 | 4.50 | 8.21 | 12.20 | 0.736 | — | Baseline |
| 2 | Seq ext naive | 1 | 1 | bf16 | 5.75 | 10.13 | 17.65 | 0.692 | -28% | DEPRECATED |
| 3 | K=0 bf16 | 0 | 1 | bf16 | 4.81 | 8.78 | 14.17 | 0.749 | -7% | DEPRECATED |
| 4 | Seq ext masked | 1 | 1 | bf16 | 4.92 | 8.98 | 14.74 | 0.745 | -9% | DEPRECATED |
| 5 | Residual K=1 random | 1 | 1 | fp32 | 7.39 | 12.00 | 17.39 | 0.509 | -64% | Done |
| 6 | Residual K=3 random | 3 | 1 | fp32 | 6.14 | 9.53 | 14.84 | 0.634 | -36% | Done |
| 7 | Residual K=1 S=10 | 1 | 10 | fp32 | 5.24 | 8.52 | 13.15 | 0.688 | -16% | Done |
| 8 | Residual K=1 ToW | 1 | 1 | fp32 | TBD | TBD | TBD | TBD | TBD | PENDING |
| 9 | Residual K=3 KNN | 3 | 1 | fp32 | TBD | TBD | TBD | TBD | TBD | PENDING |
