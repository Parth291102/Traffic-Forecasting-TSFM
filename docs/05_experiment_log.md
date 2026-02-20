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

## Experiment 5: [PENDING] Residual Correction K=1 (fp32)

| Field | Value |
|-------|-------|
| **Date** | TBD |
| **Dataset** | PEMS07M (228 nodes, 5-min interval) |
| **Mode** | `ict` — residual correction approach (v2) |
| **Precision** | float32 (no bfloat16 needed) |
| **Batch Size** | 1 |
| **K (demonstrations)** | 1 |
| **S (prefix selections)** | 1 |
| **Demo Selection** | random |
| **Forward Calls** | 2 per sample (1 query + 1 demo), each exactly 24 patches |
| **Command** | `uv run python Run.py -mode ict -model OpenCity -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1 -batch_size 1 --embed_dim 512 --skip_dim 512 --enc_depth 6` |
| **Expected** | MAE close to or better than 4.50 (zero-shot baseline) |

---

## Experiment 6: [PENDING] Residual Correction K=3 (fp32)

| Field | Value |
|-------|-------|
| **Date** | TBD |
| **K** | 3 |
| **Purpose** | More demos → better error estimation → better correction |
| **Forward Calls** | 4 per sample (1 query + 3 demos) |

---

## Experiment 7: [PENDING] Residual Correction K=1, S=10 (fp32)

| Field | Value |
|-------|-------|
| **Date** | TBD |
| **K** | 1, **S** = 10 |
| **Purpose** | Variance reduction — average predictions over 10 independent demo selections |
| **Forward Calls** | 20 per sample (10 × (1 query + 1 demo)) |

---

## Summary Table

| # | Method | K | S | Precision | MAE | RMSE | MAPE% | CORR | vs ZS | Status |
|---|--------|---|---|-----------|-----|------|-------|------|-------|--------|
| 1 | Zero-Shot | 0 | - | fp32 | 4.50 | 8.21 | 12.20 | 0.736 | — | Baseline |
| 2 | Seq ext naive | 1 | 1 | bf16 | 5.75 | 10.13 | 17.65 | 0.692 | -28% | DEPRECATED |
| 3 | K=0 bf16 | 0 | 1 | bf16 | 4.81 | 8.78 | 14.17 | 0.749 | -7% | DEPRECATED |
| 4 | Seq ext masked | 1 | 1 | bf16 | 4.92 | 8.98 | 14.74 | 0.745 | -9% | DEPRECATED |
| 5 | Residual K=1 | 1 | 1 | fp32 | TBD | TBD | TBD | TBD | TBD | PENDING |
| 6 | Residual K=3 | 3 | 1 | fp32 | TBD | TBD | TBD | TBD | TBD | PENDING |
| 7 | Residual K=1 S=10 | 1 | 10 | fp32 | TBD | TBD | TBD | TBD | TBD | PENDING |
