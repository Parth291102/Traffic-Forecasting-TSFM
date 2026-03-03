# Plan 1: Cross-Example Attention — Full Model Modification + Continued Pretraining

## 1. Core Idea

Faithfully implement the approach from "In-Context Fine-Tuning for Time-Series Foundation Models" (Das et al., 2024): concatenate demonstration examples with the query sequence along the **temporal/patch dimension**, enabling the encoder's attention mechanism to directly attend across examples. The model learns to leverage in-context examples through continued pretraining.

### Paper Method vs Current Method

| | Paper (TimesFM-ICF) | Current OpenCity ICT |
|---|---|---|
| Model | Decoder-only, 200M params | Encoder-based, ST attention |
| Demo usage | Concatenated as one long sequence | K+1 independent forward passes |
| Learning mechanism | Cross-example causal attention | No learning, naive residual averaging |
| Formula | f(demo1, SEP, demo2, ..., query) → pred | f(query) + mean(GT - f(demo)) |
| Training required | Continued pretraining | None |

### Adaptation Strategy

OpenCity's `TemporalSelfAttention` computes attention over the patch dimension: `[B, N, H, num_patches, num_patches]`. In a standard forward pass, `num_patches=2` (24 timesteps / patch_len=12).

**Key Insight**: By concatenating demo patches with query patches, num_patches grows from 2 to `3K+2`. The existing temporal attention naturally becomes cross-example attention without architectural changes to the attention module itself.

## 2. Sequence Construction

```
[demo1_p1, demo1_p2, SEP1, demo2_p1, demo2_p2, SEP2, ..., demoK_p1, demoK_p2, SEPK, query_p1, query_p2]
```

- Each demo: 2 patches + 1 separator token = 3 positions
- Query: 2 patches
- Total length: `T_total = 3K + 2`
- K=10 → 32 patches; K=50 → 152 patches

## 3. Attention Mask

Uses a **bidirectional-within-example, causal-across-examples** masking strategy:

```
demo_k patches:   can attend to itself + all previous demos (including their separators)
separator_k:      can attend to demo_k and all preceding content
query patches:    can attend to everything (all demos + separators + self)
```

Mask shape: `[T_total, T_total]` bool tensor, passed via the existing `t_attn_mask` parameter of `TemporalSelfAttention`.

## 4. Implementation Steps

### Step 1: Modify `OpenCity/model/OpenCity/OpenCity.py`

#### 1a. Add separator token parameters (in `__init__`, after `self.linear`)

```python
# ICT Plan1: Learnable separator embeddings
self.separator_embedding = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))  # for enc
self.separator_th = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))         # for TH
self.separator_tp = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))         # for TP
```

#### 1b. New method `_build_ict_attention_mask(self, K, T_total, device)`

Constructs a `[T_total, T_total]` bool mask:
- Iterates over K demos, each occupying 3 positions (k*3 and k*3+1 are patches, k*3+2 is SEP)
- Bidirectional within each demo + can attend to all preceding demos
- Query (last 2 positions) can attend to everything

#### 1c. New method `forward_ict_concat(self, input, lbls, demos_x, demos_y, select_dataset)`

Core flow:
1. **Per-example Instance Normalization**: query and each demo normalized independently (preserving per-example IN stats for DeIN)
2. **Per-example Patch Embedding**: each demo and query passed through shared `patch_embedding_flow` → `[B, 2, N, D]` and `patch_embedding_time` → `(TH, TP)` each `[B, 2, N, D]`
3. **Sequence concatenation**: interleave demo patches with separators, append query → `[B, 3K+2, N, D]` (same for TH, TP)
4. **Build attention mask**: `_build_ict_attention_mask(K, 3K+2, device)`
5. **Encoder**: `encoder_blocks` run normally with `t_attn_mask=mask`
6. **Extract outputs**: query from last 2 positions, each demo from its respective 2 positions
7. **Prediction head + DeIN**: separate prediction + denormalization for query and each demo
8. **Return**: `(query_pred, demo_preds)` — demo_preds used in training loss

Tensor flow:
```
demo_k flow:       [B, T, N, 1] → patch_embed → [B, 2, N, D]
separator:         [1, 1, 1, D] → expand → [B, 1, N, D]
concatenated:      [B, 3K+2, N, D]
encoder output:    [B, 3K+2, N, D]
query output:      [B, 2, N, D] → flatten+linear → [B, T, N, 1] → DeIN
```

### Step 2: Modify `TemporalSelfAttention`

Currently `t_attn_mask` is only applied to temporal self-attention (t_attn). It must also be applied to temporal-content attention (tc_attn):

```python
# In forward(), after tc_attn computation, before softmax:
if t_attn_mask is not None:
    tc_attn = tc_attn.masked_fill(~t_attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), -1e9)
```

### Step 3: Modify `OpenCity/model/Model.py`

Add `ict_mode` parameter to `Traffic_model.forward()`:
- `ict_mode='concat'` → route to `forward_ict_concat`
- `ict_mode='residual'` → route to original `forward_ict` (default)

### Step 4: Modify `OpenCity/lib/ict_data_process.py`

Add `ICTTrainingDataset` class:
- Similar to `ICTTrafficDataset` but designed for training
- Samples K demos from the same dataset (ensures spatial dimension N matches)
- Excludes temporally overlapping windows from demo pool
- Returns `(batch_x, batch_y, demos_x, demos_y)` with `S=1`

Add `define_ict_training_dataloader(args)` function.

### Step 5: Modify `OpenCity/model/BasicTrainer.py`

Add `multi_train_ict_concat()` training method:
- Forward: `model(inputs, targets, dataset, demos_x, demos_y, ict_mode='concat')`
- Loss: **weighted average of query + all demo prediction losses** (paper's key point: compute loss on ALL examples)
  ```
  loss = (loss_query + K * mean(loss_demo_k)) / (K + 1)
  ```
- Gradient clipping + learning rate scheduling

Add `train_ict_concat()` outer loop: epoch loop + validation + early stopping + best model saving

### Step 6: Modify `OpenCity/model/Run.py`

Add `ict_pretrain` mode:
1. Load pretrained weights
2. Separator parameters randomly initialized (all other params inherited)
3. Set lower learning rate (1e-4 or 5e-5)
4. Create ICT training dataloaders
5. Run continued pretraining
6. Save best model

### Step 7: Modify `OpenCity/lib/Params_pretrain.py`

New arguments:
```python
args.add_argument('-ict_mode', default='residual', type=str, choices=['residual', 'concat', 'learned'])
args.add_argument('-ict_num_demos_train', default=10, type=int)       # K during training
args.add_argument('-ict_pretrain_epochs', default=10, type=int)
args.add_argument('-ict_pretrain_lr', default=1e-4, type=float)
```

## 5. Memory Analysis

| K  | T_total | Attention size (per N, H) | Relative to baseline (2 patches) |
|----|---------|---------------------------|----------------------------------|
| 2  | 8       | 64                        | 16x                              |
| 5  | 17      | 289                       | 72x                              |
| 10 | 32      | 1024                      | 256x                             |
| 20 | 62      | 3844                      | 961x                             |

**Recommendations**:
- Use gradient checkpointing to reduce memory
- Reduce batch_size proportionally (B=4 or B=8)
- Use bfloat16 mixed precision
- Development: K=2-5; production: K=10-20
- **Not recommended for CPU-only training** (already compute-heavy at K=2-3)

## 6. Files Affected

| File | Change Type | Scope |
|------|-------------|-------|
| `OpenCity/model/OpenCity/OpenCity.py` | Edit | +3 methods, +3 parameters |
| `OpenCity/model/Model.py` | Edit | Add `ict_mode` to forward |
| `OpenCity/model/BasicTrainer.py` | Edit | +2 methods |
| `OpenCity/lib/ict_data_process.py` | Edit | +1 class, +1 function |
| `OpenCity/model/Run.py` | Edit | +1 mode |
| `OpenCity/lib/Params_pretrain.py` | Edit | +4 arguments |

## 7. Expected Results

The paper reports 7-25% improvement on TimesFM. For OpenCity as an encoder-based model, the effectiveness of cross-example attention depends on:
- Whether the model can extract useful spatio-temporal patterns from demos
- The value of K (larger K = richer context, but more compute)
- Adequacy of continued pretraining

Expected: should outperform the current residual correction, but likely less dramatic than gains reported on the paper's decoder-only architecture.
