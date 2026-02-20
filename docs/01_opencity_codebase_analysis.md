# OpenCity Codebase Analysis

## 1. Overview

**OpenCity** is a spatio-temporal foundation model for urban traffic forecasting. It is pretrained on diverse multi-city traffic datasets and can generalize to unseen datasets via zero-shot inference or efficient fine-tuning (updating only the prediction head). The model adopts a **Patch-based Transformer** architecture augmented with **Graph Convolutional Networks (GCN)** for spatial mixing, drawing design inspiration from LLaMA (RMSNorm, SwiGLU FFN) and PatchTST (patch tokenization).

**Key Features:**
- Multi-dataset joint pretraining on heterogeneous traffic data (varying node counts, temporal intervals, city topologies)
- Instance Normalization (RevIN-style) for distribution shift across datasets
- Temporal-Context Cross-Attention conditioning on future temporal features (known at inference)
- Laplacian Positional Encoding for spatial identity
- Variable-resolution support: 5-min, 10-min, and 30-min interval datasets unified via adaptive patch embedding

---

## 2. Architecture Design

### 2.1 High-Level Pipeline

```
main.py → Run.py → {config parsing, data loading, model init, mode dispatch} → BasicTrainer
```

| Component | File | Responsibility |
|---|---|---|
| Entry point | `main.py` | `chdir` into `model/`, spawns `Run.py` |
| Pipeline orchestrator | `model/Run.py` | Config, data, model, optimizer, training mode |
| Model wrapper | `model/Model.py` | `Traffic_model` dispatches to selected backbone |
| OpenCity model | `model/OpenCity/OpenCity.py` | Core Transformer+GCN architecture |
| Model config | `model/OpenCity/args.py` | Graph loading, architecture hyperparameters |
| Training loop | `model/BasicTrainer.py` | `Trainer` class with train/val/test |
| Data pipeline | `lib/data_process.py` | Dataset loading, splitting, normalization, batching |
| Config files | `conf/general_conf/pretrain.conf` | Global training settings |
| | `conf/general_conf/global_baselines.conf` | Model-specific training overrides |
| | `conf/OpenCity/OpenCity.conf` | Architecture hyperparameters |

### 2.2 Mode Dispatch (Run.py)

| Mode | Behavior |
|------|----------|
| `pretrain` | Uses pretrain.conf directly (val_ratio=0, test_ratio=0). Joint training on multiple datasets. Saves model periodically. |
| `ori` | Standard supervised train-from-scratch. Predictor params override pretrain params. Train/val/test split with early stopping. |
| `eval` | **Fast Adaptation**: Loads pretrained checkpoint, **freezes all parameters except `model.predictor.linear`** (the prediction head), then fine-tunes for a few epochs. |
| `test` | Pure inference: loads checkpoint, runs `trainer.test()` only. |

---

## 3. Model Architecture — `OpenCity` Class

### 3.1 Component Overview

| Component | Class | Purpose |
|-----------|-------|---------|
| `PatchEmbedding_flow` | Patch tokenizer | Unfolds flow values into non-overlapping patches (size=12, stride=12), projects via `Linear(12, embed_dim)` + sinusoidal positional encoding |
| `PatchEmbedding_time` | Temporal context encoder | Unfolds temporal features into patches, uses learned embeddings for day-of-day (1441 bins) and day-of-week (8 bins), produces separate history (TH) and prediction (TP) embeddings |
| `LaplacianPE` | Spatial encoding | Projects top-8 Laplacian eigenvectors to `embed_dim`, added to temporal embeddings |
| `STEncoderBlock` × `enc_depth` | Core block | `TemporalSelfAttention` → `FeedForward` with pre-norm (LlamaRMSNorm) and DropPath |
| `TemporalSelfAttention` | Attention layer | Two-stage: (1) TC cross-attention (Q=TP, K=TH, V=enc), (2) temporal self-attention, then GCN for spatial mixing |
| `GCN` | Spatial mixer | Single-layer GCN with residual connection (α=0.05) |
| `FeedForward` | MLP | SwiGLU-style FFN: `w2(silu(w1(x)) * w3(x))` |
| Prediction head | `nn.Linear(24*skip_dim, output_window)` | Flattens all patches per node → projects to full output horizon |

### 3.2 Forward Pass — Shape Flow

```
Input:  [B, 288, N, 3]  (flow, day_of_time, day_of_week)
Labels: [B, 288, N, 3]  (same format for future window)
```

| Step | Operation | Output Shape |
|------|-----------|-------------|
| 1 | Temporal context encoding: `patch_embedding_time(cat([TCH, TCP], dim=-1))` | `TH`, `TP`: [B, 24, N, 256] |
| 2 | Spatial PE: `spatial_embedding(lap_mx)` + broadcast | [B, 24, N, 256] added to TH and TP |
| 3 | Instance Norm: subtract mean, divide by stdev (per-sample, per-node) | `x_in`: [B, 288, N, 1] |
| 4 | Patch embed flow: `patch_embedding_flow(x_in)` | `enc`: [B, 24, N, 256] |
| 5 | Encoder blocks × 3: `encoder_block(enc, enc, enc, TH, TP, adj, ...)` | `enc`: [B, 24, N, 256] |
| 6 | Flatten: `enc.permute(0,2,3,1) → flatten` | [B, N, 6144] |
| 7 | Linear head: `linear(skip)` | [B, N, 288] |
| 8 | Reshape + De-IN: `* stdev + means` | **Output**: [B, 288, N, 1] |

### 3.3 Temporal-Context Cross-Attention (TC Attention)

This is the most distinctive component. It performs **cross-attention** where:

- **Query (Q)**: derived from `TP` — the **prediction** (future) temporal embeddings — "what time am I predicting?"
- **Key (K)**: derived from `TH` — the **history** temporal embeddings — "what time does this data come from?"
- **Value (V)**: derived from `x_q` — the **current encoder hidden representation** (flow data)

```python
tc_q = self.tc_q_conv(TP).transpose(1, 2)   # [B, N, 24, D]
tc_k = self.tc_k_conv(TH).transpose(1, 2)   # [B, N, 24, D]
tc_v = self.tc_v_conv(x_q).transpose(1, 2)  # [B, N, 24, D]
# Multi-head reshape → [B, N, heads, 24, head_dim]
tc_attn = (tc_q @ tc_k.T) * scale           # [B, N, heads, 24, 24]
tc_x = tc_attn @ tc_v                        # [B, N, heads, 24, head_dim]
tc_x = norm(tc_x + x_q)                     # residual connection
```

**Key insight**: The future temporal features (day-of-time, day-of-week) are **known at inference time** (you always know what time tomorrow will be). This is NOT data leakage — only temporal metadata (not flow values) is used from labels.

After TC attention, **standard temporal self-attention** operates on the TC output, followed by **GCN** for spatial mixing.

### 3.4 Variable Resolution Support

The model always produces **24 patches** per window regardless of the dataset's temporal interval:

| Interval | Input steps | Unfold params | Patches | Padding |
|----------|-------------|---------------|---------|---------|
| 5 min | 288 | size=12, step=12 | 24 | None |
| 10 min | 144 | size=6, step=6 | 24 | Pad to 12 |
| 30 min | 48 | size=2, step=2 | 24 | Pad to 12 |

This is handled by `gap = self.his // x.shape[-1]` in `PatchEmbedding_flow`.

---

## 4. Data Pipeline

### 4.1 Dataset Loading (`load_st_dataset`)

Each dataset is stored as `.npz` files in `data/{dataset_name}/`. Loading adds temporal features:

```python
data = np.concatenate([raw_flow, day_of_time, day_of_week], axis=-1)
# Final shape: [T_total, N, 3]
#   channel 0: raw traffic values
#   channel 1: minute-of-day (integer, 5–1440 in steps of interval)
#   channel 2: day-of-week (integer, 1–7)
```

**Pretrain mode data protection**: When `mode='pretrain'`, some datasets load reduced date ranges to avoid test data leakage (e.g., PEMS_BAY loads only Jan-Feb 2017, CAD* loads first 60 days).

### 4.2 Internal Batching (`TrafficDataset`)

The dataset class performs its own batching internally:

```python
class TrafficDataset(Dataset):
    def __init__(self, data, batch_size, input_window, output_window):
        # Create sliding windows
        self.windows = [(data[i:i+iw], data[i+iw:i+iw+ow]) for i in range(len(data)-iw-ow+1)]
        # Shuffle + drop last incomplete batch (training)
        # Group into batches
        self.batches = [self.windows[i:i+bs] for i in range(0, len(windows), bs)]

    def __getitem__(self, idx):
        # Returns a FULL BATCH, not a single sample
        return (torch.stack(batch_x), torch.stack(batch_y))
        # Shape: [batch_size, T, N, F] for each
```

The outer `DataLoader` uses `batch_size=1` (adding a leading dim that gets `squeeze(0)`'d).

### 4.3 Multi-Dataset Training

Multiple datasets are concatenated via `ConcatDataset`. At training time, the dataset a batch belongs to is identified by **matching the number of nodes**: `get_key_from_value(num_nodes_dict, inputs.shape[2])`.

Per-dataset resources:
- `scaler_dict[dataset]` — StandardScaler for normalization/denormalization
- `adj_mx_dict[dataset]` — Normalized adjacency matrix for GCN
- `lap_mx_dict[dataset]` — Laplacian eigenvectors for spatial PE
- `geo_mask_dict[dataset]` — Geo-masking based on hop distance

---

## 5. Training Details

### 5.1 Loss Function

Default: `mask_mae` — MAE with masking (ignores near-zero values):
```python
mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=0.001)
```

### 5.2 Optimizer & Scheduler

- **Adam** with `lr_init=0.001`
- Optional `MultiStepLR` scheduler (per-step, not per-epoch)
- Gradient clipping: `max_grad_norm=5`

### 5.3 Fast Adaptation (eval mode)

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False
# Unfreeze only prediction head
for param in model.predictor.linear.parameters():
    param.requires_grad = True
# Train for 3 epochs
trainer.multi_train()
```

### 5.4 Test Procedure

```python
model.eval()
with torch.no_grad():
    for inputs, targets in test_dataloader:
        output = model(inputs, targets, select_dataset)
        # targets passed but model only reads temporal features (lbls[..., output_dim:])
        metrics = All_Metrics(output, targets[..., :output_dim], ...)
```

---

## 6. Configuration System

**Two-tier config with CLI override:**

### Tier 1: `pretrain.conf` (global)
Parsed by `Params_pretrain.parse_args()` with single-hyphen flags (`-mode`, `-model`, `-batch_size`):

| Key Parameters | Default |
|---|---|
| `dataset_use` | List of dataset names |
| `his`, `pred` | 288, 288 (input/output window) |
| `val_ratio`, `test_ratio` | 0.1, 0.4 |
| `loss_func` | `mask_mae` |
| `batch_size` | 32 |
| `epochs` | 10 |
| `lr_init` | 0.001 |
| `load_pretrain_path` | `OpenCity_plus.pth` |

### Tier 2: `global_baselines.conf` + model-specific conf
Parsed by `Params_predictor.get_predictor_params()` with double-hyphen flags (`--embed_dim`, `--enc_depth`):

| OpenCity Variants | embed_dim | skip_dim | enc_depth |
|---|---|---|---|
| Mini | 128 | 128 | 3 |
| Base | 256 | 256 | 3 |
| Plus | 512 | 512 | 6 |

**Override rule**: In non-pretrain modes, predictor params override pretrain params for same-named attributes.

---

## 7. Pretrained Model Weights

Available at `model_weights/OpenCity/`:
- `OpenCity-mini.pth`, `OpenCity-mini2.0.pth`
- `OpenCity-base.pth`
- `OpenCity-plus.pth`, `OpenCity-plus2.0.pth`

---

## 8. Supported Datasets

| Dataset | Nodes | Interval | Type |
|---------|-------|----------|------|
| PEMS04 | 307 | 5 min | Highway sensor |
| PEMS08 | 170 | 5 min | Highway sensor |
| PEMS07M | 228 | 5 min | Highway sensor |
| PEMS_BAY | 325 | 5 min | Highway sensor |
| METR_LA | 207 | 5 min | Highway sensor |
| NYC_TAXI | 263 | 30 min | Taxi demand |
| CHI_TAXI | 77 | 30 min | Taxi demand |
| NYC_BIKE-3 | 540 | 30 min | Bike demand |
| CD_DIDI | 524 | 10 min | Ride-hailing |
| SZ_DIDI | 627 | 10 min | Ride-hailing |
| CAD3-CAD12-* | 211–666 | 5 min | CA highway |
| TrafficHZ/ZZ/CD/JN/SH | 576–896 | 30 min | City traffic index |

**Zero-shot evaluation datasets** (NOT in pretraining): CAD3, CAD5, PEMS07M, TrafficSH, CHI_TAXI, NYC_BIKE-3, CD_DIDI, SZ_DIDI.

---

## 9. Key Design Patterns

1. **Patch-based tokenization**: 288 time steps → 24 non-overlapping patches of 12, enabling efficient Transformer processing
2. **Instance Normalization (RevIN-style)**: Per-sample, per-node mean/std removal handles distribution shift
3. **LLaMA-inspired components**: RMSNorm (LlamaRMSNorm), SwiGLU FFN
4. **Temporal-Context Cross-Attention**: Future temporal features as queries to attend to historical temporal keys, retrieving flow-based values
5. **Graph-augmented Transformer**: GCN after attention for spatial mixing; Laplacian PE for spatial identity
6. **No fixed node count**: Model operates on variable-sized graphs via per-dataset adjacency/Laplacian lookups — enabling multi-dataset pretraining
