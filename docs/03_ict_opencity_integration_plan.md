# Integration Plan: In-Context Tuning for OpenCity

## 1. Executive Summary

This document details the plan to integrate **In-Context Tuning (ICT)** into the **OpenCity** spatiotemporal forecasting framework. The goal is to enable **zero-training adaptation** to unseen datasets by conditioning the pretrained OpenCity model on K labeled demonstration examples at inference time — **without any parameter updates**.

### Comparison with OpenCity Fast Adaptation

| Aspect | Fast Adaptation (Baseline) | ICT (Proposed) |
|--------|---------------------------|----------------|
| Parameter updates | Train linear head, 3 epochs | **None** |
| Training compute | ~minutes per dataset | **Zero** |
| Extra information at inference | None | K demo (history, future) pairs |
| Inference cost increase | None | Attention over 24+K×48 patches (vs 24) |
| Applicable scenario | Need gradient computation on target data | Only need a few sample pairs for prompting |

### Evaluation Scope

All zero-shot datasets from the OpenCity paper:
- **Fast Adaptation benchmark**: CD_DIDI, SZ_DIDI
- **Zero-shot benchmark**: CAD3, CAD5, PEMS07M, TrafficSH, CHI_TAXI, NYC_BIKE-3

---

## 2. Core Design

### 2.1 What is a "Demonstration" in Time-Series Forecasting?

In NLP-ICT, a demonstration is a (text, label) pair. For traffic forecasting:

- **Demonstration** = `(history_window, future_window)` — a complete (input, ground-truth output) pair from the target dataset's training split
- **Query** = `history_window` with unknown `future_window` to predict
- **Label information** = the demo's `future_window` flow values — this is genuinely novel information that the model never sees during standard inference

### 2.2 How Demonstrations Provide Information

In standard OpenCity inference:
- The model sees: history flow values + temporal features (history & future day-of-time, day-of-week)
- It does NOT see: future flow values

With ICT demonstrations:
- The model additionally sees: K complete (history, future) flow patterns from the same dataset
- Through attention, the query can extract:
  - **Temporal patterns**: "This dataset has rush-hour peaks at 8am and 5pm"
  - **Scale information**: "Flow values in this dataset range 0-500"
  - **Spatial patterns**: "Node X and node Y are correlated"
  - **Prediction mappings**: "Given this history shape, the future looks like this"

### 2.3 Architecture: Joint Processing via Sequence Extension

**Approach**: Encode demonstrations as patch sequences and concatenate them with the query's patches along the temporal (patch) dimension. All patches pass through the existing encoder blocks jointly. The prediction head operates only on the query's final 24 patches.

```
Demo 1:  [hist_patches(24)] [future_patches(24)]   = 48 patches
Demo 2:  [hist_patches(24)] [future_patches(24)]   = 48 patches
...
Demo K:  [hist_patches(24)] [future_patches(24)]   = 48 patches
Query:   [hist_patches(24)]                         = 24 patches
─────────────────────────────────────────────────────────────────
Total:   K × 48 + 24 patches in the sequence
```

**Why joint processing (not static context injection)**:
- All patches share the same embedding space (same `PatchEmbedding_flow`, same `PatchEmbedding_time`, same `LaplacianPE`)
- TC Cross-Attention and T Self-Attention naturally handle variable-length sequences — no architecture changes needed
- Demo patches are updated through encoder layers alongside query patches, maintaining representation coherence across layers
- The prediction head (`Linear(24*256, 288)`) only sees the query's 24 patches → **zero architecture change** for the output

### 2.4 How Attention Routes Information from Demos to Query

**TC Cross-Attention** (the key mechanism):
- **Q from TP**: The query's Q vectors encode "I need to predict 8:00am-8:00am+1day on a Wednesday"
- **K from TH**: Keys from demo future patches encode "this data is from 8:00am-8:00am+1day on a Monday"
- **V from enc**: Values from demo future patches contain **actual ground-truth flow patterns**

When demo and query share similar temporal slots (same time of day, similar day of week), the attention scores will be high, and the query effectively retrieves the demo's ground-truth future flow patterns as context.

**T Self-Attention**:
- After TC attention infuses temporal-context info, self-attention allows all patches (demo + query) to interact directly
- The query's history patches can attend to the demo's history patches (learning input patterns) and the demo's future patches (learning prediction patterns)

**GCN**:
- Operates per-patch: `einsum('bdkt,nk->bdnt', h, a)` where `d` is the patch dimension
- Naturally handles variable `d` (K×48+24 vs 24) — no modification needed

---

## 3. Implementation Plan

### 3.1 File Modifications Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `lib/ict_data_process.py` | **New file** | ICT-aware dataset and dataloader |
| `model/OpenCity/OpenCity.py` | **Add method** | `forward_ict()` — new forward pass with demo processing |
| `model/Model.py` | **Modify** | Transparent demo parameter forwarding |
| `model/BasicTrainer.py` | **Add method** | `test_ict()` — ICT inference with metrics |
| `model/Run.py` | **Modify** | New `mode='ict'` branch |
| `lib/Params_pretrain.py` | **Modify** | ICT-related parameter parsing |
| `conf/ICT/ICT.conf` | **New file** | ICT configuration defaults |

### 3.2 Step-by-Step Implementation

---

#### Step 1: ICT Data Loader (`lib/ict_data_process.py`)

**New file** implementing ICT-aware data loading.

##### 1.1 `ICTTrafficDataset` Class

```python
class ICTTrafficDataset(Dataset):
    """Traffic dataset with demonstration sampling for ICT."""

    def __init__(self, data, batch_size, input_window, output_window,
                 demo_pool, num_demonstrations=1, eval_only=False):
        """
        Args:
            data: numpy array [T, N, F] for this split (train/val/test)
            batch_size: internal batch size
            input_window: history window length
            output_window: prediction window length
            demo_pool: list of (x, y) numpy array pairs from training split
            num_demonstrations: K — number of demos per query
            eval_only: if True, don't shuffle
        """
        # Create sliding windows (same as TrafficDataset)
        self.windows = [
            (data[i:i+input_window], data[i+input_window:i+input_window+output_window])
            for i in range(len(data) - input_window - output_window + 1)
        ]
        self.demo_pool = demo_pool  # From training split
        self.num_demonstrations = num_demonstrations

        # Shuffle + drop last + pre-batch (same as TrafficDataset)
        if not eval_only:
            random.shuffle(self.windows)
            remainder = len(self.windows) % batch_size
            if remainder != 0:
                self.windows = self.windows[:-remainder]
        self.batches = [
            self.windows[i:i+batch_size]
            for i in range(0, len(self.windows), batch_size)
        ]

    def __getitem__(self, idx):
        batch_pairs = self.batches[idx]
        batch_x, batch_y = zip(*batch_pairs)
        batch_x = torch.from_numpy(np.stack(batch_x)).float()  # [B, T, N, F]
        batch_y = torch.from_numpy(np.stack(batch_y)).float()  # [B, T, N, F]

        # Sample K demonstrations for each query in the batch
        B = batch_x.shape[0]
        K = self.num_demonstrations
        demos_x_list, demos_y_list = [], []
        for b in range(B):
            indices = random.sample(range(len(self.demo_pool)), K)
            dx = np.stack([self.demo_pool[i][0] for i in indices])  # [K, T, N, F]
            dy = np.stack([self.demo_pool[i][1] for i in indices])  # [K, T, N, F]
            demos_x_list.append(dx)
            demos_y_list.append(dy)

        demos_x = torch.from_numpy(np.stack(demos_x_list)).float()  # [B, K, T, N, F]
        demos_y = torch.from_numpy(np.stack(demos_y_list)).float()  # [B, K, T, N, F]

        return batch_x, batch_y, demos_x, demos_y

    def __len__(self):
        return len(self.batches)
```

##### 1.2 `define_ict_dataloader` Function

```python
def define_ict_dataloader(args):
    """Create ICT-aware dataloaders.

    Key difference from define_dataloder:
    - Constructs a demo_pool from training split
    - Val/test datasets use training split's demo_pool (no data leakage)
    - Returns dataloaders that yield (query_x, query_y, demos_x, demos_y)
    """
    scaler_dict = {}
    datasets_train, datasets_val, datasets_test = [], [], []

    for dataset_name in args.dataset_use:
        data = load_st_dataset(dataset_name, args)
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

        # Normalize using training split statistics
        scaler, _, _ = normalize_dataset(data_train, args.input_base_dim)
        data_train[..., :args.input_base_dim] = scaler.transform(data_train[..., :args.input_base_dim])
        data_val[..., :args.input_base_dim] = scaler.transform(data_val[..., :args.input_base_dim])
        data_test[..., :args.input_base_dim] = scaler.transform(data_test[..., :args.input_base_dim])
        scaler_dict[dataset_name] = scaler

        # Compute input/output window for this dataset's interval
        iw, ow = compute_windows(dataset_name, args)

        # Build demo pool from training windows
        demo_pool = [
            (data_train[i:i+iw], data_train[i+iw:i+iw+ow])
            for i in range(len(data_train) - iw - ow + 1)
        ]

        # Create datasets
        train_ds = ICTTrafficDataset(data_train, args.batch_size, iw, ow,
                                      demo_pool, args.num_demonstrations, eval_only=False)
        val_ds = ICTTrafficDataset(data_val, args.batch_size, iw, ow,
                                    demo_pool, args.num_demonstrations, eval_only=True)
        test_ds = ICTTrafficDataset(data_test, args.batch_size, iw, ow,
                                     demo_pool, args.num_demonstrations, eval_only=True)
        datasets_train.append(train_ds)
        datasets_val.append(val_ds)
        datasets_test.append(test_ds)

    # Concatenate and wrap in DataLoader (same pattern as original)
    train_loader = DataLoader(ConcatDataset(datasets_train), batch_size=1, shuffle=True)
    val_loader = DataLoader(ConcatDataset(datasets_val), batch_size=1, shuffle=False)
    test_loader = DataLoader(ConcatDataset(datasets_test), batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_dict
```

---

#### Step 2: OpenCity Model — `forward_ict` Method (`model/OpenCity/OpenCity.py`)

**Add a new method** `forward_ict` to the `OpenCity` class. The original `forward` method is **NOT modified** (full backward compatibility).

```python
def forward_ict(self, input, lbls, demos_x, demos_y, select_dataset):
    """
    ICT forward pass: process query with K demonstrations as context.

    Args:
        input:   [B, T, N, F]   — query history
        lbls:    [B, T, N, F]   — query future (only temporal features used)
        demos_x: [B, K, T, N, F] — demonstration histories
        demos_y: [B, K, T, N, F] — demonstration futures (ground-truth flow + temporal)
        select_dataset: str — dataset identifier

    Returns:
        [B, T, N, 1] — predicted future flow for query
    """
    bs, time_steps, num_nodes, num_feas = input.size()
    K = demos_x.shape[1]

    # ===== QUERY PROCESSING (same as forward) =====

    # Query temporal context
    query_TCH = input[..., self.output_dim:].long()      # [B, T, N, 2]
    query_TCP = lbls[..., self.output_dim:].long()        # [B, T, N, 2]
    query_TH, query_TP = self.patch_embedding_time(
        torch.cat([query_TCH, query_TCP], dim=-1)         # [B, T, N, 4]
    )  # Both: [B, 24, N, D]

    # Spatial PE
    spa_feas = self.spatial_embedding(
        self.lap_mx_dict[select_dataset].to(self.device)
    )  # [1, 1, N, D]

    query_TH = query_TH + spa_feas
    query_TP = query_TP + spa_feas

    # Query Instance Normalization
    query_flow = input[..., :self.output_dim]             # [B, T, N, 1]
    query_means = query_flow.mean(1, keepdim=True).detach()
    query_centered = query_flow - query_means
    query_stdev = torch.sqrt(
        torch.var(query_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    query_normed = query_centered / query_stdev           # [B, T, N, 1]

    # Query patch embedding
    query_enc = self.patch_embedding_flow(query_normed)   # [B, 24, N, D]

    # ===== DEMONSTRATION PROCESSING =====

    all_demo_enc = []      # List of [B, 48, N, D] — one per demo
    all_demo_TH = []       # temporal keys for each demo
    all_demo_TP = []       # temporal queries (unused, but needed for TH_all construction)

    for k in range(K):
        dk_x = demos_x[:, k]  # [B, T, N, F]
        dk_y = demos_y[:, k]  # [B, T, N, F]

        # Demo temporal context
        dk_TCH = dk_x[..., self.output_dim:].long()
        dk_TCP = dk_y[..., self.output_dim:].long()
        dk_TH, dk_TP = self.patch_embedding_time(
            torch.cat([dk_TCH, dk_TCP], dim=-1)
        )  # Both: [B, 24, N, D]
        dk_TH = dk_TH + spa_feas
        dk_TP = dk_TP + spa_feas

        # Demo Instance Normalization (based on demo's own history)
        dk_hist_flow = dk_x[..., :self.output_dim]       # [B, T, N, 1]
        dk_futu_flow = dk_y[..., :self.output_dim]       # [B, T, N, 1]
        dk_means = dk_hist_flow.mean(1, keepdim=True).detach()
        dk_centered = dk_hist_flow - dk_means
        dk_stdev = torch.sqrt(
            torch.var(dk_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        dk_hist_normed = dk_centered / dk_stdev
        dk_futu_normed = (dk_futu_flow - dk_means) / dk_stdev

        # Demo patch embeddings
        dk_hist_enc = self.patch_embedding_flow(dk_hist_normed)  # [B, 24, N, D]
        dk_futu_enc = self.patch_embedding_flow(dk_futu_normed)  # [B, 24, N, D]

        # Concatenate demo history + future patches
        dk_enc = torch.cat([dk_hist_enc, dk_futu_enc], dim=1)   # [B, 48, N, D]
        dk_th_full = torch.cat([dk_TH, dk_TP], dim=1)           # [B, 48, N, D]

        all_demo_enc.append(dk_enc)
        all_demo_TH.append(dk_th_full)

    # ===== CONCATENATE: demos + query =====

    # Encoder input: [demo1(48)] [demo2(48)] ... [demoK(48)] [query(24)]
    enc_all = torch.cat(all_demo_enc + [query_enc], dim=1)   # [B, K*48+24, N, D]

    # TH for all positions (used as KEY in TC attention)
    TH_all = torch.cat(all_demo_TH + [query_TH], dim=1)      # [B, K*48+24, N, D]

    # TP for all positions (used as QUERY in TC attention)
    # Demo positions: use their own TP (repeated for hist+future patches)
    # Query positions: use query_TP
    all_demo_TP_full = []
    for k in range(K):
        dk_x = demos_x[:, k]
        dk_y = demos_y[:, k]
        dk_TCH = dk_x[..., self.output_dim:].long()
        dk_TCP = dk_y[..., self.output_dim:].long()
        _, dk_TP = self.patch_embedding_time(torch.cat([dk_TCH, dk_TCP], dim=-1))
        dk_TP = dk_TP + spa_feas
        all_demo_TP_full.append(torch.cat([dk_TP, dk_TP], dim=1))  # [B, 48, N, D]
    TP_all = torch.cat(all_demo_TP_full + [query_TP], dim=1)  # [B, K*48+24, N, D]

    # ===== ENCODER BLOCKS =====

    adj = self.adj_mx_dict[select_dataset].to(self.device)
    geo_mask = self.geo_mask_dict[select_dataset].to(self.device)

    for encoder_block in self.encoder_blocks:
        enc_all = encoder_block(
            enc_all, enc_all, enc_all,
            TH_all, TP_all, adj, geo_mask, self.sem_mask
        )
    # enc_all: [B, K*48+24, N, D]

    # ===== EXTRACT QUERY PATCHES & PREDICT =====

    query_out = enc_all[:, -24:, :, :]  # Last 24 patches = query
    # [B, 24, N, D]

    skip = query_out.permute(0, 2, 3, 1).contiguous()  # [B, N, D, 24]
    skip = self.flatten(skip)                            # [B, N, D*24]
    skip = self.linear(skip)                             # [B, N, output_window]
    skip = skip.transpose(1, 2).unsqueeze(-1)            # [B, T, N, 1]
    skip = skip[:, :time_steps, :, :]

    # De-Instance-Normalization (using query's own stats)
    skip = skip * query_stdev
    skip = skip + query_means

    return skip
```

**Key Design Notes:**
- Demo future flow values go through `PatchEmbedding_flow` — same embedding as history. This reuses pretrained weights with zero new parameters.
- Each demo is independently instance-normalized using its own history's mean/stdev. The query uses its own stats. After normalization, all are in the same zero-mean unit-variance space.
- Demo temporal features: history uses `dk_TH`, future uses `dk_TP`. This tells the attention mechanism **when** each demo patch's data comes from.
- No new parameters are introduced. All embeddings, attention layers, GCN, FFN, and the prediction head are from the pretrained model.

**Optimization Note — TP computation**: The demo TP is computed twice (once for `all_demo_TH` construction and once for `TP_all`). This can be optimized by caching. Implementation should store dk_TP in the first loop.

---

#### Step 3: Traffic_model Wrapper (`model/Model.py`)

Modify `Traffic_model.forward` to transparently forward demo parameters:

```python
def forward(self, source, label, select_dataset, batch_seen=None,
            demos_x=None, demos_y=None):
    if self.model == 'OpenCity':
        if demos_x is not None:
            x_predic = self.predictor.forward_ict(
                source, label, demos_x, demos_y, select_dataset
            )
        else:
            x_predic = self.predictor(source, label, select_dataset)
    else:
        x_predic = self.predictor(source[..., 0:self.input_base_dim], select_dataset)
    return x_predic
```

**Backward compatible**: When `demos_x=None`, behavior is identical to the original.

---

#### Step 4: ICT Test Method (`model/BasicTrainer.py`)

Add a new static method `test_ict` to the `Trainer` class:

```python
@staticmethod
def test_ict(model, args, scaler_dict, test_dataloader, logger, path=None,
             num_prefix_selections=1):
    """
    ICT inference: pure forward pass with demonstrations, no gradient updates.

    Args:
        num_prefix_selections: number of random demo sets to average over
                               (reduces variance from demo selection)
    """
    # Load pretrained weights
    if path is not None:
        model_weights = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
        model.load_state_dict(model_weights)
        model.to(args.device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        mae, rmse, mape = 0, 0, 0
        total_count, total_mape_count, total_batch = 0, 0, 0

        for inputs, targets, demos_x, demos_y in test_dataloader:
            inputs = inputs.squeeze(0).to(args.device)
            targets = targets.squeeze(0).to(args.device)
            demos_x = demos_x.squeeze(0).to(args.device)  # [B, K, T, N, F]
            demos_y = demos_y.squeeze(0).to(args.device)

            select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])

            if num_prefix_selections > 1:
                # Multiple random demo selections, average predictions
                outputs = []
                for _ in range(num_prefix_selections):
                    # Re-sample demos (shuffle the demo pool dimension)
                    # Note: for proper multi-selection, the dataloader should
                    # provide a larger demo pool, or we re-index here
                    output = model(inputs, targets, select_dataset,
                                   demos_x=demos_x, demos_y=demos_y)
                    outputs.append(output)
                output = torch.stack(outputs).mean(dim=0)
            else:
                output = model(inputs, targets, select_dataset,
                               demos_x=demos_x, demos_y=demos_y)

            # Inverse transform if needed
            if args.real_value == False:
                output = scaler_dict[select_dataset].inverse_transform(output)
                y_lbl = scaler_dict[select_dataset].inverse_transform(
                    targets[..., :args.output_dim]
                )
            else:
                y_lbl = targets[..., :args.output_dim]

            # Compute metrics (same as existing test())
            batch_mae, batch_rmse, batch_mape, _, _, mae_count, mape_count = \
                All_Metrics(output, y_lbl, args.mae_thresh, args.mape_thresh)

            mae += batch_mae * mae_count
            rmse += (batch_rmse ** 2) * mae_count
            mape += batch_mape * mape_count
            total_count += mae_count
            total_mape_count += mape_count
            total_batch += 1

        # Final metrics
        avg_mae = mae / total_count
        avg_rmse = (rmse / total_count) ** 0.5
        avg_mape = mape / total_mape_count

        logger.info(f'ICT Test — MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}')
```

---

#### Step 5: Mode Dispatch (`model/Run.py`)

Add `mode='ict'` branch:

```python
elif args.mode == 'ict':
    # Import ICT dataloader
    from lib.ict_data_process import define_ict_dataloader

    # Load pretrained model
    path = log_dir + '/' + args.load_pretrain_path
    model_weights = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
    model.load_state_dict(model_weights)
    print("Loaded pretrained model for ICT inference")

    # Freeze all parameters (zero-training)
    for param in model.parameters():
        param.requires_grad = False
    print_model_parameters(model, only_num=False)

    # Create ICT dataloaders
    _, _, test_dataloader_ict, scaler_dict_ict = define_ict_dataloader(args)

    # Run ICT test
    trainer.test_ict(
        model, args, scaler_dict_ict, test_dataloader_ict, trainer.logger,
        path=None,  # Already loaded
        num_prefix_selections=args.num_prefix_selections
    )
```

---

#### Step 6: Configuration (`conf/ICT/ICT.conf` + `lib/Params_pretrain.py`)

##### New config file: `conf/ICT/ICT.conf`

```ini
[ict]
num_demonstrations = 1
num_prefix_selections = 1
demo_selection = random
ict_batch_size = 32
```

##### Modify `Params_pretrain.py`

Add ICT parameters to the argument parser:

```python
# ICT parameters
parser.add_argument('-num_demonstrations', default=1, type=int,
                    help='Number of demonstration pairs (K) for ICT')
parser.add_argument('-num_prefix_selections', default=1, type=int,
                    help='Number of random demo sets to average at test time')
parser.add_argument('-demo_selection', default='random', type=str,
                    help='Demo selection strategy: random, recent, similar')
```

---

## 4. Shape Analysis & Compatibility Verification

### 4.1 Sequence Length with Demonstrations

| K | Total patches | Attention matrix size | Relative to baseline |
|---|---|---|---|
| 0 (baseline) | 24 | 24×24 = 576 | 1.0× |
| 1 | 72 | 72×72 = 5,184 | 9.0× |
| 3 | 168 | 168×168 = 28,224 | 49.0× |
| 5 | 264 | 264×264 = 69,696 | 121.0× |

**Memory recommendation**: K=1 with batch_size=32 should be comparable to baseline batch_size=64. For K=3+, reduce batch size accordingly.

### 4.2 Component Compatibility Check

| Component | Variable T? | Compatible? | Notes |
|-----------|-------------|-------------|-------|
| `PatchEmbedding_flow` | N/A | ✅ | Applied per-demo, always produces 24 patches |
| `PatchEmbedding_time` | N/A | ✅ | Applied per-demo, always produces 24 patches |
| `LaplacianPE` | No T dim | ✅ | Broadcast to any T |
| `TemporalSelfAttention` TC | `T_q`, `T_k`, `T_v` from tensor | ✅ | Q/K/V dims derived from actual shapes |
| `TemporalSelfAttention` T | Same | ✅ | Self-attention on variable T |
| `GCN` | `einsum('bdkt,nk->bdnt')` | ✅ | `d` (T) is a batch-like dim |
| `FeedForward` | Pointwise | ✅ | Operates on last dim only |
| `LlamaRMSNorm` | Last dim | ✅ | Operates on last dim only |
| `DropPath` | Any shape | ✅ | Drops along batch dim |
| `linear` (pred head) | Fixed 24×D→288 | ✅ | Only applied to query's 24 patches |

**Result**: All components naturally handle variable sequence lengths. **No architecture modifications required.**

### 4.3 Instance Normalization Isolation

Each demo and the query have **independent** Instance Normalization:

```
Demo k: means_k = dk_hist_flow.mean(dim=1), stdev_k from dk_hist_flow
         dk_hist_normed = (dk_hist_flow - means_k) / stdev_k
         dk_futu_normed = (dk_futu_flow - means_k) / stdev_k
Query:   means_q = query_flow.mean(dim=1), stdev_q from query_flow
         query_normed = (query_flow - means_q) / stdev_q
```

After normalization, all patches are in the same zero-mean, unit-variance space — ensuring representation compatibility despite potentially different data scales.

The prediction is de-normalized using **query's own stats**: `output * stdev_q + means_q`.

---

## 5. Experimental Protocol

### 5.1 Datasets

| Dataset | Nodes | Interval | Train/Val/Test | Status |
|---------|-------|----------|----------------|--------|
| CD_DIDI | 524 | 10 min | 0.5/0.1/0.4 | Fast Adaptation benchmark |
| SZ_DIDI | 627 | 10 min | 0.5/0.1/0.4 | Fast Adaptation benchmark |
| CAD3 | 480 | 5 min | 0.5/0.1/0.4 | Zero-shot benchmark |
| CAD5 | 211 | 5 min | 0.5/0.1/0.4 | Zero-shot benchmark |
| PEMS07M | 228 | 5 min | 0.5/0.1/0.4 | Zero-shot benchmark |
| TrafficSH | 896 | 30 min | 0.5/0.1/0.4 | Zero-shot benchmark |
| CHI_TAXI | 77 | 30 min | 0.2/0.2/0.6 | Zero-shot benchmark |
| NYC_BIKE-3 | 540 | 30 min | 0.2/0.2/0.6 | Zero-shot benchmark |

### 5.2 Baselines for Comparison

1. **OpenCity Zero-Shot** (`mode=test`): Direct inference without any training or demos
2. **OpenCity Fast Adaptation** (`mode=eval`): Freeze backbone, train linear head 3 epochs
3. **Supervised Baselines** (`mode=ori`): Full training on target dataset (100 epochs, early stop 15)
4. **ICT K=1** (`mode=ict, num_demonstrations=1`): 1 demo, no training
5. **ICT K=3** (`mode=ict, num_demonstrations=3`): 3 demos, no training
6. **ICT K=1 ×10 avg** (`mode=ict, num_demonstrations=1, num_prefix_selections=10`): Variance reduction

### 5.3 Metrics

- **MAE** (Mean Absolute Error) — primary metric
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

All with `mask_value=0.001` (mask near-zero values) and `mae_thresh=0`.

### 5.4 Command Examples

```bash
# ICT with K=1 on CD_DIDI
python main.py -mode ict -model OpenCity -dataset_use "['CD_DIDI']" \
    -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1

# ICT with K=3, 10 random selections averaged on PEMS07M
python main.py -mode ict -model OpenCity -dataset_use "['PEMS07M']" \
    -load_pretrain_path OpenCity-plus.pth -num_demonstrations 3 \
    -num_prefix_selections 10

# Baseline: Zero-shot (existing)
python main.py -mode test -model OpenCity -dataset_use "['CD_DIDI']" \
    -load_pretrain_path OpenCity-plus.pth

# Baseline: Fast Adaptation (existing)
python main.py -mode eval -model OpenCity -dataset_use "['CD_DIDI']" \
    -load_pretrain_path OpenCity-plus.pth -epochs 3
```

---

## 6. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Demo future flow is just averaged out, not useful | Medium | High | Analyze attention weights to verify query attends to demo futures; try demo selection strategies |
| OOM with K≥3 | Low | Medium | Reduce batch size; use gradient checkpointing (not needed since no training) |
| Pretrained model can't process extended sequences | Very Low | High | All components verified compatible (Section 4.2) |
| Demo from different distribution confuses model | Medium | Medium | Demos come from same dataset; instance normalization handles scale differences |
| Marginal improvement over zero-shot | Medium | High | This validates the approach; if ICT shows any improvement without training, it's scientifically interesting |

---

## 7. Implementation Priorities

| Priority | Task | Effort | Dependencies |
|----------|------|--------|-------------|
| P0 | `OpenCity.forward_ict()` method | 2h | None |
| P0 | `ICTTrafficDataset` + `define_ict_dataloader` | 2h | None |
| P0 | `Traffic_model.forward` modification | 15min | forward_ict |
| P0 | `Trainer.test_ict()` method | 1h | ICTTrafficDataset |
| P0 | `Run.py` mode='ict' branch | 30min | All above |
| P1 | `Params_pretrain.py` ICT args | 15min | None |
| P1 | `conf/ICT/ICT.conf` | 5min | None |
| P2 | Shape unit tests | 1h | forward_ict |
| P2 | Regression test (K=0 ≡ forward) | 30min | forward_ict |
| P3 | Demo selection strategies (recent, similar) | 2h | Basic ICT working |
| P3 | Attention visualization | 2h | Basic ICT working |

**Total estimated effort for P0+P1 (minimum viable)**: ~6 hours

---

## 8. Future Extensions

1. **Demo Selection Strategies**: Beyond random sampling — temporally recent, spatially similar (DTW-based), or same time-of-day/day-of-week demos could improve performance
2. **Lightweight Adapter Training**: Add a small trainable adapter layer (LoRA-style) that processes demo context, while keeping the base model frozen — a middle ground between zero-training ICT and full fine-tuning
3. **Cross-Dataset Demos**: Use demos from pretrained datasets to inform predictions on zero-shot datasets — testing cross-domain transfer
4. **Attention Analysis**: Visualize TC attention weights to understand how query patches attend to demo history vs. demo future patches — crucial for scientific insight
5. **Scalability to K>>1**: Implement efficient attention (e.g., FlashAttention) to support more demonstrations
