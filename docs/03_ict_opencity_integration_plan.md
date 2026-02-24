# Integration Plan: In-Context Tuning for OpenCity

## 1. Executive Summary

This document details the plan to integrate **In-Context Tuning (ICT)** into the **OpenCity** spatiotemporal forecasting framework. The goal is to enable **zero-training adaptation** to unseen datasets by conditioning the pretrained OpenCity model on K labeled demonstration examples at inference time — **without any parameter updates**.

### Comparison with OpenCity Fast Adaptation

| Aspect | Fast Adaptation (Baseline) | ICT (Proposed) |
|--------|---------------------------|----------------|
| Parameter updates | Train linear head, 3 epochs | **None** |
| Training compute | ~minutes per dataset | **Zero** |
| Extra information at inference | None | K demo (history, future) pairs |
| Inference cost increase | None | K+1 standard forward passes (vs 1) |
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

### 2.3 Architecture: Residual Correction (v2)

> **Note**: The original plan (v1) used sequence extension — concatenating demo+query patches into a K×48+24 sequence processed jointly through encoder blocks. This was implemented and tested (Experiments 2-4), but **abandoned** because OpenCity was pretrained on fixed 24-patch sequences; extended sequences are OOD and degrade performance. See `docs/05_experiment_log.md` for details.

**Approach**: Process each input independently through the standard `forward()` path (exactly 24 patches, identical to pretraining). Use demo ground-truth futures to estimate and correct the model's systematic error.

```
pred_query   = model.forward(query_hist, query_lbl, dataset)      # standard 24-patch forward
pred_demo_k  = model.forward(demo_k_hist, demo_k_lbl, dataset)    # same standard forward
error_k      = demo_k_ground_truth_flow - pred_demo_k             # model's systematic error on demo k
correction   = mean(error_1, ..., error_K)                        # averaged correction
final        = pred_query + correction                            # corrected prediction
```

**Why residual correction**:
- **Zero OOD**: Every forward call uses exactly 24 patches — identical to pretraining
- **Zero new parameters**: No architecture changes, no new layers
- **fp32 inference**: Same memory as zero-shot per call (no bfloat16 needed)
- **Physically interpretable**: Estimates the model's systematic prediction bias on a similar sample, then corrects it
- **Compute cost**: K+1 sequential forward passes (K=1 → 2× compute; trivially parallelizable)

### 2.4 How Demonstrations Provide Information (Residual Correction)

The key insight: if the model makes similar systematic errors on nearby traffic patterns, then the demo's prediction error is a good estimate of the query's prediction error.

**Error estimation**:
- For each demo k, we have both the ground-truth future `dk_gt` and the model's prediction `pred_dk`
- The residual `dk_gt - pred_dk` captures the model's systematic bias on that sample
- If demo k is from a similar traffic regime as the query, this bias transfers

**Correction averaging**:
- With K>1 demos, averaging residuals reduces noise from individual demo selection
- With S>1 prefix selections, averaging over S independent demo sets further reduces variance

**What the model extracts from demos**:
- **Scale calibration**: If the model consistently under/over-predicts by a factor, the residual corrects it
- **Pattern-specific bias**: Rush-hour prediction errors differ from off-peak — demos from similar time slots provide better corrections
- **Dataset adaptation**: Different datasets have different characteristics; demos from the target dataset capture dataset-specific biases

---

## 3. Implementation Plan

### 3.1 File Modifications Summary

| File | Change Type | Description | Status |
|------|-------------|-------------|--------|
| `conf/general_conf/dataset_splits.conf` | **New file** | Per-dataset split ratios (centralized, all modes) | **Done** |
| `lib/data_process.py` | **Modify** | `define_dataloder()` uses `dataset_splits.conf` for per-dataset splits | **Done** |
| `lib/ict_data_process.py` | **New file** | ICT-aware dataset and dataloader | **Done** |
| `model/OpenCity/OpenCity.py` | **Add method** | `forward_ict()` — residual correction forward pass | **Done** |
| `model/Model.py` | **Modify** | Transparent demo parameter forwarding | **Done** |
| `model/BasicTrainer.py` | **Add method** | `test_ict()` — ICT inference with metrics | **Done** |
| `model/Run.py` | **Modify** | New `mode='ict'` branch | **Done** |
| `lib/Params_pretrain.py` | **Modify** | ICT-related parameter parsing | **Done** |
| `conf/ICT/ICT.conf` | **New file** | ICT configuration defaults | **Done** |

### 3.2 Step-by-Step Implementation

---

#### Step 0 ✅ (Done): Configuration Foundation

> **Completed first** — prerequisite for all modes (`ict`, `test`, `eval`, `ori`, `pretrain`).

##### 0.1 New config file: `conf/general_conf/dataset_splits.conf`

Centralized per-dataset split ratios. **Single entry point for all modes** — edit this file to adjust any dataset's train/val/test split without touching Python code or `pretrain.conf`.

```ini
# ── Default split (fallback for unlisted datasets) ──
[default]
val_ratio = 0.1
test_ratio = 0.4

# ── Zero-shot benchmark (not in pretraining) ──
[CAD3]
val_ratio = 0.1
test_ratio = 0.4

[CAD5]
val_ratio = 0.1
test_ratio = 0.4

[PEMS07M]
val_ratio = 0.1
test_ratio = 0.4

[TrafficSH]
val_ratio = 0.1
test_ratio = 0.4

[CHI_TAXI]
val_ratio = 0.2
test_ratio = 0.6

[NYC_BIKE-3]
val_ratio = 0.2
test_ratio = 0.6

# ── Fast Adaptation benchmark ──
[CD_DIDI]
val_ratio = 0.1
test_ratio = 0.4

[SZ_DIDI]
val_ratio = 0.1
test_ratio = 0.4

# (plus supervised eval and pretraining datasets — see full file)
# CAD4-*, CAD7-*, CAD12-1 all use default (0.1, 0.4)
```

##### 0.2 Modified `lib/data_process.py` — Helper functions & `define_dataloder()`

Added `load_dataset_splits()` and `get_dataset_split()` to `data_process.py`. Modified `define_dataloder()` to use per-dataset splits from `dataset_splits.conf` instead of `args.val_ratio` / `args.test_ratio`.

```python
def load_dataset_splits(conf_path=None):
    """Load per-dataset split ratios from dataset_splits.conf."""
    if conf_path is None:
        conf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'conf', 'general_conf', 'dataset_splits.conf'
        )
    config = configparser.ConfigParser()
    config.read(conf_path)
    default_val = config.getfloat('default', 'val_ratio')
    default_test = config.getfloat('default', 'test_ratio')
    splits = {}
    for section in config.sections():
        if section == 'default':
            continue
        splits[section] = (
            config.getfloat(section, 'val_ratio'),
            config.getfloat(section, 'test_ratio'),
        )
    return splits, (default_val, default_test)


def get_dataset_split(dataset_name, splits, default):
    """Return (val_ratio, test_ratio) for a given dataset."""
    return splits.get(dataset_name, default)
```

In `define_dataloder()`:
```python
splits, default_split = load_dataset_splits()
for dataset_name in args.dataset_use:
    val_ratio, test_ratio = get_dataset_split(dataset_name, splits, default_split)
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    # ... rest unchanged ...
```

> **Backward compatible**: If `dataset_splits.conf` is missing or a dataset isn't listed, falls back to `[default]` section (0.1, 0.4). The `args.val_ratio` / `args.test_ratio` from `pretrain.conf` are no longer used for splitting — they remain in `pretrain.conf` only for reference / legacy compatibility.

##### 0.3 New config file: `conf/ICT/ICT.conf`

```ini
[ict]
num_demonstrations = 1
num_prefix_selections = 1
demo_selection = random
ict_batch_size = 32
```

##### 0.4 Modified `lib/Params_pretrain.py` — ICT parameters

```python
# ICT parameters
args.add_argument('-num_demonstrations', default=1, type=int,
                  help='Number of demonstration pairs (K) for ICT')
args.add_argument('-num_prefix_selections', default=1, type=int,
                  help='Number of independent demo sets (S) to average at test time')
args.add_argument('-demo_selection', default='random', type=str,
                  help='Demo selection strategy: random, recent, similar')
```

---

#### Step 1: ICT Data Loader (`lib/ict_data_process.py`)

**New file** implementing ICT-aware data loading.

##### 1.1 `ICTTrafficDataset` Class

```python
class ICTTrafficDataset(Dataset):
    """Traffic dataset with demonstration sampling for ICT."""

    def __init__(self, data, batch_size, input_window, output_window,
                 demo_pool, num_demonstrations=1, num_prefix_selections=1,
                 eval_only=False):
        """
        Args:
            data: numpy array [T, N, F] for this split (train/val/test)
            batch_size: internal batch size
            input_window: history window length
            output_window: prediction window length
            demo_pool: list of (x, y) numpy array pairs from training split
            num_demonstrations: K — number of demos per query
            num_prefix_selections: S — number of independent demo sets to sample
                                   (for variance reduction at test time)
            eval_only: if True, don't shuffle
        """
        # Create sliding windows (same as TrafficDataset)
        self.windows = [
            (data[i:i+input_window], data[i+input_window:i+input_window+output_window])
            for i in range(len(data) - input_window - output_window + 1)
        ]
        self.demo_pool = demo_pool  # From training split
        self.num_demonstrations = num_demonstrations
        self.num_prefix_selections = num_prefix_selections

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

        # Sample S independent sets of K demonstrations for each query
        B = batch_x.shape[0]
        K = self.num_demonstrations
        S = self.num_prefix_selections
        demos_x_list, demos_y_list = [], []
        for b in range(B):
            sets_x, sets_y = [], []
            for s in range(S):
                indices = random.sample(range(len(self.demo_pool)), K)
                dx = np.stack([self.demo_pool[i][0] for i in indices])  # [K, T, N, F]
                dy = np.stack([self.demo_pool[i][1] for i in indices])  # [K, T, N, F]
                sets_x.append(dx)
                sets_y.append(dy)
            demos_x_list.append(np.stack(sets_x))  # [S, K, T, N, F]
            demos_y_list.append(np.stack(sets_y))  # [S, K, T, N, F]

        demos_x = torch.from_numpy(np.stack(demos_x_list)).float()  # [B, S, K, T, N, F]
        demos_y = torch.from_numpy(np.stack(demos_y_list)).float()  # [B, S, K, T, N, F]

        return batch_x, batch_y, demos_x, demos_y

    def __len__(self):
        return len(self.batches)
```

> **Design note — `num_prefix_selections` (S)**:  When S=1, `demos_x` shape is `[B, 1, K, T, N, F]` — the S dimension is always present.  In `test_ict`, the loop iterates over S, each time passing `demos_x[:, s]` (shape `[B, K, T, N, F]`) to `forward_ict`.  Since the model is in `eval()` mode with no dropout, each selection *must* use **different demo indices** for the averaging to be meaningful.  The sampling happens in `__getitem__` at data loading time, guaranteeing truly independent demo sets.

##### 1.2 `define_ict_dataloader` Function

```python
import configparser


def load_dataset_splits(conf_path='../conf/general_conf/dataset_splits.conf'):
    """Load per-dataset split ratios from conf file.

    Returns:
        splits: dict  {dataset_name: (val_ratio, test_ratio)}
        default: tuple (val_ratio, test_ratio) from [default] section
    """
    config = configparser.ConfigParser()
    config.read(conf_path)
    default_val = config.getfloat('default', 'val_ratio')
    default_test = config.getfloat('default', 'test_ratio')
    splits = {}
    for section in config.sections():
        if section == 'default':
            continue
        splits[section] = (
            config.getfloat(section, 'val_ratio'),
            config.getfloat(section, 'test_ratio'),
        )
    return splits, (default_val, default_test)


def get_dataset_split(dataset_name, splits, default):
    """Resolve per-dataset split ratios.

    Args:
        dataset_name: str
        splits: dict from load_dataset_splits()
        default: (val_ratio, test_ratio) fallback
    """
    return splits.get(dataset_name, default)


def define_ict_dataloader(args):
    """Create ICT-aware dataloaders.

    Key differences from define_dataloder:
    - Per-dataset split ratios loaded from conf/general_conf/dataset_splits.conf
    - Constructs a demo_pool from training split
    - Val/test datasets use training split's demo_pool (no data leakage)
    - Returns dataloaders that yield (query_x, query_y, demos_x, demos_y)
    """
    splits, default_split = load_dataset_splits()
    scaler_dict = {}
    datasets_train, datasets_val, datasets_test = [], [], []

    for dataset_name in args.dataset_use:
        data = load_st_dataset(dataset_name, args)
        val_ratio, test_ratio = get_dataset_split(dataset_name, splits, default_split)
        data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)

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

        # Create datasets (S=1 for train/val, S=num_prefix_selections for test)
        train_ds = ICTTrafficDataset(data_train, args.batch_size, iw, ow,
                                      demo_pool, args.num_demonstrations,
                                      num_prefix_selections=1, eval_only=False)
        val_ds = ICTTrafficDataset(data_val, args.batch_size, iw, ow,
                                    demo_pool, args.num_demonstrations,
                                    num_prefix_selections=1, eval_only=True)
        test_ds = ICTTrafficDataset(data_test, args.batch_size, iw, ow,
                                     demo_pool, args.num_demonstrations,
                                     num_prefix_selections=args.num_prefix_selections,
                                     eval_only=True)
        datasets_train.append(train_ds)
        datasets_val.append(val_ds)
        datasets_test.append(test_ds)

    # Concatenate and wrap in DataLoader (same pattern as original)
    train_loader = DataLoader(ConcatDataset(datasets_train), batch_size=1, shuffle=True)
    val_loader = DataLoader(ConcatDataset(datasets_val), batch_size=1, shuffle=False)
    test_loader = DataLoader(ConcatDataset(datasets_test), batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_dict
```
## 3.3 v3 Implementation: KNN-Based Demo Retrieval

### Motivation

Initial ICT experiments used random demonstration sampling.  
This produced unstable residual corrections because demonstrations often came from unrelated traffic regimes.

As a result:

- residual estimates varied significantly,
- performance degraded relative to zero-shot inference,
- increasing averaging (S) only partially reduced noise.

To resolve this, demonstration selection was upgraded to **similarity-based retrieval using K-Nearest Neighbors (KNN)**.

---

### Key Idea

Instead of randomly sampling demonstrations, each query retrieves demonstrations whose historical traffic patterns are closest in feature space.

Formally:

\[
demo^* = \arg\min_{d \in pool} \| x_{query} - x_d \|_2
\]

where histories are flattened into vectors and compared using Euclidean distance.

The assumption is:

> similar histories → similar prediction bias → transferable residuals.

---

### Implementation Location

All changes are implemented inside:
lib/ict_data_process.py


Specifically within:
ICTTrafficDataset


---

### Implementation Details

#### 1. Feature Construction

Each history window is flattened:
[num_windows, T × N × F]


This allows efficient similarity search while preserving temporal information.

---

#### 2. KNN Index Construction (Initialization)

During dataset initialization:

```python
from sklearn.neighbors import NearestNeighbors

self.knn = NearestNeighbors(
    n_neighbors=self.K,
    metric="euclidean"
)
self.knn.fit(self.demo_feature_matrix)

```

#### 3. Demo Retrieval (__getitem__)

For each query sample:

Flatten query history

Run KNN lookup

distances, indices = self.knn.kneighbors(query_vector)

Retrieve demo windows using returned indices

Return deterministic demo set
---

#### Step 2 ✅ (Done): OpenCity Model — `forward_ict` Method (`model/OpenCity/OpenCity.py`)

**Added method** `forward_ict` to the `OpenCity` class using **Residual Correction** approach (v2). The original `forward` method is **NOT modified** (full backward compatibility).

> **History**: v1 (sequence extension) was implemented first, tested in Experiments 2-4, and abandoned. The current code is v2 (residual correction). See `docs/05_experiment_log.md § Approach Evolution`.

```python
def forward_ict(self, input, lbls, demos_x, demos_y, select_dataset):
    """
    ICT forward pass using Residual Correction.

    Each input (query + each demo) is processed independently through
    the standard 24-patch forward() path. Demo residuals estimate and
    correct the model's systematic prediction error.

    Args:
        input:   [B, T, N, F]   — query history
        lbls:    [B, T, N, F]   — query future (temporal features only)
        demos_x: [B, K, T, N, F] — demonstration histories
        demos_y: [B, K, T, N, F] — demonstration futures (ground-truth flow + temporal)
        select_dataset: str — dataset identifier

    Returns:
        [B, T, N, 1] — corrected predicted future flow for query
    """
    K = demos_x.shape[1]

    # --- Query prediction (standard 24-patch forward) ---
    pred_query = self.forward(input, lbls, select_dataset)  # [B, T, N, 1]

    if K == 0:
        return pred_query  # No demos → pure zero-shot

    # --- Demo residuals ---
    residuals = []
    for k in range(K):
        dk_x = demos_x[:, k]  # [B, T, N, F]
        dk_y = demos_y[:, k]  # [B, T, N, F]

        # Demo prediction (same standard forward)
        pred_dk = self.forward(dk_x, dk_y, select_dataset)  # [B, T, N, 1]

        # Ground-truth future flow
        dk_gt_flow = dk_y[..., :self.output_dim]  # [B, T, N, 1]

        # Residual = ground truth - prediction (systematic error)
        residuals.append(dk_gt_flow - pred_dk)

    # --- Correction ---
    avg_residual = torch.stack(residuals).mean(dim=0)  # [B, T, N, 1]
    return pred_query + avg_residual
```

**Key Design Notes:**
- Every `self.forward()` call uses exactly 24 patches — identical to pretraining. Zero OOD.
- No new parameters. Reuses the pretrained model entirely.
- fp32 inference. No bfloat16 needed (same VRAM as zero-shot per call).
- K=0 gracefully degrades to zero-shot prediction.
- Residual correction is physically interpretable: corrects systematic model bias.

> **Legacy note**: `TemporalSelfAttention.forward()` and `STEncoderBlock.forward()` still have an optional `t_attn_mask=None` parameter from the v1 block-diagonal mask experiment. It defaults to None and is harmless — kept for reference but not used by v2.

---

#### Step 3 ✅ (Done): Traffic_model Wrapper (`model/Model.py`)

Modified `Traffic_model.forward` to transparently forward demo parameters:

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

#### Step 4 ✅ (Done): ICT Test Method (`model/BasicTrainer.py`)

Added static method `test_ict` to the `Trainer` class:

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
            inputs = inputs.squeeze(0).to(args.device)     # [B, T, N, F]
            targets = targets.squeeze(0).to(args.device)   # [B, T, N, F]
            demos_x = demos_x.squeeze(0).to(args.device)   # [B, S, K, T, N, F]
            demos_y = demos_y.squeeze(0).to(args.device)   # [B, S, K, T, N, F]

            select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
            S = demos_x.shape[1]  # num_prefix_selections

            # Average predictions over S independent demo sets
            # Each set uses *different* randomly sampled demos (sampled in dataloader)
            outputs = []
            for s in range(S):
                output_s = model(inputs, targets, select_dataset,
                                 demos_x=demos_x[:, s],    # [B, K, T, N, F]
                                 demos_y=demos_y[:, s])    # [B, K, T, N, F]
                outputs.append(output_s)
            output = torch.stack(outputs).mean(dim=0)      # [B, T, N, 1]

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

#### Step 5 ✅ (Done): Mode Dispatch (`model/Run.py`)

Added `mode='ict'` branch (fp32 inference, no bfloat16):

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

#### ~~Step 6: Configuration~~ → Merged into Step 0 ✅

> All configuration work (`dataset_splits.conf`, `ICT.conf`, `Params_pretrain.py`, `data_process.py` integration) has been completed and is documented in **Step 0**.

---

## 4. Compute & Memory Analysis (Residual Correction)

### 4.1 Forward Passes per Sample

With residual correction, each forward call processes exactly 24 patches (identical to zero-shot). The cost is K+1 forward calls per sample.

| K | Forward Calls | Relative Compute | Memory (per call) |
|---|---|---|---|
| 0 (zero-shot) | 1 | 1.0× | Same as baseline |
| 1 | 2 | 2.0× | Same as baseline |
| 3 | 4 | 4.0× | Same as baseline |
| 5 | 6 | 6.0× | Same as baseline |

**Key advantage over v1 (sequence extension)**:
- v1 K=1: 72×72 = 5,184 attention matrix (9× memory)
- v2 K=1: 2 × (24×24 = 576) attention matrix (2× compute, 1× memory per call)
- v1 K=3: 168×168 = 28,224 attention matrix (49× memory, needed bfloat16)
- v2 K=3: 4 × (24×24 = 576) attention matrix (4× compute, 1× memory per call)

**Memory**: No VRAM increase per call. Batch size does not need to be reduced.

### 4.2 Residual Correction: No Architecture Compatibility Concerns

Since every forward call goes through the unmodified `forward()` method with standard 24-patch input, there are **zero compatibility concerns**:
- All encoder blocks see their pretrained sequence length
- All attention matrices are 24×24
- The prediction head sees 24 patches as expected
- Instance normalization operates as designed
- fp32 precision preserved throughout

This is the key insight: by avoiding sequence extension entirely, we sidestep all the v1 issues (OOD sequence length, attention dilution, position encoding conflicts, bfloat16 precision loss).

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

> **Dataset configuration**: Set `dataset_use` in `conf/general_conf/pretrain.conf` before running. Split ratios (`val_ratio`, `test_ratio`) are **auto-resolved** per dataset from `conf/general_conf/dataset_splits.conf` — **all modes** (`ict`, `test`, `eval`, `ori`) benefit from this. No need to manually edit `val_ratio` / `test_ratio` in `pretrain.conf` anymore.

```bash
# ── pretrain.conf: only set dataset_use ──
#   dataset_use = ['CD_DIDI']
# (val_ratio / test_ratio auto-resolved from dataset_splits.conf for ALL modes)

# ICT with K=1 on CD_DIDI
python main.py -mode ict -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1

# ── pretrain.conf: dataset_use = ['PEMS07M'] ──

# ICT with K=3, 10 random selections averaged on PEMS07M
python main.py -mode ict -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth -num_demonstrations 3 \
    -num_prefix_selections 10

# Zero-shot (also auto-resolved)
python main.py -mode test -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth

# Fast Adaptation (also auto-resolved)
python main.py -mode eval -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth -epochs 3
```

---

## 6. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Residual correction adds noise if demos are dissimilar to query | Medium | Medium | Use demo selection strategies (recent, similar time-of-day) to pick more relevant demos |
| Random demo selection has high variance | Medium | Low | Average over S independent demo sets (S=10) to reduce variance |
| Systematic error is not transferable between samples | Medium | High | Core assumption of residual correction — validate empirically across datasets and K values |
| Marginal improvement over zero-shot | Medium | High | Validates the approach; any improvement without training is scientifically interesting |
| K+1 forward passes too slow for production | Low | Low | Forward passes are independent and trivially parallelizable |

**Resolved risks (from v1)**:
- ~~OOM with K≥3~~ → No longer an issue; each call is standard 24-patch size
- ~~Pretrained model can't process extended sequences~~ → No sequence extension in v2
- ~~bfloat16 precision loss~~ → fp32 throughout in v2

---

## 7. Implementation Priorities

| Priority | Task | Effort | Dependencies | Status |
|----------|------|--------|-------------|--------|
| P0 | `OpenCity.forward_ict()` method | 2h | None | ✅ Done (v2 residual correction) |
| P0 | `ICTTrafficDataset` + `define_ict_dataloader` | 2h | None | ✅ Done |
| P0 | `Traffic_model.forward` modification | 15min | forward_ict | ✅ Done |
| P0 | `Trainer.test_ict()` method | 1h | ICTTrafficDataset | ✅ Done |
| P0 | `Run.py` mode='ict' branch | 30min | All above | ✅ Done |
| P1 | `Params_pretrain.py` ICT args | 15min | None | ✅ Done |
| P1 | `conf/ICT/ICT.conf` | 5min | None | ✅ Done |
| P2 | E2E validation: Residual K=1 fp32 | 1h | All P0+P1 | 🔄 Pending (Exp 5) |
| P2 | Multi-K validation: K=3, K=5 | 1h | Exp 5 | Pending (Exp 6) |
| P2 | Variance reduction: S=10 | 1h | Exp 5 | Pending (Exp 7) |
| P3 | Demo selection strategies (recent, similar) | 2h | Basic ICT working | Pending |
| P3 | Full 8-dataset evaluation | 4h | K/S validated | Pending |

**P0+P1 completed**. Next: P2 experiments to validate residual correction approach.

---

## 8. Future Extensions

1. **Demo Selection Strategies**: Beyond random sampling — temporally recent, spatially similar (DTW-based), or same time-of-day/day-of-week demos could improve residual correction accuracy
2. **Weighted Residual Correction**: Weight each demo's residual by similarity to the query (e.g., cosine similarity of history embeddings) instead of uniform averaging
3. **Lightweight Adapter Training**: Add a small trainable MLP that learns to combine residuals, while keeping the base model frozen — a middle ground between zero-training ICT and full fine-tuning
4. **Cross-Dataset Demos**: Use demos from pretrained datasets to inform predictions on zero-shot datasets — testing cross-domain residual transfer
5. **Node-Specific Correction**: Instead of a global residual, compute per-node residuals using spatially similar nodes from demos
6. **Hybrid Approach**: Combine residual correction (for bias) with sequence extension (for pattern learning) if residual correction proves effective — use residual as initialization plus fine-tuned attention
