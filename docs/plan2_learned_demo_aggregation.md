# Plan 2: Learned Demo Aggregation — Lightweight Learned Correction

## 1. Core Idea

Keep the current K+1 independent forward pass framework, but **replace naive averaging with a learned module** that weights and combines demo corrections based on query-demo feature similarity.

Borrows the paper's core insight: the model should **learn how to use demos**, rather than blindly averaging.

```
Current:  pred = f(query) + (1/K) * sum(GT_k - f(demo_k))              ← naive average
Proposed: pred = f(query) + Aggregator(query_enc, demo_encs, corrections)  ← learned weighting
```

### Why This is Better Than Naive Averaging

Problems with naive averaging:
- All demos weighted equally, but some are more relevant to the query than others
- Same correction applied uniformly across all nodes, though error patterns may differ per node
- Cannot adapt to varying K values or demo quality

Learned aggregation advantages:
- **Per-node, per-demo weights**: each node can assign different importance to each demo
- **Similarity-based**: demos more similar to query receive higher weight
- **Low training cost**: base model frozen, only small module trained (~3K-200K params)
- **CPU-friendly**: no additional forward pass overhead, just a lightweight aggregation step

## 2. Architecture Overview

```
                                     +-----------------+
Query History ────> Base Model ────> Query Pred (f_q)  │
                       │             +-----------------+
                       │ extract              │
                       │ encoder              │
                       │ features             v
                       │             ┌═══════════════════┐
Demo 1 History ──> Base Model ──> Pred (f_1) ──────────┤
Demo 1 GT ─────────────────────────> Corr_1 = GT-f_1 ──┤
                                                        │
Demo 2 History ──> Base Model ──> Pred (f_2) ──────────┤──> DemoAggregator ──> weighted correction
Demo 2 GT ─────────────────────────> Corr_2 = GT-f_2 ──┤    (learned)
                       ...                              │
Demo K History ──> Base Model ──> Pred (f_K) ──────────┤
Demo K GT ─────────────────────────> Corr_K = GT-f_K ──┤
                                    └═══════════════════┘
                                            │
                                            v
                                    pred_query + weighted_correction
```

## 3. Two Aggregator Variants

### Variant A: SimpleDemoAggregator (~8K params) — Recommended First

**Principle**: cosine-similarity-based weighting

```
Inputs:
  query_feat:  [B, N, D]        ← encoder output mean-pooled over patches
  demo_feats:  [B, K, N, D]     ← each demo's encoder output mean-pooled
  corrections: [B, K, T, N, 1]  ← per-demo residual corrections

Computation:
  1. q_proj = normalize(Linear(query_feat))     [B, N, D]
  2. d_proj = normalize(Linear(demo_feats))     [B, K, N, D]
  3. similarity = cosine(q_proj, d_proj)         [B, K, N]     ← per-node per-demo
  4. weights = softmax(similarity / tau, dim=K)  [B, K, N]     ← tau is learnable temperature
  5. correction = scale * sum_k(weights_k * corrections_k)  [B, T, N, 1]

Learnable parameters:
  - query_proj: Linear(D, D)     ~D^2 params
  - demo_proj:  Linear(D, D)     ~D^2 params
  - temperature: 1 param
  - correction_scale: 1 param
  - Total: ~2*D^2 + 2 ≈ 8194 params (D=64)
```

### Variant B: DemoAggregator (~198K params) — Multi-head Cross-Attention

**Principle**: standard multi-head Q·K cross-attention + per-node softplus scale

**Key design choices:**
- Multi-head Q·K attention (H=4 heads, hd=32): each head learns a different demo-weighting strategy
- Corrections used directly as "values" — no V-projection, no correction_encoder, no mean-pool over T
- H candidate corrections combined via learned Linear(H→1) (init to 1/H = head-average)
- Per-node softplus scale replaces sigmoid gate — can amplify corrections, always ≥ 0
- Zero-init on scale_net → softplus(0) ≈ 0.693 for gentle start
- No FFN (V1's 1M-param FFN was overkill for a scalar gate)

```
Inputs: same as Variant A

Computation:
  1. Q = Linear(LayerNorm(query_feat))             [B, N, H, hd]  ← D → H*hd
  2. K = Linear(LayerNorm(demo_feats))             [B, K, N, H, hd]
  3. attn = softmax(Q·K^T / sqrt(hd), dim=K)      [B, N, H, K]   ← per-head demo weights
  4. per-head weighted corr:                        [B, T, N, H]   ← H candidate corrections
     weighted_h = sum_k(attn[h,k] * corrections_k)
  5. out = Linear(weighted, H→1)                   [B, T, N, 1]   ← combine heads
  6. scale = softplus(MLP(query_feat))              [B, N, 1]      ← per-node scale (≥0)
  7. output = out * scale                           [B, T, N, 1]

Learnable parameters (D=512, H=4, proj_dim=128, hd=32):
  - LayerNorm:        2 * D = 1,024
  - Q proj (D→128):   D * 128 + 128 = 65,664
  - K proj (D→128):   D * 128 + 128 = 65,664
  - head_combine (4→1, no bias): 4              (init to 1/H)
  - scale_net:        D * 128 + 128 + 128 + 1 = 65,793  (zero-init last layer)
  - Total: ~198K params
```

## 4. Implementation Steps

### Step 1: Create `OpenCity/model/OpenCity/DemoAggregator.py` (New File)

Two classes sharing the same interface:
- `SimpleDemoAggregator(embed_dim)` — cosine similarity variant
- `DemoAggregator(embed_dim, num_heads=4, proj_dim=None, dropout=0.1)` — multi-head cross-attention variant

Both implement: `forward(query_feat, demo_feats, corrections) → [B, T, N, 1]`

### Step 2: Modify `OpenCity/model/OpenCity/OpenCity.py`

#### 2a. Add attribute in `__init__`

After `self.linear = nn.Linear(...)`:
```python
self.demo_aggregator = None  # lazily initialized when needed
```

#### 2b. New method `init_demo_aggregator(self, aggregator_type='simple')`

```python
def init_demo_aggregator(self, aggregator_type='simple'):
    from .DemoAggregator import SimpleDemoAggregator, DemoAggregator
    if aggregator_type == 'attention':
        self.demo_aggregator = DemoAggregator(self.embed_dim, num_heads=4)
    else:
        self.demo_aggregator = SimpleDemoAggregator(self.embed_dim)
    return self.demo_aggregator
```

#### 2c. New method `_forward_with_features(self, input, lbls, select_dataset)`

Identical computation to `forward()`, but additionally returns encoder output mean-pooled over patches:

```python
def _forward_with_features(self, input, lbls, select_dataset):
    # ... (same IN, patch_embed, encoder steps as forward()) ...

    # Extract encoder features (mean pool over patches)
    enc_feat = enc.mean(dim=1)  # [B, num_patches=2, N, D] → [B, N, D]

    # ... (prediction head + DeIN same as forward()) ...

    return skip, enc_feat  # (prediction [B,T,N,1], features [B,N,D])
```

#### 2d. New method `forward_ict_learned(self, input, lbls, demos_x, demos_y, select_dataset)`

```python
def forward_ict_learned(self, input, lbls, demos_x, demos_y, select_dataset):
    K = demos_x.shape[1]
    if K == 0:
        return self.forward(input, lbls, select_dataset)

    # 1. Query forward (with features)
    pred_query, query_feat = self._forward_with_features(input, lbls, select_dataset)

    # 2. Each demo forward (with features)
    demo_feats_list = []
    corrections_list = []
    for k in range(K):
        dk_pred, dk_feat = self._forward_with_features(
            demos_x[:, k], demos_y[:, k], select_dataset)
        dk_gt = demos_y[:, k, ..., :self.output_dim]
        corrections_list.append(dk_gt - dk_pred)
        demo_feats_list.append(dk_feat)

    # 3. Stack → [B, K, N, D] and [B, K, T, N, 1]
    demo_feats = torch.stack(demo_feats_list, dim=1)
    corrections = torch.stack(corrections_list, dim=1)

    # 4. Learned aggregation
    weighted_corr = self.demo_aggregator(query_feat, demo_feats, corrections)

    return pred_query + weighted_corr
```

### Step 3: Modify `OpenCity/model/Model.py`

Update `Traffic_model.forward()` signature and routing:

```python
def forward(self, source, label, select_dataset, batch_seen=None,
            demos_x=None, demos_y=None, ict_mode='residual'):
    if self.model == 'OpenCity':
        if demos_x is not None:
            if ict_mode == 'learned':
                x_predic = self.predictor.forward_ict_learned(
                    source, label, demos_x, demos_y, select_dataset)
            else:
                x_predic = self.predictor.forward_ict(
                    source, label, demos_x, demos_y, select_dataset)
        else:
            x_predic = self.predictor(source, label, select_dataset)
    else:
        x_predic = self.predictor(source[..., 0:self.input_base_dim], select_dataset)
    return x_predic
```

### Step 4: Modify `OpenCity/model/BasicTrainer.py`

#### 4a. New method `train_aggregator(self)`

Core flow:
```python
def train_aggregator(self):
    # 1. Freeze all base model params
    for param in self.model.parameters():
        param.requires_grad = False

    # 2. Initialize aggregator (get predictor reference)
    predictor = self.model.predictor  # no DataParallel on CPU
    aggregator = predictor.init_demo_aggregator(self.args.aggregator_type)
    aggregator = aggregator.to(self.args.device)
    for param in aggregator.parameters():
        param.requires_grad = True

    # 3. Optimizer for aggregator params only
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=self.args.aggregator_lr)

    # 4. Training loop (same pattern as multi_train)
    best_loss = float('inf')
    best_state = None
    for epoch in range(self.args.aggregator_epochs):
        # --- Train ---
        self.model.train()
        total_loss = 0
        for batch_data in self.train_dataloader:
            inputs, targets, demos_x, demos_y = batch_data
            # ... squeeze + to(device) ...
            output = self.model(inputs, targets, dataset,
                              demos_x=demos_x[:, 0], demos_y=demos_y[:, 0],
                              ict_mode='learned')
            optimizer.zero_grad()
            loss = self.loss(output, targets[..., :self.args.output_dim], scaler)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Validation + early stopping ---
        val_loss = self._val_aggregator_epoch()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(aggregator.state_dict())

    # 5. Save aggregator weights
    torch.save(best_state, os.path.join(self.args.log_dir, 'aggregator_best.pth'))
```

#### 4b. Modify `test_ict()` signature

Add `ict_mode='residual'` parameter, pass it through to `model()`:
```python
@staticmethod
def test_ict(model, args, scaler_dict, test_dataloader, logger, path=None,
             num_prefix_selections=1, ict_mode='residual'):
    # ...
    output_s = model(inputs_m, targets_m, select_dataset,
                     demos_x=demos_x_m[:, s],
                     demos_y=demos_y_m[:, s],
                     ict_mode=ict_mode)       # ← pass ict_mode through
```

### Step 5: Modify `OpenCity/lib/Params_pretrain.py`

Add after existing ICT parameters (line 73):
```python
# ICT learned aggregation parameters
args.add_argument('-ict_mode', default='residual', type=str,
                  choices=['residual', 'learned'],
                  help='ICT mode: residual (average) or learned (aggregator)')
args.add_argument('-aggregator_type', default='simple', type=str,
                  choices=['simple', 'attention'],
                  help='Demo aggregator: simple (cosine) or attention (cross-attn)')
args.add_argument('-aggregator_epochs', default=5, type=int,
                  help='Epochs to train demo aggregator')
args.add_argument('-aggregator_lr', default=1e-4, type=float,
                  help='Learning rate for aggregator training')
```

### Step 6: Modify `OpenCity/model/Run.py`

#### 6a. Add `ict_train_aggregator` mode (after the `ict` block)

```python
elif args.mode == 'ict_train_aggregator':
    from lib.ict_data_process import define_ict_dataloader

    # Load pretrained base model
    path = log_dir + '/' + args.load_pretrain_path
    # ... (same loading logic as ict mode) ...

    # Create ICT dataloaders (train + val + test)
    train_dl_ict, val_dl_ict, test_dl_ict, scaler_dict_ict = define_ict_dataloader(args)

    trainer_ict = Trainer(model, loss, optimizer, train_dl_ict, val_dl_ict,
                          test_dl_ict, scaler_dict_ict, args, scheduler=None)

    # Train aggregator (base model frozen)
    trainer_ict.train_aggregator()

    # Test with learned aggregation
    trainer_ict.test_ict(model, args, scaler_dict_ict, test_dl_ict,
                         trainer_ict.logger, ict_mode='learned')
```

#### 6b. Modify existing `ict` mode to support `ict_mode`

When `args.ict_mode == 'learned'`:
1. Load base model weights
2. Initialize aggregator and load `aggregator_best.pth`
3. Pass `ict_mode='learned'` to `test_ict`

## 5. Training Cost Analysis

| Item | Value |
|------|-------|
| Base model params | ~millions (all frozen, no gradients) |
| Aggregator params (simple) | ~8K (all trained) |
| Aggregator params (attention) | ~198K (all trained) |
| Forward passes | K+1 (same as current) |
| Backward pass | Only through aggregator (~8K-200K params) |
| Expected training epochs | 5-10 |
| CPU feasibility | **Fully feasible** |

Key advantage: base model frozen means K+1 forward passes do not need to store intermediate activations for backpropagation. Aggregator gradients only flow through corrections (detached from base model) and the aggregator's own parameters.

## 6. Files Affected

| File | Change Type | Scope |
|------|-------------|-------|
| `OpenCity/model/OpenCity/DemoAggregator.py` | **New** | ~150 lines, 2 classes |
| `OpenCity/model/OpenCity/OpenCity.py` | Edit | +3 methods, +1 attribute |
| `OpenCity/model/Model.py` | Edit | Add `ict_mode` param to forward |
| `OpenCity/model/BasicTrainer.py` | Edit | +1 method, modify test_ict signature |
| `OpenCity/lib/Params_pretrain.py` | Edit | +4 arguments |
| `OpenCity/model/Run.py` | Edit | +1 mode, modify ict mode |

## 7. Usage

```bash
# 1. Train aggregator (CPU, K=3, quick validation)
python model/Run.py -mode ict_train_aggregator -model OpenCity \
  -num_demonstrations 3 -demo_selection similar \
  -aggregator_type simple -aggregator_epochs 2 -aggregator_lr 1e-4 \
  -use_cpu True -load_pretrain_path <pretrained_weights.pth>

# 2. Test with learned aggregation
python model/Run.py -mode ict -model OpenCity \
  -ict_mode learned -num_demonstrations 5 -demo_selection similar \
  -aggregator_type simple -use_cpu True \
  -load_pretrain_path <pretrained_weights.pth>

# 3. Compare against baseline (naive averaging)
python model/Run.py -mode ict -model OpenCity \
  -ict_mode residual -num_demonstrations 5 -demo_selection similar \
  -use_cpu True -load_pretrain_path <pretrained_weights.pth>
```

## 8. Expected Results

- **Simple aggregator**: moderate improvement over naive averaging (~2-5% MAE reduction)
  - Main gains from: demos more similar to query receiving higher weight
  - Especially effective with KNN `demo_selection=similar`, where it further refines weight allocation

- **Attention aggregator**: potentially slight additional improvement (~1-3%)
  - More parameters provide more flexible weighting strategies
  - But may overfit on small datasets

**Recommended order**: Simple → validate effectiveness → try Attention if needed
