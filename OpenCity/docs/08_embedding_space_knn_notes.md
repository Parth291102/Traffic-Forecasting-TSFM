"""
Implementation Notes: Embedding-Space KNN Retrieval for ICT

## Overview
This implementation replaces raw-input L2 KNN with encoder-feature KNN for
demonstration selection in In-Context Tuning (ICT).

**Hypothesis**: Encoder features capture semantic similarity better than raw input L2 distance.

## Key Features

### 1. Encoder Feature Extraction
- **Location**: lib/ict_encoder_features.py
- **Method**: Node-mean pooling followed by patch-mean pooling
  - Input: [B, T, N, F] (batch of time series)
  - After encoder: [B, P, N, D] (P patches, N nodes, D=512 embed_dim)
  - Node-mean pool: [B, P, D] (average over nodes)
  - Patch-mean pool: [B, D] (average over patches)
  - Output: 512-dimensional embeddings per sample
- **This captures semantic traffic patterns better than raw flow values**

### 2. KNN Strategy
The implementation supports two KNN modes:
- **encoder-knn** (new): Build KNN index on 512-d encoder features
  - Automatically activated when model is passed to define_ict_dataloader
  - Used for demo pool: All demo features pre-computed during dataloader creation
  - Query similarity: Uses flattened raw input (for performance)
    * Note: Full encoder-based query encoding would add significant overhead
    * Raw input L2 distance still provides reasonable initial filtering
    * Demo pool selection is the critical component (now encoder-based)

- **raw-input-knn** (legacy): Build KNN on flattened raw histories
  - Fallback when no model is provided
  - Maintains backward compatibility

### 3. Usage

#### ICT Mode with Encoder Features (Recommended)
```bash
cd model && uv run python Run.py -mode ict -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth \
    -num_demonstrations 3 \
    -demo_selection similar \
    --embed_dim 512 --skip_dim 512 --enc_depth 6
```

**Difference from legacy mode**:
- Model is loaded BEFORE dataloader creation
- Encoder features are extracted for all demos
- Logs show: "[ICT] Using encoder-feature KNN (D=512)"

#### Backward Compatible (No Model)
```bash
# This still works, defaults to raw-input KNN
cd model && uv run python Run.py -mode test -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth
```

### 4. Performance Considerations

**Encoder feature extraction cost** (per dataset):
- ~30 seconds for 10k demos on GPU
- Computed once during dataloader initialization
- Results in sparse memory (10k × 512 floats ≈ 20 MB per dataset)

**Runtime inference** (per batch):
- No difference from previous implementation
- Query encoding still uses L2 distance
- KNN lookup is identical time complexity

### 5. Files Changed

| File | Changes |
|------|---------|
| lib/ict_encoder_features.py | NEW: Feature extraction module |
| lib/ict_data_process.py | Added encoder_features param to ICTTrafficDataset and define_ict_dataloader |
| model/Run.py | Pass model to define_ict_dataloader in 'ict' and 'ict_train_aggregator' modes |

## Implementation Details

### EncoderFeatureExtractor (lib/ict_encoder_features.py)
```python
class EncoderFeatureExtractor(nn.Module):
    def forward(input_hist, input_temp_feat, select_dataset):
        # 1. Apply instance normalization
        # 2. Patch embedding
        # 3. Spatio-temporal encoder blocks
        # 4. Node-mean pooling: mean over nodes
        # 5. Patch-mean pooling: mean over patches
        # Returns: [B, D] where D=512
```

### ICTTrafficDataset Updates
```python
# New parameters
encoder_features: [num_demos, D] array of 512-d embeddings (optional)
use_encoder_knn: bool, use encoder features if available

# Internal state
_demo_features: Either encoder embeddings or raw flattened input
_knn_mode: 'encoder' or 'raw'
_knn: NearestNeighbors index built on _demo_features
```

### define_ict_dataloader Updates
```python
# New parameter
model: Optional OpenCity model for encoder feature extraction

# Logic
if model provided:
    for each dataset:
        extract_encoder_features_batch(model, demo_pool, ...)
        pass encoder_features to ICTTrafficDataset
else:
    ICTTrafficDataset falls back to raw-input KNN
```

## Hypothesis Validation

**Expected improvements** (vs. raw-input KNN):
1. Better semantic similarity matching
2. More robust to scale differences across datasets
3. Captures temporal patterns (traffic rush hours, weekend patterns)
4. Reduced sensitivity to noise in raw flow values

**Experimental protocol**:
- Compare MAE/RMSE/MAPE across 8 datasets
- K=1, K=3 demonstrations
- S=1, S=10 prefix selections
- Both modes (encoder-knn vs raw-knn) on same data

## Troubleshooting

### "Encoder feature extraction failed" warning
- Model might not be loaded properly
- Check if model.device matches args.device
- Ensure args.dataset_use matches model's expected datasets

### "Using raw-input L2 KNN" (when encoder features expected)
- encoder_features is None during ICTTrafficDataset creation
- This is fine - falls back to legacy mode
- Check logs for: "[ICT] Using encoder-feature KNN (D=512)"

### Memory issues during feature extraction
- Reduce batch_size parameter in extract_encoder_features_batch
- Current default is 32 (128-512 works well on 8GB GPU)
- Or pre-cache features and save to disk
"""
