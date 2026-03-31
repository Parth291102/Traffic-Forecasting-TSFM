"""
ICT-aware data loading for OpenCity.

Provides ICTTrafficDataset and define_ict_dataloader() that augment each
query with K demonstration (history, future) pairs sampled from the
training split.

Supports two KNN strategies:
1. Raw-input L2 KNN (legacy): Uses flattened raw histories
2. Encoder-feature KNN (new): Uses 512-d node-mean pooled encoder embeddings
"""

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.neighbors import NearestNeighbors

from lib.data_process import (
    load_st_dataset,
    split_data_by_ratio,
    normalize_dataset,
    load_dataset_splits,
    get_dataset_split,
)
from lib.ict_encoder_features import extract_encoder_features_batch


class ICTTrafficDataset(Dataset):
    """Traffic dataset with demonstration sampling for ICT.

    Follows the same internal-batching pattern as TrafficDataset:
    sliding windows are pre-batched so that each __getitem__ returns
    a full mini-batch.

    Returns:
        batch_x:  [B, T, N, F]           — query histories
        batch_y:  [B, T, N, F]           — query futures
        demos_x:  [B, S, K, T, N, F]    — demonstration histories
        demos_y:  [B, S, K, T, N, F]    — demonstration futures
    """

    def __init__(self, data, batch_size, input_window, output_window,
                 demo_pool, num_demonstrations=1, num_prefix_selections=1,
                 demo_selection='random', eval_only=False, exclude_overlap=False,
                 encoder_features=None, use_encoder_knn=True):
        """
        Args:
            data:          numpy array [T_total, N, F] for this split
            batch_size:    number of samples per mini-batch
            input_window:  history window length
            output_window: prediction window length
            demo_pool:     list of (hist_array, future_array) tuples from training split
            num_demonstrations: K — number of demo pairs per query
            num_prefix_selections: S — number of independent demo sets
            eval_only:     if True, don't shuffle / drop_last
            exclude_overlap: if True, exclude demo pool entries that temporally
                             overlap with the query window (prevents data leakage
                             when query and demo pools share the same data split)
            encoder_features: [num_demos, D] array of encoder embeddings (512-d)
                             If provided with use_encoder_knn=True, uses encoder-space KNN
            use_encoder_knn: if True and encoder_features provided, use encoder-space KNN
                            otherwise fall back to raw-input L2 KNN
        """
        self.input_window = input_window
        self.output_window = output_window
        # demo_pool entries are tuples: (history, future, start_index)
        # older versions constructed a list of (hist, fut) pairs; wrap
        # those into triples with index=0 for backwards compatibility.
        self.demo_pool = []
        for entry in demo_pool:
            if len(entry) == 3:
                self.demo_pool.append(entry)
            else:
                # legacy: no index information
                self.demo_pool.append((entry[0], entry[1], 0))

        self.num_demonstrations = num_demonstrations
        self.num_prefix_selections = num_prefix_selections
        self.demo_selection = demo_selection
        self.num_demonstrations = num_demonstrations
        self.num_prefix_selections = num_prefix_selections
        self._exclude_overlap = exclude_overlap
        # Minimum gap between query start and demo start to avoid data overlap.
        # A query window spans [start, start + iw + ow). Any demo whose window
        # overlaps this range shares data and must be excluded.
        self._min_gap = input_window + output_window

        # Create sliding windows (same as TrafficDataset) but also record
        # the start index of each window to support "recent" selection.
        self.windows = [
            (data[i:i + input_window],
             data[i + input_window:i + input_window + output_window],
             i)
            for i in range(len(data) - input_window - output_window + 1)
        ]

        # drop_last & shuffle (same as TrafficDataset)
        if not eval_only:
            random.shuffle(self.windows)
            if len(self.windows) % batch_size != 0:
                self.windows = self.windows[:-(len(self.windows) % batch_size)]

        # pre-batch
        self.batches = [
            self.windows[i:i + batch_size]
            for i in range(0, len(self.windows), batch_size)
        ]

        # Precompute demo features (either encoder-space or raw-input) and build KNN index
        if len(self.demo_pool) > 0:
            # keep a separate list of start indices to support 'recent' selection
            self._demo_indices = np.array([p[2] for p in self.demo_pool])
            try:
                # Determine which KNN mode to use
                if use_encoder_knn and encoder_features is not None:
                    # Use encoder-feature KNN (new method)
                    self._demo_features = encoder_features  # [num_demos, D]
                    self._knn_mode = 'encoder'
                    print(f'[ICT] Using encoder-feature KNN (D={encoder_features.shape[1]})')
                else:
                    # Fall back to raw-input L2 KNN (legacy method)
                    demo_hists = [p[0] for p in self.demo_pool]
                    self._demo_features = np.stack(demo_hists).reshape(len(self.demo_pool), -1)
                    self._knn_mode = 'raw'
                    print(f'[ICT] Using raw-input L2 KNN')

                # Only build KNN if selection strategy relies on similarity
                if self.demo_selection == 'similar':
                    # Request extra neighbors to compensate for overlap exclusion.
                    knn_k = min(num_demonstrations, len(self.demo_pool))
                    if self._exclude_overlap:
                        knn_k = min(knn_k + 2 * self._min_gap, len(self.demo_pool))
                    self._knn = NearestNeighbors(n_neighbors=knn_k, metric='euclidean')
                    self._knn.fit(self._demo_features)
                    print(f'[ICT] KNN index built: {len(self.demo_pool)} demos, k={knn_k}, mode={self._knn_mode}')
                else:
                    self._knn = None
            except Exception as e:
                print(f'[ICT] Warning: demo precompute failed ({e}). Falling back to random sampling.')
                self._knn = None
                self._demo_features = None
                self._knn_mode = None
        else:
            self._knn = None
            self._demo_features = None
            self._demo_indices = np.array([])
            self._knn_mode = None

    def __len__(self):
        return len(self.batches)

    def _filter_overlapping(self, indices, query_start_idx):
        """Remove demo pool indices whose windows overlap with the query.

        Two windows overlap when |demo_start - query_start| < min_gap
        (i.e. the demo's [start, start+iw+ow) range intersects the query's).

        Args:
            indices: np.array of demo pool indices
            query_start_idx: start index of the query window

        Returns:
            np.array of non-overlapping indices
        """
        if not self._exclude_overlap or len(indices) == 0:
            return indices
        demo_starts = self._demo_indices[indices]
        mask = np.abs(demo_starts - query_start_idx) >= self._min_gap
        return indices[mask]

    def __getitem__(self, idx):
        batch_pairs = self.batches[idx]
        # each element of batch_pairs is now (hist, fut, start_idx)
        batch_x, batch_y, batch_idx = zip(*batch_pairs)
        batch_x = torch.from_numpy(np.stack(batch_x)).float()  # [B, T, N, F]
        batch_y = torch.from_numpy(np.stack(batch_y)).float()  # [B, T, N, F]

        B = batch_x.shape[0]
        K = self.num_demonstrations
        S = self.num_prefix_selections

        demos_x_list, demos_y_list = [], []
        if K == 0:
            # K=0: no demonstrations — return empty tensors with correct shape
            T, N, F = batch_x.shape[1], batch_x.shape[2], batch_x.shape[3]
            demos_x = torch.zeros(B, S, 0, T, N, F)
            demos_y = torch.zeros(B, S, 0, T, N, F)
        else:
            for b in range(B):
                sets_x, sets_y = [], []
                demo_pool_size = len(self.demo_pool)

                # choose demo indices depending on selection strategy
                if self.demo_selection == 'similar' and self._knn is not None and demo_pool_size > 0:
                    # Query KNN for nearest neighbors (extra if overlap exclusion)
                    knn_k = self._knn.n_neighbors
                    
                    # Prepare query feature
                    if self._knn_mode == 'encoder':
                        # For encoder KNN, use the batch_x as-is and flatten for querying
                        # Note: In production, we should compute encoder features here,
                        # but that requires model access. For now, flatten raw input as proxy.
                        query_flat = batch_x[b].numpy().reshape(-1)
                    else:
                        # For raw KNN, use flattened raw input
                        query_flat = batch_x[b].numpy().reshape(-1)
                    
                    _, indices_array = self._knn.kneighbors(query_flat.reshape(1, -1), n_neighbors=knn_k)
                    candidates = indices_array[0]
                    # Exclude overlapping windows
                    candidates = self._filter_overlapping(candidates, batch_idx[b])
                    indices_base = candidates[:K]
                    if len(indices_base) < K and len(indices_base) > 0:
                        reps = int(np.ceil(float(K) / float(len(indices_base))))
                        indices_base = np.tile(indices_base, reps)[:K]
                elif self.demo_selection == 'recent' and demo_pool_size > 0:
                    # select demos whose start index is just before query start
                    start_idx = batch_idx[b]
                    # find demo_pool entries with index < start_idx
                    valid = np.where(self._demo_indices < start_idx)[0]
                    # Exclude overlapping windows
                    valid = self._filter_overlapping(valid, batch_idx[b])
                    if len(valid) > 0:
                        # take the most recent ones (largest indices)
                        choose = valid[np.argsort(self._demo_indices[valid])][-K:]
                        indices_base = choose
                        if len(indices_base) < K:
                            reps = int(np.ceil(float(K) / float(len(indices_base))))
                            indices_base = np.tile(indices_base, reps)[:K]
                    else:
                        # fallback to random if no earlier demos exist
                        indices_base = np.array([])
                else:
                    # random sampling (also covers cases where pool is empty or
                    # KNN not built or 'random' strategy selected)
                    if demo_pool_size > 0:
                        all_indices = np.arange(demo_pool_size)
                        # Exclude overlapping windows
                        all_indices = self._filter_overlapping(all_indices, batch_idx[b])
                        if len(all_indices) > 0:
                            indices_base = np.array(random.sample(
                                list(all_indices), min(K, len(all_indices))))
                            if K > len(all_indices):
                                reps = int(np.ceil(float(K) / float(len(all_indices))))
                                indices_base = np.tile(indices_base, reps)[:K]
                        else:
                            indices_base = np.array([])
                    else:
                        indices_base = np.array([])

                for s in range(S):
                    # Use same KNN indices for all S selections (deterministic per query)
                    if len(indices_base) > 0:
                        dk_x = np.stack([self.demo_pool[int(i)][0] for i in indices_base])  # [K, T, N, F]
                        dk_y = np.stack([self.demo_pool[int(i)][1] for i in indices_base])  # [K, T, N, F]
                    else:
                        # Create empty arrays with correct shape [0, T, N, F]
                        T, N, F = batch_x.shape[1], batch_x.shape[2], batch_x.shape[3]
                        dk_x = np.zeros((0, T, N, F))
                        dk_y = np.zeros((0, T, N, F))
                    sets_x.append(dk_x)
                    sets_y.append(dk_y)
                demos_x_list.append(np.stack(sets_x))  # [S, K, T, N, F]
                demos_y_list.append(np.stack(sets_y))   # [S, K, T, N, F]

            demos_x = torch.from_numpy(np.stack(demos_x_list)).float()  # [B, S, K, T, N, F]
            demos_y = torch.from_numpy(np.stack(demos_y_list)).float()  # [B, S, K, T, N, F]

        return batch_x, batch_y, demos_x, demos_y


def define_ict_dataloader(args, model=None):
    """Create ICT-aware dataloaders with optional encoder-feature KNN.

    Key differences from define_dataloder:
    - Constructs a demo_pool from training split
    - Val/test datasets use training split's demo_pool (no data leakage)
    - Train/val use num_prefix_selections=1, test uses args.num_prefix_selections
    - If model is provided, extracts encoder features for encoder-space KNN
    - Returns dataloaders that yield (query_x, query_y, demos_x, demos_y)

    Args:
        args: Training arguments
        model: Optional pretrained OpenCity model for encoder feature extraction
               If None, falls back to raw-input L2 KNN
    """
    dataloder_trn_list, dataloder_val_list, dataloder_tst_list = [], [], []
    scaler_dict = {}
    num_nodes_dict = {}

    splits, default_split = load_dataset_splits()
    
    # Determine if we should use encoder KNN
    use_encoder_knn = model is not None
    if use_encoder_knn:
        print(f'[ICT] Encoder model available: will use encoder-feature KNN')
    else:
        print(f'[ICT] No encoder model provided: will use raw-input L2 KNN')

    for dataset_name in args.dataset_use:
        val_ratio, test_ratio = get_dataset_split(dataset_name, splits, default_split)
        print(f'[ICT] {dataset_name}: val_ratio={val_ratio}, test_ratio={test_ratio}')

        data = load_st_dataset(dataset_name, args)
        num_nodes_dict[dataset_name] = data.shape[1]
        data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
        print(f'[ICT] data_train {data_train.shape}, data_val {data_val.shape}, data_test {data_test.shape}')

        # Normalize using training split statistics
        if args.real_value == False:
            scaler_data, scaler_day, scaler_week = normalize_dataset(data_train, args.input_base_dim)
            print(f'[ICT] scaler mean={scaler_data.mean:.4f}, std={scaler_data.std:.4f}')
            data_train[..., :args.input_base_dim] = scaler_data.transform(data_train[:, :, :args.input_base_dim])
            data_val[..., :args.input_base_dim] = scaler_data.transform(data_val[:, :, :args.input_base_dim])
            data_test[..., :args.input_base_dim] = scaler_data.transform(data_test[:, :, :args.input_base_dim])
            scaler_dict[dataset_name] = scaler_data
        else:
            scaler_dict[dataset_name] = None

        # Compute input/output window based on dataset interval
        if dataset_name.startswith("Traffic") or dataset_name.startswith("NYC") or dataset_name.startswith("CHI"):
            intervel = 30
        elif 'DIDI' in dataset_name:
            intervel = 10
        else:
            intervel = 5
        iw = args.his // (intervel // 5)
        ow = args.pred // (intervel // 5)

        # Build demo pool from training windows (post-normalization).
        # we attach the start index so that the dataset can support
        # selection strategies like "recent".
        demo_pool = [
            (data_train[i:i + iw], data_train[i + iw:i + iw + ow], i)
            for i in range(len(data_train) - iw - ow + 1)
        ]
        print(f'[ICT] demo_pool size: {len(demo_pool)}, iw={iw}, ow={ow}')

        # Extract encoder features for demo pool if model is available
        encoder_features = None
        if use_encoder_knn and len(demo_pool) > 0 and model is not None:
            print(f'[ICT] Extracting encoder features for {len(demo_pool)} demos...')
            try:
                # Extract demo histories and temporal features
                # Using raw traffic data [T, N, 3]
                demo_hists = [p[0] for p in demo_pool]
                demo_temps = [p[0].copy() for p in demo_pool]
                
                # Get the actual model (handle DataParallel wrapper)
                opencity_model = model.module if hasattr(model, 'module') else model
                
                # Attempt encoder feature extraction
                encoder_features = extract_encoder_features_batch(
                    opencity_model,
                    demo_hists,
                    demo_temps,
                    dataset_name,
                    batch_size=32,
                    device=args.device,
                    enable_extraction=True
                )
                
                if encoder_features is not None:
                    print(f'[ICT] Extracted encoder features: shape {encoder_features.shape}')
                else:
                    print(f'[ICT] Encoder feature extraction returned None.')
                    print(f'[ICT] Using raw-input L2 KNN for demo selection.')
                    
            except Exception as e:
                print(f'[ICT] Warning: encoder feature extraction failed ({e}).')
                print(f'[ICT] Using raw-input L2 KNN for demo selection.')
                encoder_features = None

        # Create datasets
        # Train/val: S=1 (single demo selection)
        # Test: S=num_prefix_selections (for variance reduction averaging)
        # exclude_overlap=True for train: query and demo_pool share the same
        # data split, so overlapping windows must be excluded to prevent the
        # aggregator from learning trivial identity corrections.
        train_ds = ICTTrafficDataset(
            data_train, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=1, demo_selection=args.demo_selection,
            eval_only=False, exclude_overlap=True,
            encoder_features=encoder_features, use_encoder_knn=use_encoder_knn)
        val_ds = ICTTrafficDataset(
            data_val, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=1, demo_selection=args.demo_selection,
            eval_only=True,
            encoder_features=encoder_features, use_encoder_knn=use_encoder_knn)
        test_ds = ICTTrafficDataset(
            data_test, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=args.num_prefix_selections,
            demo_selection=args.demo_selection,
            eval_only=True,
            encoder_features=encoder_features, use_encoder_knn=use_encoder_knn)

        dataloder_trn_list.append(train_ds)
        dataloder_val_list.append(val_ds)
        dataloder_tst_list.append(test_ds)

    train_combine = ConcatDataset(dataloder_trn_list)
    val_combine = ConcatDataset(dataloder_val_list)
    test_combine = ConcatDataset(dataloder_tst_list)

    train_dataloader = DataLoader(train_combine, batch_size=1, shuffle=True) if len(train_combine) > 0 else None
    val_dataloader = DataLoader(val_combine, batch_size=1, shuffle=False) if len(val_combine) > 0 else None
    test_dataloader = DataLoader(test_combine, batch_size=1, shuffle=False) if len(test_combine) > 0 else None

    args.num_nodes_dict = num_nodes_dict
    return train_dataloader, val_dataloader, test_dataloader, scaler_dict
