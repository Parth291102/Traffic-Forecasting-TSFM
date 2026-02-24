"""
ICT-aware data loading for OpenCity.

Provides ICTTrafficDataset and define_ict_dataloader() that augment each
query with K demonstration (history, future) pairs sampled from the
training split.
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
                 eval_only=False):
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
        """
        self.input_window = input_window
        self.output_window = output_window
        self.demo_pool = demo_pool
        self.num_demonstrations = num_demonstrations
        self.num_prefix_selections = num_prefix_selections

        # Create sliding windows (same as TrafficDataset)
        self.windows = [
            (data[i:i + input_window], data[i + input_window:i + input_window + output_window])
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

        # Precompute flattened demo histories and build KNN index
        if len(self.demo_pool) > 0:
            try:
                self._demo_hist_flat = np.stack([p[0] for p in self.demo_pool]).reshape(len(self.demo_pool), -1)
                self._knn = NearestNeighbors(n_neighbors=min(num_demonstrations, len(self.demo_pool)), metric='euclidean')
                self._knn.fit(self._demo_hist_flat)
            except Exception as e:
                print(f'[ICT] Warning: KNN index build failed: {e}. Falling back to random sampling.')
                self._knn = None
                self._demo_hist_flat = None
        else:
            self._knn = None
            self._demo_hist_flat = None

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_pairs = self.batches[idx]
        batch_x, batch_y = zip(*batch_pairs)
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
                # Flatten query for KNN search
                query_flat = batch_x[b].numpy().reshape(-1)
                demo_pool_size = len(self.demo_pool)

                # Select K nearest neighbors using KNN
                if self._knn is not None and demo_pool_size > 0:
                    # Query KNN for nearest neighbors
                    k_query = min(K, demo_pool_size)
                    _, indices_array = self._knn.kneighbors(query_flat.reshape(1, -1), n_neighbors=k_query)
                    indices_base = indices_array[0]
                    
                    # If K > pool_size, tile to get K samples
                    if K > demo_pool_size:
                        reps = int(np.ceil(float(K) / float(demo_pool_size)))
                        indices_base = np.tile(indices_base, reps)[:K]
                else:
                    # Fallback to random sampling if KNN not available
                    if demo_pool_size > 0:
                        indices_base = np.array(random.sample(range(demo_pool_size), min(K, demo_pool_size)))
                        if K > demo_pool_size:
                            reps = int(np.ceil(float(K) / float(demo_pool_size)))
                            indices_base = np.tile(indices_base, reps)[:K]
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


def define_ict_dataloader(args):
    """Create ICT-aware dataloaders.

    Key differences from define_dataloder:
    - Constructs a demo_pool from training split
    - Val/test datasets use training split's demo_pool (no data leakage)
    - Train/val use num_prefix_selections=1, test uses args.num_prefix_selections
    - Returns dataloaders that yield (query_x, query_y, demos_x, demos_y)
    """
    dataloder_trn_list, dataloder_val_list, dataloder_tst_list = [], [], []
    scaler_dict = {}
    num_nodes_dict = {}

    splits, default_split = load_dataset_splits()

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

        # Build demo pool from training windows (post-normalization)
        demo_pool = [
            (data_train[i:i + iw], data_train[i + iw:i + iw + ow])
            for i in range(len(data_train) - iw - ow + 1)
        ]
        print(f'[ICT] demo_pool size: {len(demo_pool)}, iw={iw}, ow={ow}')

        # Create datasets
        # Train/val: S=1 (single demo selection)
        # Test: S=num_prefix_selections (for variance reduction averaging)
        train_ds = ICTTrafficDataset(
            data_train, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=1, eval_only=False)
        val_ds = ICTTrafficDataset(
            data_val, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=1, eval_only=True)
        test_ds = ICTTrafficDataset(
            data_test, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=args.num_prefix_selections,
            eval_only=True)

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
