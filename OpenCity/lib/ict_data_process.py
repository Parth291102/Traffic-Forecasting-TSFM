"""ICT-aware data loading for OpenCity.

Provides ICTTrafficDataset (with demonstration sampling) and
define_ict_dataloader() used by mode='ict' in Run.py.
"""

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Reuse helpers already in data_process.py
from lib.data_process import (
    load_st_dataset,
    split_data_by_ratio,
    normalize_dataset,
    load_dataset_splits,
    get_dataset_split,
)


def _compute_interval(dataset_name):
    """Return the sampling interval (minutes) for a dataset — mirrors define_dataloder logic."""
    if (dataset_name.startswith("Traffic")
            or dataset_name.startswith("NYC")
            or dataset_name.startswith("CHI")):
        return 30
    elif 'DIDI' in dataset_name:
        return 10
    else:
        return 5


class ICTTrafficDataset(Dataset):
    """Traffic dataset with demonstration sampling for ICT.

    Each __getitem__ returns:
        batch_x:  [B, T, N, F]
        batch_y:  [B, T, N, F]
        demos_x:  [B, S, K, T, N, F]
        demos_y:  [B, S, K, T, N, F]

    where S = num_prefix_selections, K = num_demonstrations.
    """

    def __init__(self, data, batch_size, input_window, output_window,
                 demo_pool, num_demonstrations=1, num_prefix_selections=1,
                 eval_only=False):
        """
        Args:
            data:  numpy array [T_total, N, F] for this split
            batch_size:  internal batch size
            input_window:  history window length
            output_window:  prediction window length
            demo_pool:  list of (x_np, y_np) pairs from the training split
            num_demonstrations:  K — number of demos per query
            num_prefix_selections:  S — number of independent demo sets
            eval_only:  if True, don't shuffle / drop-last
        """
        # Create sliding windows (same logic as TrafficDataset)
        self.windows = [
            (data[i:i + input_window], data[i + input_window:i + input_window + output_window])
            for i in range(len(data) - input_window - output_window + 1)
        ]
        self.demo_pool = demo_pool
        self.num_demonstrations = num_demonstrations
        self.num_prefix_selections = num_prefix_selections

        # Shuffle + drop_last + pre-batch (mirrors TrafficDataset)
        if not eval_only:
            random.shuffle(self.windows)
            remainder = len(self.windows) % batch_size
            if remainder != 0:
                self.windows = self.windows[:-remainder]

        self.batches = [
            self.windows[i:i + batch_size]
            for i in range(0, len(self.windows), batch_size)
        ]

    def __getitem__(self, idx):
        batch_pairs = self.batches[idx]
        batch_x, batch_y = zip(*batch_pairs)
        batch_x = torch.from_numpy(np.stack(batch_x)).float()  # [B, T, N, F]
        batch_y = torch.from_numpy(np.stack(batch_y)).float()  # [B, T, N, F]

        B = batch_x.shape[0]
        K = self.num_demonstrations
        S = self.num_prefix_selections

        # Sample S independent sets of K demonstrations for each query
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


def define_ict_dataloader(args):
    """Create ICT-aware dataloaders.

    Key differences from define_dataloder():
      - Constructs a demo_pool from the training split
      - Val/test datasets share training split's demo_pool (no data leakage)
      - Returns dataloaders that yield (query_x, query_y, demos_x, demos_y)

    Returns:
        (train_loader, val_loader, test_loader, scaler_dict)
    """
    splits, default_split = load_dataset_splits()
    scaler_dict = {}
    num_nodes_dict = {}
    datasets_train, datasets_val, datasets_test = [], [], []

    for dataset_name in args.dataset_use:
        val_ratio, test_ratio = get_dataset_split(dataset_name, splits, default_split)
        print(f'[ICT] {dataset_name}  val_ratio={val_ratio}  test_ratio={test_ratio}')

        data = load_st_dataset(dataset_name, args)
        num_nodes_dict[dataset_name] = data.shape[1]
        data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
        print(f'[ICT] data_train {data_train.shape}, data_val {data_val.shape}, data_test {data_test.shape}')

        # Normalize using training-split statistics
        if args.real_value == False:
            scaler_data, _, _ = normalize_dataset(data_train, args.input_base_dim)
            data_train[..., :args.input_base_dim] = scaler_data.transform(data_train[:, :, :args.input_base_dim])
            data_val[..., :args.input_base_dim] = scaler_data.transform(data_val[:, :, :args.input_base_dim])
            data_test[..., :args.input_base_dim] = scaler_data.transform(data_test[:, :, :args.input_base_dim])
            scaler_dict[dataset_name] = scaler_data
        else:
            scaler_dict[dataset_name] = None

        # Compute input / output window (same logic as define_dataloder)
        intervel = _compute_interval(dataset_name)
        iw = args.his // (intervel // 5)
        ow = args.pred // (intervel // 5)

        # Build demo pool from training windows
        demo_pool = [
            (data_train[i:i + iw], data_train[i + iw:i + iw + ow])
            for i in range(len(data_train) - iw - ow + 1)
        ]
        print(f'[ICT] demo_pool size={len(demo_pool)}  iw={iw}  ow={ow}')

        # ---- datasets ----
        train_ds = ICTTrafficDataset(
            data_train, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=1, eval_only=False,
        )
        val_ds = ICTTrafficDataset(
            data_val, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=1, eval_only=True,
        )
        test_ds = ICTTrafficDataset(
            data_test, args.batch_size, iw, ow,
            demo_pool, args.num_demonstrations,
            num_prefix_selections=args.num_prefix_selections,
            eval_only=True,
        )
        datasets_train.append(train_ds)
        datasets_val.append(val_ds)
        datasets_test.append(test_ds)

    # Concatenate + DataLoader (same pattern as original)
    train_combine = ConcatDataset(datasets_train)
    val_combine = ConcatDataset(datasets_val)
    test_combine = ConcatDataset(datasets_test)

    train_loader = DataLoader(train_combine, batch_size=1, shuffle=True) if len(train_combine) else None
    val_loader = DataLoader(val_combine, batch_size=1, shuffle=False) if len(val_combine) else None
    test_loader = DataLoader(test_combine, batch_size=1, shuffle=False) if len(test_combine) else None

    args.num_nodes_dict = num_nodes_dict
    return train_loader, val_loader, test_loader, scaler_dict
