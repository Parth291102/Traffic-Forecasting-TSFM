"""
Encoder feature extraction for embedding-space KNN retrieval.

Provides utilities to extract encoder-level embeddings from OpenCity model
and perform node-mean pooling for KNN-based demo selection.
"""

import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, Optional


class EncoderFeatureExtractor(nn.Module):
    """Wrapper to extract encoder features from OpenCity model.

    Extracts features at the encoder output (before the prediction head)
    and performs node-mean pooling to get per-sample embeddings.
    """

    def __init__(self, opencity_model: nn.Module):
        """
        Args:
            opencity_model: OpenCity instance with encoder blocks
        """
        super().__init__()
        self.model = opencity_model
        self.device = opencity_model.device
        self.output_dim = opencity_model.output_dim
        self.lap_mx_dict = opencity_model.lap_mx_dict
        self.adj_mx_dict = opencity_model.adj_mx_dict
        self.geo_mask_dict = opencity_model.geo_mask_dict
        self.sem_mask = opencity_model.sem_mask
        self.patch_embedding_flow = opencity_model.patch_embedding_flow
        self.patch_embedding_time = opencity_model.patch_embedding_time
        self.spatial_embedding = opencity_model.spatial_embedding
        self.encoder_blocks = opencity_model.encoder_blocks

    def forward(self, input_hist: torch.Tensor, input_temp_feat: torch.Tensor,
                select_dataset: str) -> torch.Tensor:
        """Extract encoder features with node-mean pooling.

        Args:
            input_hist:       [B, T, N, C+4] — history with flow data + temporal context
                              where C=3 (flow, speed, occupancy) and last 4 are temporal indices
            input_temp_feat:  Same structure as input_hist (used for temporal encoding)
            select_dataset:   str — dataset name to retrieve masks/adjacency

        Returns:
            enc_feat_pooled:  [B, D] — encoder features averaged over nodes and patches
                                        where D=512 (embed_dim)
        """
        bs, time_steps, num_nodes, num_feas = input_hist.size()

        # Ensure we have temporal features - if not, create dummy ones
        if num_feas <= self.output_dim:
            # Add dummy temporal features (zeros will be treated as default temporal indices)
            dummy_temporal = torch.zeros(bs, time_steps, num_nodes, 4, 
                                       dtype=torch.long, device=input_hist.device)
            input_hist = torch.cat([input_hist, dummy_temporal], dim=-1)
            input_temp_feat = torch.cat([input_temp_feat, dummy_temporal], dim=-1)
            num_feas = input_hist.shape[-1]

        # Temporal context encoding
        TCH = input_hist[..., self.output_dim:].long()
        feas_time = self.patch_embedding_time(TCH)

        spa_feas = self.spatial_embedding(
            self.lap_mx_dict[select_dataset].to(self.device)
        ).repeat(bs, feas_time.shape[1], 1, 1)
        feas_time = feas_time + spa_feas

        # Instance normalization
        x_in = input_hist[..., :self.output_dim]
        means = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - means
        stdev = torch.sqrt(
            torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_in /= stdev

        # Patch embedding
        enc = self.patch_embedding_flow(x_in)  # [B, P, N, D]

        # Adjacency
        adj = self.adj_mx_dict[select_dataset].to(self.device)

        # Spatio-Temporal encoder blocks
        for encoder_block in self.encoder_blocks:
            enc = encoder_block(
                enc, enc, enc, feas_time, feas_time, adj,
                self.geo_mask_dict[select_dataset].to(self.device),
                self.sem_mask
            )

        # Node-mean pooling followed by patch-mean pooling
        # enc shape: [B, P, N, D]
        # Node-mean: [B, P, D]
        enc_node_pooled = enc.mean(dim=2)  # mean over nodes (dim 2)
        # Patch-mean: [B, D]
        enc_feat_pooled = enc_node_pooled.mean(dim=1)  # mean over patches (dim 1)

        return enc_feat_pooled


def extract_encoder_features_batch(
    model: nn.Module,
    data_windows: list,
    temporal_features_windows: list,
    select_dataset: str,
    batch_size: int = 32,
    device: torch.device = None,
    enable_extraction: bool = True
) -> Optional[np.ndarray]:
    """Extract encoder features for a batch of demo windows.

    Uses default temporal indices (noon on Monday) to extract semantic traffic patterns.
    Assumption: Encoder features capture semantic similarity irrespective of exact temporal context.

    Args:
        model:                      OpenCity model with encoder
        data_windows:               List of [T, N, F] arrays (demo histories)
        temporal_features_windows:  List of [T, N, F] arrays (temporal features, unused)
        select_dataset:             Dataset name
        batch_size:                 Processing batch size
        device:                     torch device
        enable_extraction:          bool — enable feature extraction

    Returns:
        features:                   [num_demos, 512] ndarray of encoder features, or None if disabled
    """
    if not enable_extraction or model is None:
        return None

    if device is None:
        device = next(model.parameters()).device

    feature_extractor = EncoderFeatureExtractor(model).to(device)
    feature_extractor.eval()

    all_features = []
    num_demos = len(data_windows)

    with torch.no_grad():
        for i in range(0, num_demos, batch_size):
            batch_windows = data_windows[i:i+batch_size]
            batch_size_actual = len(batch_windows)

            try:
                # Stack windows into batch: [B, T, N, F]
                hist_batch = torch.stack([
                    torch.from_numpy(w).float() for w in batch_windows
                ]).to(device)

                # Create default temporal features (noon on Monday = hour:12, dow:0)
                # For each sample: 24-hour cyclic + 7-day cyclic + holiday + (unused)
                B, T, N, _ = hist_batch.shape
                temp_batch = torch.zeros(B, T, N, 4, dtype=torch.long, device=device)
                temp_batch[..., 0] = 12  # hour = 12 (noon)
                temp_batch[..., 1] = 0   # day of week = 0 (Monday)
                temp_batch[..., 2] = 0   # holiday = 0 (not holiday)

                # Concatenate temporal features to history
                hist_with_temp = torch.cat([hist_batch, temp_batch.float()], dim=-1)

                # Extract features
                batch_features = feature_extractor(hist_with_temp, hist_with_temp, select_dataset)
                all_features.append(batch_features.cpu().numpy())

            except Exception as e:
                print(f'[ICT] Batch {i//batch_size} extraction failed: {e}')
                # Skip this batch on error
                continue

    if not all_features:
        return None

    features = np.concatenate(all_features, axis=0)
    return features


def extract_single_feature(
    model: nn.Module,
    data_window: np.ndarray,
    temporal_features_window: np.ndarray,
    select_dataset: str,
    device: torch.device = None
) -> np.ndarray:
    """Extract encoder feature for a single demo window.

    Args:
        model:                      OpenCity model
        data_window:                [T, N, F] array (demo history)
        temporal_features_window:   [T, N, F] array (temporal features)
        select_dataset:             Dataset name
        device:                     torch device

    Returns:
        feature:                    [D] ndarray (512-d embedding)
    """
    if device is None:
        device = next(model.parameters()).device

    feature_extractor = EncoderFeatureExtractor(model).to(device)
    feature_extractor.eval()

    with torch.no_grad():
        # Add batch dimension
        hist_tensor = torch.from_numpy(data_window[np.newaxis]).float().to(device)
        temp_tensor = torch.from_numpy(temporal_features_window[np.newaxis]).float().to(device)

        # Extract feature
        feature = feature_extractor(hist_tensor, temp_tensor, select_dataset)

    return feature.cpu().numpy()[0]  # Remove batch dimension
