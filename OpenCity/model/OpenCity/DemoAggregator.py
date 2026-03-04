"""
Learned Demo Aggregation modules for ICT.

Instead of naive averaging of demo residuals, these modules learn to
weight each demo's correction based on query-demo feature similarity.

Variant A — SimpleDemoAggregator:
    Cosine similarity + learnable temperature → per-node demo weights.

Variant B — DemoAggregator:
    Cross-attention over demo corrections with query as context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDemoAggregator(nn.Module):
    """Learns per-node demo weights via cosine similarity.

    Computes:
        weights = softmax(cos_sim(proj_q, proj_d) / tau)  [B, K, N]
        correction = scale * sum_k(weights_k * correction_k)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.demo_proj = nn.Linear(embed_dim, embed_dim)
        self.temperature = nn.Parameter(torch.ones(1))
        self.correction_scale = nn.Parameter(torch.ones(1))

    def forward(self, query_feat, demo_feats, corrections):
        """
        Args:
            query_feat:  [B, N, D]        - mean-pooled encoder output for query
            demo_feats:  [B, K, N, D]     - mean-pooled encoder outputs for demos
            corrections: [B, K, T, N, 1]  - per-demo residual corrections

        Returns:
            weighted_correction: [B, T, N, 1]
        """
        # Project features
        q = F.normalize(self.query_proj(query_feat), dim=-1)   # [B, N, D]
        d = F.normalize(self.demo_proj(demo_feats), dim=-1)    # [B, K, N, D]

        # Cosine similarity per node: [B, K, N]
        similarity = torch.einsum('bnd,bknd->bkn', q, d)

        # Softmax over K with learned temperature
        tau = self.temperature.abs().clamp(min=0.1)
        weights = F.softmax(similarity / tau, dim=1)  # [B, K, N]

        # Weighted sum of corrections: [B, K, N] * [B, K, T, N, 1] → sum over K
        # Reshape weights to [B, K, 1, N, 1] for broadcasting
        w = weights.unsqueeze(2).unsqueeze(-1)         # [B, K, 1, N, 1]
        weighted = (w * corrections).sum(dim=1)        # [B, T, N, 1]

        return weighted * self.correction_scale


class DemoAggregator(nn.Module):
    """Multi-head cross-attention aggregator for demo corrections.

    Follows standard cross-attention: Q from query, K from demos, attention
    weights applied to corrections (the "values"). Multi-head lets different
    heads learn different demo-weighting strategies; a learned linear
    combination merges H candidate corrections into one.

    Pipeline:
        1. LayerNorm → Q/K projections (D → H * hd)
        2. Multi-head dot-product attention → [B, N, H, K]
        3. Per-head weighted corrections → [B, T, N, H]
        4. Head combination: Linear(H → 1)  → [B, T, N, 1]
        5. Per-node softplus scale          → [B, T, N, 1]

    ~198K params (D=512, H=4, proj_dim=128, hd=32)
    """

    def __init__(self, embed_dim, num_heads=4, proj_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim or embed_dim // 4
        self.head_dim = self.proj_dim // num_heads
        assert self.proj_dim % num_heads == 0, \
            f'proj_dim ({self.proj_dim}) must be divisible by num_heads ({num_heads})'
        self.attn_scale = self.head_dim ** -0.5

        # 1. Pre-norm + Q/K projections (D → proj_dim, split into H heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, self.proj_dim)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim)

        # 2. Head combination: merge H candidate corrections → 1
        #    Init to 1/H so output starts as head-average (≈ single-head baseline)
        self.head_combine = nn.Linear(num_heads, 1, bias=False)
        nn.init.constant_(self.head_combine.weight, 1.0 / num_heads)

        # 3. Per-node scale: softplus(MLP(query_feat)) — can amplify, always ≥ 0
        #    Zero-init last layer: softplus(0) ≈ 0.693 → gentle start
        self.scale_net = nn.Sequential(
            nn.Linear(embed_dim, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, 1),
        )
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feat, demo_feats, corrections):
        """
        Args:
            query_feat:  [B, N, D]
            demo_feats:  [B, K, N, D]
            corrections: [B, K, T, N, 1]

        Returns:
            aggregated_correction: [B, T, N, 1]
        """
        B, K, T, N, _ = corrections.shape
        H = self.num_heads
        hd = self.head_dim

        # ---- 1. Multi-head Q·K attention ----
        q = self.q_proj(self.norm(query_feat))            # [B, N, proj_dim]
        k = self.k_proj(self.norm(demo_feats))            # [B, K, N, proj_dim]

        q = q.view(B, N, H, hd)                           # [B, N, H, hd]
        k = k.view(B, K, N, H, hd)                        # [B, K, N, H, hd]

        # Per-node, per-head attention over K demos
        attn = torch.einsum('bnhd,bknhd->bnhk', q, k)    # [B, N, H, K]
        attn = attn * self.attn_scale
        attn = F.softmax(attn, dim=-1)                     # [B, N, H, K]
        attn = self.dropout(attn)

        # ---- 2. Per-head weighted corrections ----
        # attn: [B, N, H, K] → [B, K, 1, N, H]  (for broadcasting)
        w = attn.permute(0, 3, 1, 2).unsqueeze(2)         # [B, K, 1, N, H]
        # corrections: [B, K, T, N, 1] broadcasts with w → [B, K, T, N, H]
        weighted = (w * corrections).sum(dim=1)            # [B, T, N, H]

        # ---- 3. Combine heads → single correction ----
        out = self.head_combine(weighted)                  # [B, T, N, 1]

        # ---- 4. Per-node scale (softplus, can amplify) ----
        scale = F.softplus(self.scale_net(query_feat))     # [B, N, 1]
        out = out * scale.unsqueeze(1)                     # [B, T, N, 1]

        return out
