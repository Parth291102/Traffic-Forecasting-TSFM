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
    """Cross-attention aggregator for demo corrections.

    Uses multi-head attention where the query representation attends to
    demo representations, and the attended output modulates corrections.
    """

    def __init__(self, embed_dim, num_heads=4, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.hidden_dim = hidden_dim or 2 * embed_dim
        self.scale = self.head_dim ** -0.5

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # Encode scalar corrections to embed_dim for values
        self.correction_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output layers
        self.out_proj = nn.Linear(embed_dim, 1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
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
        D = self.embed_dim
        H = self.num_heads
        hd = self.head_dim

        # --- Per-node cross-attention: query attends to K demos ---
        # Q from query: [B, N, D] → [B, N, H, hd]
        Q = self.q_proj(query_feat).view(B, N, H, hd)      # [B, N, H, hd]

        # K from demos: [B, K, N, D] → [B, K, N, H, hd]
        K_attn = self.k_proj(demo_feats).view(B, K, N, H, hd)

        # Attention scores: [B, N, H, 1, hd] @ [B, N, H, hd, K] → [B, N, H, 1, K]
        Q_exp = Q.permute(0, 1, 2, 3).unsqueeze(3)          # [B, N, H, 1, hd]
        K_exp = K_attn.permute(0, 2, 3, 4, 1)               # [B, N, H, hd, K]
        attn = torch.matmul(Q_exp, K_exp).squeeze(3)         # [B, N, H, K]
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)                       # [B, N, H, K]
        attn = self.dropout(attn)

        # Encode corrections for values: [B, K, T, N, 1] → [B, K, T, N, D]
        corr_encoded = self.correction_encoder(corrections)
        # Mean over T to get per-demo-per-node value: [B, K, N, D]
        corr_mean = corr_encoded.mean(dim=2)
        V = self.v_proj(corr_mean).view(B, K, N, H, hd)

        # Weighted sum: [B, N, H, K] @ [B, N, H, K, hd] → [B, N, H, hd]
        V_exp = V.permute(0, 2, 3, 4, 1)                    # [B, N, H, hd, K]
        V_for_attn = V.permute(0, 2, 3, 1, 4)               # [B, N, H, K, hd]
        out = torch.matmul(attn.unsqueeze(3), V_for_attn)    # [B, N, H, 1, hd]
        out = out.squeeze(3)                                  # [B, N, H, hd]
        out = out.reshape(B, N, D)                            # [B, N, D]

        # Norm + FFN
        out = self.norm1(out)
        out = self.norm2(out + self.ffn(out))

        # Project to scalar correction weight per node: [B, N, 1]
        gate = torch.sigmoid(self.out_proj(out))              # [B, N, 1]

        # Apply gate to the attention-weighted correction mean
        # Also reuse the attention weights to weight the actual corrections
        # attn: [B, N, H, K] → average over heads → [B, N, K]
        attn_avg = attn.mean(dim=2)                           # [B, N, K]

        # Weight corrections: [B, N, K] → [B, K, N] → [B, K, 1, N, 1]
        w = attn_avg.permute(0, 2, 1).unsqueeze(2).unsqueeze(-1)  # [B, K, 1, N, 1]
        weighted = (w * corrections).sum(dim=1)               # [B, T, N, 1]

        # Apply learned gate
        return weighted * gate.unsqueeze(1)                   # [B, T, N, 1]
