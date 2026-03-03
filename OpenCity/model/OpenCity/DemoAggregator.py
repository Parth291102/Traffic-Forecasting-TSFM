import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDemoAggregator(nn.Module):
    """Lightweight cosine-similarity based demo correction aggregator.

    This class implements Variant A from the plan: query and demo features
    are projected into the same space, normalized, and a per-node
    cosine similarity is used to compute weights over K demonstrations. The
    final correction is a weighted sum of the per-demo residuals followed by
    a learnable scale factor.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.demo_proj = nn.Linear(embed_dim, embed_dim)
        # temperature controls sharpness of softmax over demos
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # global scale on the aggregated correction
        self.correction_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query_feat: torch.Tensor, demo_feats: torch.Tensor,
                corrections: torch.Tensor) -> torch.Tensor:
        """Aggregate demo corrections conditioned on query features.

        Args:
            query_feat:  [B, N, D]
            demo_feats:  [B, K, N, D]
            corrections: [B, K, T, N, 1]
        Returns:
            weighted_corr: [B, T, N, 1]
        """
        # project and normalize
        q_proj = F.normalize(self.query_proj(query_feat), dim=-1)          # [B,N,D]
        d_proj = F.normalize(self.demo_proj(demo_feats), dim=-1)           # [B,K,N,D]

        # cosine similarity over last dim -> [B,K,N]
        # compute by einsum for clarity
        sim = torch.einsum('bnd,bknd->bkn', q_proj, d_proj)
        # softmax over demos
        weights = torch.softmax(sim / self.temperature, dim=1)            # [B,K,N]

        # apply weights to corrections
        # expand to align dims: [B,K,1,N,1]
        w = weights.unsqueeze(2).unsqueeze(-1)
        weighted = (w * corrections).sum(dim=1)                            # [B,T,N,1]
        return self.correction_scale * weighted
