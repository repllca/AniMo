
# motion_decoder.py
import torch
import torch.nn as nn

from config import AniMoShapeConfig


class SimpleRVQMotionDecoder3D(nn.Module):
    """
    入力:
        tokens: [B,T,Q]

    出力:
        motion_3d: [B,T,J,3]
    """

    def __init__(self, cfg: AniMoShapeConfig, d_code: int = 128):
        super().__init__()
        self.cfg = cfg
        self.d_code = d_code

        Q = cfg.num_quantizers
        K = cfg.codebook_size

        # codebook embeddings
        self.codebook = nn.Embedding(Q * K, d_code)

        # MLP for per-frame feature -> motion_dim (J*3)
        self.proj = nn.Sequential(
            nn.Linear(d_code, d_code),
            nn.ReLU(),
            nn.Linear(d_code, cfg.motion_dim),
        )

        offsets = torch.arange(Q) * K
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, Q = tokens.shape
        device = tokens.device

        offsets = self.offsets.to(device).view(1,1,Q)
        indices = tokens + offsets      # [B,T,Q]

        emb = self.codebook(indices)    # [B,T,Q,Dc]
        feat = emb.sum(dim=2)           # [B,T,Dc]

        motion_flat = self.proj(feat)   # [B,T,J*3]

        J = cfg.num_joints
        motion_3d = motion_flat.view(B, T, J, 3)
        return motion_3d
