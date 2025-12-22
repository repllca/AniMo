from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: [..., 4]
    return q / (q.norm(dim=-1, keepdim=True) + eps)


class AniMoLite(nn.Module):
    """
    AniMo簡易版：
      text_emb (B,D) -> (B,T,hidden) -> root_vel(T,3), joint_quat(T,J,4), contact(T,F)

    - root_translation は root_vel を積分して作る
    - joint_quat は正規化して unit quaternion にする
    - contact は sigmoid で 0..1 に
    """
    def __init__(self, text_dim: int, J: int, F_feet: int = 2,
                 hidden: int = 256, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.J = J
        self.F_feet = F_feet
        self.hidden = hidden

        # text embedding -> condition vector
        self.text_proj = nn.Linear(text_dim, hidden)

        # time embedding（フレーム位置）
        self.time_embed = nn.Embedding(2048, hidden)  # T<=2048想定（必要なら増やす）

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 出力ヘッド
        self.root_vel_head = nn.Linear(hidden, 3)           # [T,3]
        self.joint_quat_head = nn.Linear(hidden, J * 4)     # [T,J*4]
        self.contact_head = nn.Linear(hidden, F_feet)       # [T,F]

    def forward(self, text_emb: torch.Tensor, T: int) -> dict[str, torch.Tensor]:
        """
        text_emb: (B, D)
        returns dict:
          root_translation: (B,T,3)
          joint_quat: (B,T,J,4)
          foot_contacts: (B,T,F)
        """
        B = text_emb.shape[0]
        device = text_emb.device

        # condition (B,hidden)
        c = self.text_proj(text_emb)

        # 時間インデックス 0..T-1
        t_idx = torch.arange(T, device=device).clamp(max=self.time_embed.num_embeddings - 1)
        te = self.time_embed(t_idx)[None, :, :].expand(B, T, self.hidden)  # (B,T,hidden)

        # condition を各時刻に足す（簡易な条件付け）
        x = te + c[:, None, :]  # (B,T,hidden)

        # transformer
        h = self.tr(x)  # (B,T,hidden)

        # root vel -> integrate
        root_vel = self.root_vel_head(h)  # (B,T,3)
        root_translation = torch.cumsum(root_vel, dim=1)  # (B,T,3)

        # joint quat
        q = self.joint_quat_head(h).view(B, T, self.J, 4)  # (B,T,J,4)
        joint_quat = _quat_normalize(q)

        # contacts
        foot_contacts = torch.sigmoid(self.contact_head(h))  # (B,T,F)

        return {
            "root_translation": root_translation,
            "joint_quat": joint_quat,
            "foot_contacts": foot_contacts,
        }
