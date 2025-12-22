# animo_io/models/animo_lite.py
from __future__ import annotations

import torch
import torch.nn as nn


def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q: (...,4)
    return: (...,4) unit quaternion
    """
    return q / (q.norm(dim=-1, keepdim=True) + eps)


class AniMoLite(nn.Module):
    """
    AniMo の「形だけ」簡易版（学習可能な最小モデル）

    入力:
      - text_emb: (B, D)  テキスト埋め込み（例: D=256）
      - T: int            フレーム数

    出力（dict）:
      - root_translation: (B, T, 3)
      - joint_quat:       (B, T, J, 4)  quaternion (x,y,z,w) unit
      - foot_contacts:    (B, T, F)     0..1 (sigmoid)

    実装方針:
      - text_emb を hidden 次元に射影して条件にする
      - time embedding を足して TransformerEncoder へ
      - root は "速度→積分" で位置にする（滑らかになりやすい）
      - joint_quat は unit quaternion に正規化
      - foot_contacts は sigmoid

    互換性:
      - 古い torch でも動くよう batch_first を使わず、
        Transformer は (T,B,H) 形式で通す
    """

    def __init__(
        self,
        text_dim: int,
        J: int,
        F_feet: int = 2,
        hidden: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_T: int = 2048,
        activation: str = "gelu",  # 古いtorchでダメなら "relu" に変える
    ):
        super().__init__()
        self.J = int(J)
        self.F_feet = int(F_feet)
        self.hidden = int(hidden)
        self.max_T = int(max_T)

        # text embedding -> condition vector (B,H)
        self.text_proj = nn.Linear(text_dim, hidden)

        # frame index -> time embedding (T,H)
        self.time_embed = nn.Embedding(max_T, hidden)

        # TransformerEncoder（torch古め互換のため batch_first は使わない）
        # activation="gelu" が通らない場合があるので、落ちたら "relu" に変更してください
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation=activation,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Heads
        self.root_vel_head = nn.Linear(hidden, 3)          # (T,B,3) after TR
        self.joint_quat_head = nn.Linear(hidden, J * 4)    # (T,B,J*4)
        self.contact_head = nn.Linear(hidden, F_feet)      # (T,B,F)

    def forward(self, text_emb: torch.Tensor, T: int) -> dict[str, torch.Tensor]:
        """
        text_emb: (B,D)
        T: int

        returns:
          dict with:
            root_translation: (B,T,3)
            joint_quat: (B,T,J,4)
            foot_contacts: (B,T,F)
        """
        if text_emb.dim() != 2:
            raise ValueError(f"text_emb must be (B,D), got shape={tuple(text_emb.shape)}")
        B = text_emb.shape[0]
        device = text_emb.device

        if T <= 0:
            raise ValueError("T must be > 0")
        if T > self.max_T:
            # embedding範囲を超えると落ちるので、ここは明示的に止める
            raise ValueError(f"T={T} exceeds max_T={self.max_T}. Increase max_T in AniMoLite.")

        # condition (B,H)
        c = self.text_proj(text_emb)

        # time embeddings (T,H)
        t_idx = torch.arange(T, device=device)
        te = self.time_embed(t_idx)  # (T,H)

        # TR入力 (T,B,H) を作る
        # te: (T,H) -> (T,B,H)
        x = te[:, None, :].expand(T, B, self.hidden)  # (T,B,H)
        x = x + c[None, :, :]                         # (T,B,H)

        # Transformer
        h = self.tr(x)  # (T,B,H)

        # root velocity -> integrate to translation
        root_vel = self.root_vel_head(h)              # (T,B,3)
        root_translation = torch.cumsum(root_vel, dim=0)  # (T,B,3)

        # joint quat
        q = self.joint_quat_head(h)                   # (T,B,J*4)
        q = q.view(T, B, self.J, 4)                   # (T,B,J,4)
        joint_quat = _quat_normalize(q)               # (T,B,J,4)

        # contacts
        foot_contacts = torch.sigmoid(self.contact_head(h))  # (T,B,F)

        # (T,B,*) -> (B,T,*)
        root_translation = root_translation.transpose(0, 1).contiguous()  # (B,T,3)
        joint_quat = joint_quat.transpose(0, 1).contiguous()              # (B,T,J,4)
        foot_contacts = foot_contacts.transpose(0, 1).contiguous()        # (B,T,F)

        return {
            "root_translation": root_translation,
            "joint_quat": joint_quat,
            "foot_contacts": foot_contacts,
        }
