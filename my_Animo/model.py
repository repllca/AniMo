
# model.py
import torch
import torch.nn as nn

from config import AniMoShapeConfig


class MinimalAniMoLike(nn.Module):
    """
    入力:
        text_ids : [B, L]
        species_ids : [B]
        attr_ids : [B]

    出力:
        logits : [B, T, Q, K]
    """

    def __init__(self, cfg: AniMoShapeConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model

        # === Embeddings ====== #
        self.text_emb = nn.Embedding(cfg.vocab_size, d_model, padding_idx=cfg.pad_id)
        self.species_emb = nn.Embedding(cfg.num_species, d_model)
        self.attr_emb = nn.Embedding(cfg.num_attrs, d_model)
        self.pos_emb = nn.Embedding(cfg.max_motion_len, d_model)

        # === Transformer ===== #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.n_heads,
            dim_feedforward=d_model * 4,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # === Output projection (Q * K classification) === #
        self.out_proj = nn.Linear(
            d_model,
            cfg.num_quantizers * cfg.codebook_size,
        )

    def forward(
        self,
        text_ids: torch.LongTensor,    # [B, L]
        species_ids: torch.LongTensor, # [B]
        attr_ids: torch.LongTensor,    # [B]
        T: int = None,
    ) -> torch.Tensor:

        cfg = self.cfg
        device = text_ids.device

        if T is None:
            T = cfg.max_motion_len

        B, L = text_ids.shape

        # === Text conditioning === #
        text_emb = self.text_emb(text_ids)    # [B,L,D]
        text_feat = text_emb.mean(dim=1)      # [B,D]

        sp = self.species_emb(species_ids)    # [B,D]
        at = self.attr_emb(attr_ids)          # [B,D]
        cond = text_feat + sp + at            # [B,D]

        # === Motion query === #
        pos_ids = torch.arange(T, device=device)
        pos = self.pos_emb(pos_ids)              # [T,D]
        pos = pos.unsqueeze(0).expand(B, T, -1)  # [B,T,D]

        x = pos + cond.unsqueeze(1)              # [B,T,D]

        x = x.transpose(0,1)   # [T,B,D]
        h = self.encoder(x)    # [T,B,D]
        h = h.transpose(0,1)   # [B,T,D]

        logits_flat = self.out_proj(h)  # [B,T,Q*K]

        Q = cfg.num_quantizers
        K = cfg.codebook_size
        logits = logits_flat.view(B, T, Q, K)
        return logits
