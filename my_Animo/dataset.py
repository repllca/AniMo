# dataset.py
import random
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from config import AniMoShapeConfig


class ToyAniMoTokenDataset(Dataset):
    """
    デモ用ダミーデータ:
      text_ids     : [max_text_len]
      species_id   : []
      attr_id      : []
      target_tokens: [T, Q]
    """

    def __init__(self, cfg: AniMoShapeConfig, size: int = 1000):
        self.cfg = cfg
        self.size = size

    def __len__(self):
        return self.size

    def _random_text(self) -> torch.LongTensor:
        L = random.randint(1, self.cfg.max_text_len)
        ids = [random.randint(1, self.cfg.vocab_size - 1) for _ in range(L)]
        if L < self.cfg.max_text_len:
            ids += [self.cfg.pad_id] * (self.cfg.max_text_len - L)
        return torch.tensor(ids, dtype=torch.long)

    def _random_tokens(self) -> torch.LongTensor:
        T = self.cfg.max_motion_len
        Q = self.cfg.num_quantizers
        K = self.cfg.codebook_size
        return torch.randint(low=0, high=K, size=(T, Q), dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_ids = self._random_text()          # [max_text_len]
        species_id = torch.randint(0, self.cfg.num_species, (1,), dtype=torch.long)[0]
        attr_id = torch.randint(0, self.cfg.num_attrs, (1,), dtype=torch.long)[0]
        tokens = self._random_tokens()          # [T, Q]
        return text_ids, species_id, attr_id, tokens


def collate_fn(batch):
    text_list, sp_list, at_list, tok_list = zip(*batch)
    text_ids = torch.stack(text_list, dim=0)        # [B, L]
    species_ids = torch.stack(sp_list, dim=0)       # [B]
    attr_ids = torch.stack(at_list, dim=0)          # [B]
    target_tokens = torch.stack(tok_list, dim=0)    # [B, T, Q]
    return text_ids, species_ids, attr_ids, target_tokens


def make_dataloader(cfg: AniMoShapeConfig, batch_size: int = 8, size: int = 1000, shuffle: bool = True):
    dataset = ToyAniMoTokenDataset(cfg, size=size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader
