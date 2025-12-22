from __future__ import annotations
import hashlib
import numpy as np
from .base import TextEmbedder

class HashTextEmbedder(TextEmbedder):
    """
    日本語/英語どちらでも受け取り可能なダミー埋め込み。
    同じ text は同じ embedding（再現性）になるようにする。
    """
    def __init__(self, dim: int = 256):
        self.dim = dim  # ← 埋め込み次元 D はここで定義

    def embed(self, text: str) -> np.ndarray:
        # text -> hash -> seed（再現性のある乱数）
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)

        v = rng.standard_normal(self.dim).astype(np.float32)  # shape=[D]
        v /= (np.linalg.norm(v) + 1e-8)  # 正規化して安定化
        return v
