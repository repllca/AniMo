from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class TextEmbedder(ABC):
    """
    テキスト → ベクトル の変換器（後で CLIP/Gemini 等に差し替える）
    """
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        returns:
          embedding: shape = [D]
        """
        raise NotImplementedError
