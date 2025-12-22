from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import AniMoLikeOutput

class MotionGenerator(ABC):
    """
    text/species を受けて AniMoLikeOutput を返す generator。
    後でここを AniMo / MoMask / RAC に差し替える。
    """
    @abstractmethod
    def generate(self, text: str, species: str, T: int) -> AniMoLikeOutput:
        raise NotImplementedError
