from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import Skeleton

class SkeletonProvider(ABC):
    """
    種(species)などの条件に応じて Skeleton を返す。
    後で AniMo4D の本物スケルトンへ差し替える場所。
    """
    @abstractmethod
    def get(self, species: str) -> Skeleton:
        raise NotImplementedError
