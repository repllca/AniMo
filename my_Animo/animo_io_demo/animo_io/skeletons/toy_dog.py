from __future__ import annotations
import numpy as np
from ..types import Skeleton
from .base import SkeletonProvider

class ToySkeletonProvider(SkeletonProvider):
    """
    とりあえず動くための toy スケルトン定義。
    J は joint_names の長さで確定する。
    """
    def get(self, species: str) -> Skeleton:
        # 今は species を無視して固定。後で dog/cat/horse で分岐してOK。
        joint_names = ["root", "spine", "head", "leg_L", "leg_R"]  # ← J=5
        parents     = [-1,      0,       1,      0,      0]        # ← 長さJ
        rest_offsets = np.array(
            [
                [0.0,  0.0,  0.0],   # root
                [0.0,  0.2,  0.0],   # spine (rootの上)
                [0.0,  0.2,  0.0],   # head  (spineの上)
                [-0.1, -0.2, 0.0],   # leg_L (rootの左下)
                [ 0.1, -0.2, 0.0],   # leg_R (rootの右下)
            ],
            dtype=np.float32
        )  # ← shape=[J,3]

        return Skeleton(joint_names=joint_names, parents=parents, rest_offsets=rest_offsets)
