from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Skeleton:
    """
    スケルトン（固定情報）
      - joint_names: 関節名の配列。長さ = J
      - parents: 親関節 index の配列。長さ = J（root の親は -1）
      - rest_offsets: 親→子の初期オフセット。shape = [J, 3]
    """
    joint_names: list[str]        # [J]
    parents: list[int]            # [J]
    rest_offsets: np.ndarray      # [J,3] float32

    @property
    def J(self) -> int:
        return len(self.joint_names)

@dataclass
class Motion:
    """
    モーション（時系列情報）
      - T: フレーム数
      - root_translation: root のワールド位置。shape = [T, 3]
      - joint_quat: 各関節の回転(Quaternion)。shape = [T, J, 4] (x,y,z,w)
      - foot_contacts: 接地フラグなど（任意）。shape = [T, F] or None
    """
    T: int
    root_translation: np.ndarray  # [T,3]
    joint_quat: np.ndarray        # [T,J,4]
    foot_contacts: np.ndarray | None  # [T,F] or None

@dataclass
class AniMoLikeOutput:
    """
    最終出力：骨格 + 時系列
    """
    skeleton: Skeleton
    motion: Motion
