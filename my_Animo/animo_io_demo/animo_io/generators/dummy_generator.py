from __future__ import annotations
import numpy as np
from .base import MotionGenerator
from ..types import AniMoLikeOutput, Motion
from ..embedding.base import TextEmbedder
from ..skeletons.base import SkeletonProvider

class DummyMotionGenerator(MotionGenerator):
    """
    I/O形確認用：
      text -> embedder -> RNG seed -> ダミーモーション
    """
    def __init__(self, embedder: TextEmbedder, skeletons: SkeletonProvider):
        self.embedder = embedder
        self.skeletons = skeletons

    def generate(self, text: str, species: str = "dog", T: int = 60) -> AniMoLikeOutput:
        # 1) 骨格（固定情報）
        skeleton = self.skeletons.get(species)
        J = skeleton.J  # ← 関節数はここで確定

        # 2) テキスト埋め込み（shape=[D]）
        emb = self.embedder.embed(f"{species}:{text}")

        # 3) embedding から seed を作って再現性を持たせる
        seed = int(abs(float(emb[0])) * 1e6) % (2**32)
        rng = np.random.default_rng(seed)

        # 4) root_translation: shape=[T,3]
        #    小さな速度を累積してランダムウォークにする
        root_translation = np.cumsum(
            rng.standard_normal((T, 3)).astype(np.float32) * 0.01,
            axis=0
        )

        # 5) joint_quat: shape=[T,J,4]
        #    quaternion は unit length に正規化する
        joint_quat = rng.standard_normal((T, J, 4)).astype(np.float32)
        joint_quat /= (np.linalg.norm(joint_quat, axis=-1, keepdims=True) + 1e-8)

        # 6) foot_contacts: shape=[T,F]（ここではF=2とする）
        foot_contacts = (rng.random((T, 2)) > 0.5).astype(np.float32)

        motion = Motion(
            T=T,
            root_translation=root_translation,
            joint_quat=joint_quat,
            foot_contacts=foot_contacts
        )

        return AniMoLikeOutput(skeleton=skeleton, motion=motion)
