from __future__ import annotations
import numpy as np
import torch

from .base import MotionGenerator
from ..types import AniMoLikeOutput, Motion
from ..embedding.base import TextEmbedder
from ..skeletons.base import SkeletonProvider
from animo_io.models.animo_lite import AniMoLite

class AniMoLiteGenerator(MotionGenerator):
    """
    AniMoLiteモデルを使って
      text -> (root_translation, joint_quat, foot_contacts)
    を生成し、既存の AniMoLikeOutput 形式で返す。

    注意：
      - ここでは「未学習」でも動くように、そのまま推論可能
      - 学習済み重みがある場合は load して使う
    """
    def __init__(
        self,
        embedder: TextEmbedder,
        skeletons: SkeletonProvider,
        text_dim: int = 256,
        device: str = "cpu",
        weights_path: str | None = None,
        F_feet: int = 2,
    ):
        self.embedder = embedder
        self.skeletons = skeletons
        self.device = torch.device(device)
        self.text_dim = text_dim
        self.F_feet = F_feet
        self._model_cache: dict[int, AniMoLite] = {}  # key=J

        self.weights_path = weights_path

    def _get_model(self, J: int) -> AniMoLite:
        if J in self._model_cache:
            return self._model_cache[J]

        model = AniMoLite(text_dim=self.text_dim, J=J, F_feet=self.F_feet).to(self.device)
        model.eval()

        # もし重みがあるならロード（Jが一致する前提）
        if self.weights_path is not None:
            sd = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(sd, strict=True)

        self._model_cache[J] = model
        return model

    @torch.no_grad()
    def generate(self, text: str, species: str = "dog", T: int = 60) -> AniMoLikeOutput:
        skeleton = self.skeletons.get(species)
        J = skeleton.J

        # text -> embedding (D,)
        emb_np = self.embedder.embed(f"{species}:{text}")
        if emb_np.shape[0] != self.text_dim:
            raise ValueError(f"text embedding dim mismatch: expected {self.text_dim}, got {emb_np.shape[0]}")

        # (B=1,D)
        text_emb = torch.from_numpy(emb_np).float().to(self.device)[None, :]

        model = self._get_model(J)
        out = model(text_emb=text_emb, T=T)

        root_translation = out["root_translation"][0].cpu().numpy().astype(np.float32)  # (T,3)
        joint_quat = out["joint_quat"][0].cpu().numpy().astype(np.float32)              # (T,J,4)
        foot_contacts = out["foot_contacts"][0].cpu().numpy().astype(np.float32)        # (T,F)

        motion = Motion(
            T=T,
            root_translation=root_translation,
            joint_quat=joint_quat,
            foot_contacts=foot_contacts,
        )
        return AniMoLikeOutput(skeleton=skeleton, motion=motion)
