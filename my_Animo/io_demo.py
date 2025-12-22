# io_demo.py
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass
class Skeleton:
    joint_names: list[str]        # [J]
    parents: list[int]            # [J] root=-1
    rest_offsets: np.ndarray      # [J,3] float32

@dataclass
class Motion:
    T: int
    root_translation: np.ndarray  # [T,3]
    joint_quat: np.ndarray        # [T,J,4] (x,y,z,w)
    foot_contacts: np.ndarray | None  # [T,F] or None

@dataclass
class AniMoLikeOutput:
    skeleton: Skeleton
    motion: Motion

def text_to_embedding(text: str, dim: int = 256) -> np.ndarray:
    英文/日本語を問わず受け取り、デモ用に 'それっぽい' 埋め込みを作る。
        """
    - 学習なし
    - 同じ text なら毎回同じベクトル（デモが安定）
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def dummy_generate_motion(text: str, species: str = "dog", T: int = 60) -> AniMoLikeOutput:
    # 1) embedding（中身はダミーでOK）
    emb = text_to_embedding(f"{species}:{text}")

    # 2) skeleton（デモ用に固定の超簡易スケルトン）
    joint_names = ["root", "spine", "head", "leg_L", "leg_R"]
    parents     = [-1,      0,       1,      0,      0]
    rest_offsets = np.array([
        [0,0,0],
        [0,0.2,0],
        [0,0.2,0],
        [-0.1,-0.2,0],
        [ 0.1,-0.2,0],
    ], dtype=np.float32)
    skel = Skeleton(joint_names, parents, rest_offsets)

    J = len(joint_names)

    # 3) motion（デモ用に embedding から “なんとなく” 作る）
    rng = np.random.default_rng(int(abs(emb[0])*1e6) % (2**32))
    root_translation = np.cumsum(rng.standard_normal((T,3)).astype(np.float32) * 0.01, axis=0)

    # quaternion: (x,y,z,w) で unit に正規化
    joint_quat = rng.standard_normal((T,J,4)).astype(np.float32)
    joint_quat /= (np.linalg.norm(joint_quat, axis=-1, keepdims=True) + 1e-8)

    # foot contact（任意）：2足だと仮定して [T,2]
    foot_contacts = (rng.random((T,2)) > 0.5).astype(np.float32)

    mot = Motion(T=T, root_translation=root_translation, joint_quat=joint_quat, foot_contacts=foot_contacts)
    return AniMoLikeOutput(skeleton=skel, motion=mot)

if __name__ == "__main__":
    out = dummy_generate_motion("犬が歩く", species="dog", T=120)
    print("J =", len(out.skeleton.joint_names))
    print("T =", out.motion.T)
    print("root_translation:", out.motion.root_translation.shape)  # (T,3)
    print("joint_quat:", out.motion.joint_quat.shape)              # (T,J,4)
    print("foot_contacts:", out.motion.foot_contacts.shape)        # (T,2)
