from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from animo_io.types import Skeleton, Motion, AniMoLikeOutput


def load_result_dir(result_dir: str | Path) -> AniMoLikeOutput:
    """
    results/<timestamp>/ を読み込んで AniMoLikeOutput を復元する
    必要ファイル:
      - skeleton.json
      - motion.npz
    """
    d = Path(result_dir)

    # skeleton
    sk = json.loads((d / "skeleton.json").read_text(encoding="utf-8"))
    skeleton = Skeleton(
        joint_names=list(sk["joint_names"]),
        parents=list(sk["parents"]),
        rest_offsets=np.array(sk["rest_offsets"], dtype=np.float32),
    )

    # motion
    npz = np.load(d / "motion.npz", allow_pickle=True)
    T = int(npz["T"])
    root_translation = np.array(npz["root_translation"], dtype=np.float32)
    joint_quat = np.array(npz["joint_quat"], dtype=np.float32)

    foot_contacts = None
    if "foot_contacts" in npz.files:
        foot = npz["foot_contacts"]
        # Noneがobjectで入るケース対策
        if foot.dtype == object:
            foot_contacts = None if foot.item() is None else np.array(foot, dtype=np.float32)
        else:
            foot_contacts = np.array(foot, dtype=np.float32)

    # shape最小検査（壊れた結果を早期に検出）
    if root_translation.shape != (T, 3):
        raise ValueError(f"root_translation shape mismatch: {root_translation.shape} vs (T,3) with T={T}")
    if joint_quat.shape[0] != T or joint_quat.shape[-1] != 4:
        raise ValueError(f"joint_quat shape looks wrong: {joint_quat.shape} (expected (T,J,4))")

    motion = Motion(
        T=T,
        root_translation=root_translation,
        joint_quat=joint_quat,
        foot_contacts=foot_contacts,
    )

    return AniMoLikeOutput(skeleton=skeleton, motion=motion)


def read_meta(result_dir: str | Path) -> dict:
    d = Path(result_dir)
    p = d / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def find_latest_result(root: str | Path = "results") -> Path:
    """
    results/ 配下の最新フォルダを返す（mtimeで安全に判定）
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"results root not found: {root}")

    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no result dirs under: {root}")

    return max(dirs, key=lambda p: p.stat().st_mtime)
