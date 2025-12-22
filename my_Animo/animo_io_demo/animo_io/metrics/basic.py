from __future__ import annotations
import numpy as np
from animo_io.kinematics.fk import fk_joint_positions


def quat_norm_error(joint_quat: np.ndarray) -> float:
    """
    joint_quat: (T,J,4)
    unit quaternion からのズレの平均
    """
    n = np.linalg.norm(joint_quat, axis=-1)  # (T,J)
    return float(np.mean(np.abs(n - 1.0)))


def root_speed_stats(root_translation: np.ndarray, fps: float = 30.0) -> dict:
    """
    root_translation: (T,3)
    return:
      mean_speed, max_speed (units/sec)
    """
    v = np.diff(root_translation, axis=0) * fps  # (T-1,3)
    speed = np.linalg.norm(v, axis=-1)
    return {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)) if speed.size else 0.0,
    }


def bone_length_consistency(out) -> dict:
    """
    FKして各ボーン長の分散を見る（0に近いほど良い）
    """
    sk = out.skeleton
    mot = out.motion
    pos = fk_joint_positions(sk.parents, sk.rest_offsets, mot.root_translation, mot.joint_quat)  # (T,J,3)

    lens = []
    for j, p in enumerate(sk.parents):
        if p == -1:
            continue
        d = pos[:, j] - pos[:, p]          # (T,3)
        lens.append(np.linalg.norm(d, axis=-1))  # (T,)
    if not lens:
        return {"bone_len_std_mean": 0.0}

    L = np.stack(lens, axis=1)  # (T, num_bones)
    return {"bone_len_std_mean": float(np.mean(np.std(L, axis=0)))}
