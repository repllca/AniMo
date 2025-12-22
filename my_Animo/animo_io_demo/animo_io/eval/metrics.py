from __future__ import annotations
import numpy as np
from animo_io.kinematics.fk import fk_joint_positions
from animo_io.math3d.quat import quat_to_rotmat

def _finite_diff(x: np.ndarray) -> np.ndarray:
    # x: [T,...] -> [T-1,...]
    return x[1:] - x[:-1]

def root_speed_stats(root_translation: np.ndarray, dt: float = 1.0) -> dict:
    v = _finite_diff(root_translation) / dt  # [T-1,3]
    speed = np.linalg.norm(v, axis=-1)       # [T-1]
    return {
        "root_speed_mean": float(speed.mean()),
        "root_speed_std": float(speed.std()),
        "root_speed_max": float(speed.max()) if speed.size else 0.0,
    }

def joint_angvel_stats(joint_quat: np.ndarray, dt: float = 1.0) -> dict:
    """
    ざっくり角速度: R_t と R_{t+1} の相対回転角を使う
    joint_quat: [T,J,4]
    """
    R = quat_to_rotmat(joint_quat)  # [T,J,3,3]
    # relative rotation: R_rel = R_t^T R_{t+1}
    Rt = R[:-1]
    Rt1 = R[1:]
    Rrel = np.einsum("tjab,tjbc->tjac", np.transpose(Rt, (0,1,3,2)), Rt1)  # [T-1,J,3,3]
    # 回転角: arccos((trace-1)/2)
    tr = np.clip((np.trace(Rrel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)  # [T-1,J]
    ang = np.arccos(tr)  # [T-1,J]
    angvel = ang / dt
    return {
        "joint_angvel_mean": float(angvel.mean()),
        "joint_angvel_std": float(angvel.std()),
        "joint_angvel_max": float(angvel.max()) if angvel.size else 0.0,
    }

def foot_slip(joint_pos: np.ndarray, foot_idx: int, foot_contacts: np.ndarray | None = None) -> dict:
    """
    foot_idx: 足の関節index
    joint_pos: [T,J,3]
    foot_contacts: [T] or [T,1] or None
    """
    p = joint_pos[:, foot_idx, :]  # [T,3]
    dp = np.linalg.norm(_finite_diff(p), axis=-1)  # [T-1]
    if foot_contacts is None:
        return {"foot_slip_mean": float(dp.mean()), "foot_slip_max": float(dp.max()) if dp.size else 0.0}

    c = foot_contacts.reshape(-1)  # [T]
    c = c.astype(np.float32)
    # 接地判定: >0.5
    mask = c[:-1] > 0.5
    if mask.sum() == 0:
        return {"foot_slip_contact_mean": 0.0, "foot_slip_contact_max": 0.0}

    slip = dp[mask]
    return {"foot_slip_contact_mean": float(slip.mean()), "foot_slip_contact_max": float(slip.max())}
