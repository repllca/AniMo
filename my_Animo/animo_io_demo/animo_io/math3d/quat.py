from __future__ import annotations
import numpy as np

def quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """q: (...,4) quaternion (x,y,z,w) -> unit quaternion"""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (n + eps)

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    q: (...,4) quaternion (x,y,z,w)
    return: (...,3,3) rotation matrix
    """
    q = quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    R = np.stack(
        [
            np.stack([m00, m01, m02], axis=-1),
            np.stack([m10, m11, m12], axis=-1),
            np.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )
    return R
