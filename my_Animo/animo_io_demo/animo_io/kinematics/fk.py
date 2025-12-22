from __future__ import annotations
import numpy as np
from animo_io.math3d.quat import quat_to_rotmat

def fk_joint_positions(
    parents: list[int],
    rest_offsets: np.ndarray,      # [J,3]
    root_translation: np.ndarray,  # [T,3]
    joint_quat: np.ndarray,        # [T,J,4] (x,y,z,w)
) -> np.ndarray:
    """
    Forward Kinematics:
      global_pos[root] = root_translation[t]
      global_rot[root] = R(q_root)
      global_rot[j]    = global_rot[parent] @ R(q_j)
      global_pos[j]    = global_pos[parent] + global_rot[parent] @ rest_offsets[j]
    returns:
      joint_pos: [T,J,3]
    """
    T, J = joint_quat.shape[0], joint_quat.shape[1]
    rest_offsets = rest_offsets.astype(np.float32)
    root_translation = root_translation.astype(np.float32)
    joint_quat = joint_quat.astype(np.float32)

    R_local = quat_to_rotmat(joint_quat)  # (T,J,3,3)
    joint_pos = np.zeros((T, J, 3), dtype=np.float32)
    R_global = np.zeros((T, J, 3, 3), dtype=np.float32)

    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p == -1:
                R_global[t, j] = R_local[t, j]
                joint_pos[t, j] = root_translation[t]
            else:
                R_global[t, j] = R_global[t, p] @ R_local[t, j]
                joint_pos[t, j] = joint_pos[t, p] + (R_global[t, p] @ rest_offsets[j])
    return joint_pos
