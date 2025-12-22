# visualize_motion.py
# AniMoLikeOutput (Skeleton + Motion) を 3D でアニメーション表示する最小ビューア
#
# 期待する入力:
#   out.skeleton:
#     - joint_names: [J]
#     - parents:     [J]  (root=-1)
#     - rest_offsets:[J,3]
#   out.motion:
#     - T
#     - root_translation: [T,3]
#     - joint_quat:       [T,J,4]  (x,y,z,w)
#
# 表示方法:
#   FK（順運動学）で joint_pos[T,J,3] を計算し、親子を線で結んで描画。

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection を登録するだけ)
# ----------------------------
# Quaternion utilities
# ----------------------------

def quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """q: (...,4) -> unit quaternion"""
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

    # 3x3
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
    )  # (...,3,3)
    return R

# ----------------------------
# Forward Kinematics (FK)
# ----------------------------

def fk_joint_positions(
    parents: list[int],
    rest_offsets: np.ndarray,     # [J,3]
    root_translation: np.ndarray, # [T,3]
    joint_quat: np.ndarray,       # [T,J,4]
) -> np.ndarray:
    """
    returns:
      joint_pos: [T,J,3]
    ルール:
      global_rot[root] = R(q_root)
      global_pos[root] = root_translation[t]
      global_rot[j]    = global_rot[parent] @ R(q_j)
      global_pos[j]    = global_pos[parent] + global_rot[parent] @ rest_offsets[j]
    """
    T, J = joint_quat.shape[0], joint_quat.shape[1]
    rest_offsets = rest_offsets.astype(np.float32)
    root_translation = root_translation.astype(np.float32)
    joint_quat = joint_quat.astype(np.float32)

    # [T,J,3,3]
    R_local = quat_to_rotmat(joint_quat)

    joint_pos = np.zeros((T, J, 3), dtype=np.float32)
    R_global = np.zeros((T, J, 3, 3), dtype=np.float32)

    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p == -1:
                # root
                R_global[t, j] = R_local[t, j]
                joint_pos[t, j] = root_translation[t]
            else:
                # child
                R_global[t, j] = R_global[t, p] @ R_local[t, j]
                joint_pos[t, j] = joint_pos[t, p] + (R_global[t, p] @ rest_offsets[j])
    return joint_pos

# ----------------------------
# Visualization
# ----------------------------

def visualize_animo_output(out, fps: int = 30, stride: int = 1, elev: int = 18, azim: int = -60):
    """
    out: AniMoLikeOutput (あなたの types.AniMoLikeOutput)
    fps: 表示FPS（間引き stride と合わせて調整）
    stride: フレーム間引き（例:2なら半分のフレームで再生）
    """
    skel = out.skeleton
    mot = out.motion

    parents = skel.parents
    rest_offsets = skel.rest_offsets
    root_translation = mot.root_translation
    joint_quat = mot.joint_quat

    # FKで [T,J,3] を作る（描画に必要）
    joint_pos = fk_joint_positions(parents, rest_offsets, root_translation, joint_quat)

    # 間引き
    joint_pos = joint_pos[::stride]
    T, J, _ = joint_pos.shape

    # 親子辺リスト
    edges = [(j, parents[j]) for j in range(J) if parents[j] != -1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # 描画オブジェクト（点と線）
    scat = ax.scatter([], [], [], s=30)

    lines = []
    for _ in edges:
        (ln,) = ax.plot([], [], [], linewidth=2)
        lines.append(ln)

    # 軸範囲を固定（ガタガタしない）
    # 全フレームの min/max を見て少し余白をつける
    mn = joint_pos.reshape(-1, 3).min(axis=0)
    mx = joint_pos.reshape(-1, 3).max(axis=0)
    center = (mn + mx) / 2.0
    span = (mx - mn).max()
    if span < 1e-6:
        span = 1.0
    half = span * 0.6

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("AniMoLikeOutput Viewer (Skeleton Animation)")

    # アニメ更新関数
    def update(frame_idx: int):
        P = joint_pos[frame_idx]  # [J,3]

        # scatter更新
        scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

        # line更新（親子を結ぶ）
        for k, (child, parent) in enumerate(edges):
            a = P[parent]
            b = P[child]
            lines[k].set_data([a[0], b[0]], [a[1], b[1]])
            lines[k].set_3d_properties([a[2], b[2]])

        return [scat, *lines]

    interval_ms = int(1000 / fps)
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    plt.show()
    return anim

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # 例: 既存の demo と同じ生成器で out を作って表示する場合
    # （あなたの環境に合わせて import を直してください）
    from animo_io.embedding.hash_embedder import HashTextEmbedder
    from animo_io.skeletons.toy_dog import ToySkeletonProvider
    # どちらでもOK:
    # from animo_io.generators.dummy_generator import DummyMotionGenerator
    from animo_io.generators.animo_lite_generator import AniMoLiteGenerator

    embedder = HashTextEmbedder(dim=256)
    skels = ToySkeletonProvider()

    gen = AniMoLiteGenerator(
        embedder=embedder,
        skeletons=skels,
        text_dim=256,
        device="cuda",      # GPUなら "cuda"
        weights_path=None, # 学習済みがあればパス
        F_feet=2
    )

    out = gen.generate("犬が歩く / a dog is walking", species="dog", T=180)

    # 可視化
    visualize_animo_output(out, fps=30, stride=1)
