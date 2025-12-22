# viz/mpl_viewer.py
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from animo_io.kinematics.fk import fk_joint_positions


def build_animation(out, fps: int = 30, stride: int = 1, elev: int = 18, azim: int = -60):
    """
    可視化用アニメーションを作って返す（保存にも再利用できる）
    returns:
      fig, anim
    """
    skel = out.skeleton
    mot = out.motion

    joint_pos = fk_joint_positions(
        parents=skel.parents,
        rest_offsets=skel.rest_offsets,
        root_translation=mot.root_translation,
        joint_quat=mot.joint_quat,
    )

    joint_pos = joint_pos[::stride]
    T, J, _ = joint_pos.shape
    edges = [(j, skel.parents[j]) for j in range(J) if skel.parents[j] != -1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    scat = ax.scatter([], [], [], s=30)
    lines = []
    for _ in edges:
        (ln,) = ax.plot([], [], [], linewidth=2)
        lines.append(ln)

    # 軸範囲固定（見た目安定）
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
    ax.set_title("AniMoLikeOutput Viewer")

    def update(frame_idx: int):
        P = joint_pos[frame_idx]
        scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

        for k, (child, parent) in enumerate(edges):
            a, b = P[parent], P[child]
            lines[k].set_data([a[0], b[0]], [a[1], b[1]])
            lines[k].set_3d_properties([a[2], b[2]])

        return [scat, *lines]

    interval_ms = int(1000 / fps)
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    return fig, anim


def save_gif(anim: FuncAnimation, out_path: str | Path, fps: int = 30):
    """
    GIF保存（ffmpeg不要）
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)


def visualize_animo_output(out, fps: int = 30, stride: int = 1, elev: int = 18, azim: int = -60):
    """
    画面表示だけしたい場合用
    """
    fig, anim = build_animation(out, fps=fps, stride=stride, elev=elev, azim=azim)
    plt.show()
    return anim
