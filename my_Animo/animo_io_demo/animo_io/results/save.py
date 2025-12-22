from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def create_result_dir(root: str = "results") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(root) / ts
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_input_text(out_dir: Path, text: str, species: str):
    (out_dir / "input.txt").write_text(
        f"species: {species}\ntext: {text}\n",
        encoding="utf-8"
    )


def save_skeleton(out_dir: Path, skeleton):
    data = {
        "joint_names": skeleton.joint_names,
        "parents": skeleton.parents,
        "rest_offsets": skeleton.rest_offsets.tolist(),
    }
    with open(out_dir / "skeleton.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_motion(out_dir: Path, motion):
    np.savez(
        out_dir / "motion.npz",
        T=motion.T,
        root_translation=motion.root_translation,
        joint_quat=motion.joint_quat,
        foot_contacts=motion.foot_contacts,
    )


def save_meta(out_dir: Path, meta: dict):
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
