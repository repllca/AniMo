from __future__ import annotations
import argparse
import json

from animo_io.results.load import load_result_dir, find_latest_result
from animo_io.eval.metrics import root_speed_stats, joint_angvel_stats
from animo_io.kinematics.fk import fk_joint_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()

    if args.latest:
        d = find_latest_result("results")
    elif args.dir:
        d = args.dir
    else:
        raise SystemExit("Use --latest or --dir")

    out = load_result_dir(d)

    metrics = {}
    metrics.update(root_speed_stats(out.motion.root_translation))
    metrics.update(joint_angvel_stats(out.motion.joint_quat))

    joint_pos = fk_joint_positions(
        out.skeleton.parents, out.skeleton.rest_offsets,
        out.motion.root_translation, out.motion.joint_quat
    )
    # toy_dog の足indexが分かるならここで foot_slip も入れる（後で）

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
