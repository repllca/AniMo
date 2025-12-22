# scripts/replay.py
from __future__ import annotations
import argparse

from animo_io.results.load import load_result_dir, find_latest_result
from viz.mpl_viewer import visualize_animo_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="results/<timestamp> directory")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    if args.latest:
        d = find_latest_result("results")
    elif args.dir is not None:
        d = args.dir
    else:
        raise SystemExit("Use --latest or --dir")

    out = load_result_dir(d)
    print("[INFO] replay:", d)
    visualize_animo_output(out, fps=args.fps, stride=args.stride)


if __name__ == "__main__":
    main()
