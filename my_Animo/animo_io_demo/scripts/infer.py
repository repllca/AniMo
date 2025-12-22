# scripts/infer.py
from __future__ import annotations
import argparse
from pathlib import Path

from animo_io.embedding.hash_embedder import HashTextEmbedder
from animo_io.skeletons.toy_dog import ToySkeletonProvider
from animo_io.generators.animo_lite_generator import AniMoLiteGenerator

from animo_io.results.save import (
    create_result_dir,
    save_input_text,
    save_skeleton,
    save_motion,
    save_meta,
)

from animo_io.results.load import (
    load_result_dir,
    find_latest_result,
    read_meta,
)

from viz.mpl_viewer import visualize_animo_output, save_gif


def build_generator(device: str, text_dim: int = 256, weights_path: str | None = None):
    embedder = HashTextEmbedder(dim=text_dim)
    skels = ToySkeletonProvider()
    return AniMoLiteGenerator(
        embedder=embedder,
        skeletons=skels,
        text_dim=text_dim,
        device=device,
        weights_path=weights_path,
        F_feet=2,
    )


def print_summary(out, text: str | None = None, species: str | None = None):
    print("=== Inference Output ===")
    if text is not None:
        print("text:", text)
    if species is not None:
        print("species:", species)
    print("J =", out.skeleton.J)
    print("T =", out.motion.T)
    print("root_translation:", out.motion.root_translation.shape)  # (T,3)
    print("joint_quat:", out.motion.joint_quat.shape)              # (T,J,4)
    if out.motion.foot_contacts is not None:
        print("foot_contacts:", out.motion.foot_contacts.shape)    # (T,F)


def save_result(out, text: str, species: str, device: str, T: int, text_dim: int = 256, root: str = "results") -> Path:
    result_dir = create_result_dir(root)
    save_input_text(result_dir, text, species)
    save_skeleton(result_dir, out.skeleton)
    save_motion(result_dir, out.motion)
    save_meta(
        result_dir,
        {
            "generator": "AniMoLite",
            "device": device,
            "T": T,
            "text_dim": text_dim,
            "note": "demo inference",
        },
    )
    print(f"[INFO] result saved to: {result_dir}")
    return result_dir


def main():
    p = argparse.ArgumentParser()
    # 推論モード
    p.add_argument("--text", type=str, default=None, help="Input text (Japanese/English OK)")
    p.add_argument("--species", type=str, default="dog")
    p.add_argument("--T", type=int, default=120)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--weights", type=str, default=None)

    # 保存/可視化
    p.add_argument("--save", action="store_true", help="Save result into results/<timestamp>/")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)

    # ロードモード（推論せず過去結果を使う）
    p.add_argument("--result_dir", type=str, default=None, help="Load from this results/<timestamp>/ dir")
    p.add_argument("--latest", action="store_true", help="Load latest result under results_root")

    # GIF保存
    p.add_argument("--save_gif", action="store_true", help="Save visualization GIF into result_dir")
    p.add_argument("--gif_name", type=str, default="preview.gif")

    args = p.parse_args()

    # 1) load mode（優先）
    if args.latest:
        rd = find_latest_result(args.results_root)
        out = load_result_dir(rd)
        meta = read_meta(rd)
        print("[INFO] loaded:", rd)
        if meta:
            print("[INFO] meta:", meta)
        print_summary(out)
        if args.visualize:
            anim = visualize_animo_output(out, fps=args.fps, stride=args.stride)
            if args.save_gif:
                save_gif(anim, str(Path(rd) / args.gif_name), fps=args.fps)
        return

    if args.result_dir is not None:
        rd = Path(args.result_dir)
        out = load_result_dir(rd)
        meta = read_meta(rd)
        print("[INFO] loaded:", rd)
        if meta:
            print("[INFO] meta:", meta)
        print_summary(out)
        if args.visualize:
            anim = visualize_animo_output(out, fps=args.fps, stride=args.stride)
            if args.save_gif:
                save_gif(anim, str(rd / args.gif_name), fps=args.fps)
        return

    # 2) inference mode
    if args.text is None:
        raise SystemExit("ERROR: --text is required unless you use --result_dir or --latest")

    gen = build_generator(args.device, text_dim=256, weights_path=args.weights)
    out = gen.generate(args.text, species=args.species, T=args.T)

    print_summary(out, text=args.text, species=args.species)

    # save
    result_dir = None
    if args.save:
        result_dir = save_result(out, args.text, args.species, args.device, args.T, text_dim=256, root=args.results_root)

    # visualize
    if args.visualize:
        anim = visualize_animo_output(out, fps=args.fps, stride=args.stride)
        # GIFは “保存先がある” ときだけ確実に書けるので、保存時に限定するのが安全
        if args.save_gif:
            if result_dir is None:
                # 保存先が無いなら results/latest みたいな運用が困るので明示エラーにする
                raise SystemExit("ERROR: --save_gif requires --save (so we know where to write the gif)")
            save_gif(anim, str(result_dir / args.gif_name), fps=args.fps)


if __name__ == "__main__":
    main()
