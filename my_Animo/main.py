
# main.py
from config import AniMoShapeConfig
from train_utils import train_demo, full_pipeline_3d_demo


def main():
    cfg = AniMoShapeConfig(
        vocab_size=30000,
        max_text_len=32,
        max_motion_len=64,
        num_quantizers=6,
        codebook_size=1024,
        num_species=114,
        num_attrs=32,
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_joints=24,      # 3D関節数
    )

    model = train_demo(cfg, num_epochs=2, batch_size=8, dataset_size=200)
    full_pipeline_3d_demo(model, cfg)


if __name__ == "__main__":
    main()
