# config.py
from dataclasses import dataclass


@dataclass
class AniMoShapeConfig:
    # ==== テキスト側 ==== #
    vocab_size: int = 20000
    max_text_len: int = 32

    # ==== トークン側（RVQ） ==== #
    max_motion_len: int = 64
    num_quantizers: int = 6
    codebook_size: int = 1024

    # ==== 条件側（属性） ==== #
    num_species: int = 64
    num_attrs: int = 32

    # ==== Transformer設定 ==== #
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4

    pad_id: int = 0

    # ==== 3Dモーション設定 ==== #
    num_joints: int = 24      # 動物の関節数
    motion_dim: int = 24 * 3  # Joints * 3D
