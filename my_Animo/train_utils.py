
# train_utils.py
import torch
import torch.nn.functional as F

from config import AniMoShapeConfig
from dataset import make_dataloader
from model import MinimalAniMoLike
from motion_decoder import SimpleRVQMotionDecoder3D


def train_demo(
    cfg: AniMoShapeConfig,
    num_epochs: int = 2,
    batch_size: int = 8,
    dataset_size: int = 200,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)
    model = MinimalAniMoLike(cfg).to(device)
    loader = make_dataloader(cfg, batch_size=batch_size, size=dataset_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_count = 0

        for text_ids, species_ids, attr_ids, target_tokens in loader:
            text_ids = text_ids.to(device)
            species_ids = species_ids.to(device)
            attr_ids = attr_ids.to(device)
            target_tokens = target_tokens.to(device)

            optimizer.zero_grad()
            logits = model(text_ids, species_ids, attr_ids)     # [B,T,Q,K]

            B, T, Q, K = logits.shape
            logits_flat = logits.view(B*T*Q, K)
            target_flat = target_tokens.view(B*T*Q)

            loss = F.cross_entropy(logits_flat, target_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_count += B

        print(f"Epoch {epoch+1}/{num_epochs} loss={total_loss/total_count:.4f}")

    return model


@torch.no_grad()
def full_pipeline_3d_demo(model: MinimalAniMoLike, cfg: AniMoShapeConfig):
    """
    テキスト -> トークン -> 3Dモーション のデモ
    """
    model.eval()
    device = next(model.parameters()).device
    decoder = SimpleRVQMotionDecoder3D(cfg).to(device)

    # ダミー入力
    B = 1
    L = cfg.max_text_len
    text_ids = torch.randint(1, cfg.vocab_size, (B,L), dtype=torch.long, device=device)
    species_ids = torch.randint(0, cfg.num_species, (B,), dtype=torch.long, device=device)
    attr_ids = torch.randint(0, cfg.num_attrs, (B,), dtype=torch.long, device=device)

    logits = model(text_ids, species_ids, attr_ids)  # [B,T,Q,K]
    tokens = logits.argmax(dim=-1)                  # [B,T,Q]

    motion_3d = decoder(tokens)                     # [B,T,J,3]

    print("tokens :", tokens.shape)
    print("motion_3d :", motion_3d.shape)
    print("sample motion_3d:", motion_3d[0, :2, :3, :])
