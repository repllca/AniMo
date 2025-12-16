# train.py
import torch
from torch.utils.data import DataLoader
from models import VQVAE
from moving_mnist import MovingMNISTDataset
from tqdm import tqdm
import imageio
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- データセット ---
dataset = MovingMNISTDataset("./data", seq_len=16, image_size=64)
dloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# --- モデル ---
model = VQVAE(in_ch=1, hidden=64, z_dim=64, num_embeddings=512, beta=0.25).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# --- 保存先ディレクトリ ---
os.makedirs("samples", exist_ok=True)

# --- 学習ループ ---
for epoch in range(1, 3):  # 最初は2エポックで動作確認
    for i, x in enumerate(tqdm(dloader)):
        x = x.to(device)  # B,1,T,H,W
        x_recon, loss, recon_loss, vq_loss, indices = model(x)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # --- 最初のバッチで再構成動画保存 ---
        if i == 0:
            x_np = x[0,0].cpu().numpy()         # T,H,W
            x_recon_np = x_recon[0,0].detach().cpu().numpy()
            
            # GIF に保存
            frames_orig = [ (frame*255).astype('uint8') for frame in x_np ]
            frames_recon = [ (frame*255).astype('uint8') for frame in x_recon_np ]
            
            imageio.mimsave(f"samples/epoch{epoch}_orig.gif", frames_orig, fps=5)
            imageio.mimsave(f"samples/epoch{epoch}_recon.gif", frames_recon, fps=5)
            
    print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f}")

print("学習終了！ samples/ に元動画と再構成動画が保存されました。")
