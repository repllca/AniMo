
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------
# Encoder (small Conv3D)
# ----------------
class VideoEncoder(nn.Module):
    def __init__(self, in_ch=1, hidden=64, z_dim=64):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, hidden, kernel_size=4, stride=2, padding=1)  # /2
        self.conv2 = nn.Conv3d(hidden, hidden, kernel_size=4, stride=2, padding=1)  # /4
        self.conv3 = nn.Conv3d(hidden, z_dim, kernel_size=3, stride=1, padding=1)   # latent channels = z_dim

    def forward(self, x):  # x: B, C, T, H, W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        z = self.conv3(x)  # B, z_dim, T', H', W'
        return z

# ----------------
# Decoder (small ConvTranspose3D)
# ----------------
class VideoDecoder(nn.Module):
    def __init__(self, z_dim=64, hidden=64, out_ch=1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose3d(z_dim, hidden, kernel_size=4, stride=2, padding=1)  # x2
        self.deconv2 = nn.ConvTranspose3d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.Conv3d(hidden, out_ch, kernel_size=3, padding=1)

    def forward(self, z_q):
        x = F.relu(self.deconv1(z_q))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

# ----------------
# Vector Quantizer (vanilla with codebook loss)
# ----------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embed = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embed.weight, -1.0 / self.K, 1.0 / self.K)

    def forward(self, z):  # z: B, D, T, H, W
        # move channels last to compute distances
        z_perm = z.permute(0,2,3,4,1).contiguous()  # B, T, H, W, D
        flat_z = z_perm.view(-1, self.D)  # (B*T*H*W, D)

        # compute L2 distance to embeddings efficiently
        emb = self.embed.weight  # K, D
        # dist = ||z||^2 - 2 z e^T + ||e||^2
        z_sq = (flat_z**2).sum(dim=1, keepdim=True)  # N,1
        e_sq = (emb**2).sum(dim=1)  # K
        dist = z_sq - 2 * flat_z @ emb.t() + e_sq.unsqueeze(0)  # N, K

        indices = torch.argmin(dist, dim=1)  # N
        z_q = self.embed(indices).view(z_perm.shape)  # B, T, H, W, D

        # straight-through
        z_q_st = z_perm + (z_q - z_perm).detach()
        z_q_st = z_q_st.permute(0,4,1,2,3).contiguous()  # B, D, T, H, W

        # losses
        e_q = z_q.permute(0,4,1,2,3).contiguous()  # B, D, T, H, W for codebook loss calc
        codebook_loss = F.mse_loss(e_q.detach(), z)
        commitment_loss = F.mse_loss(e_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # for logging: reshape indices to B, T, H, W
        indices = indices.view(z_perm.shape[0], z_perm.shape[1], z_perm.shape[2], z_perm.shape[3])  # B, T, H, W
        return z_q_st, vq_loss, indices

# ----------------
# VQ-VAE wrapper
# ----------------
class VQVAE(nn.Module):
    def __init__(self, in_ch=1, hidden=64, z_dim=64, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = VideoEncoder(in_ch, hidden, z_dim)
        self.vq = VectorQuantizer(num_embeddings, z_dim, beta)
        self.decoder = VideoDecoder(z_dim, hidden, in_ch)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        return x_recon, loss, recon_loss, vq_loss, indices
