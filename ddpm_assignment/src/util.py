import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import make_swiss_roll


# ============================================================
# Data loaders
# ============================================================

def make_double_peak_dataloader(n_samples=10000, batch_size=256):
    """1D bimodal Gaussian: mixture of N(-3, 0.5) and N(3, 0.5).

    Returns:
        dataloader: DataLoader yielding [B, 1] tensors
        true_data: [N, 1] tensor of the full dataset (for visualization)
    """
    half = n_samples // 2
    data = np.concatenate([
        np.random.normal(-3, 0.5, half),
        np.random.normal(3, 0.5, half),
    ])
    np.random.shuffle(data)
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # [N, 1]
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    return loader, tensor


def make_swiss_roll_dataloader(n_samples=10000, batch_size=256):
    """2D Swiss roll (x-z plane), normalized to ~[-1,1].

    Returns:
        dataloader: DataLoader yielding [B, 2] tensors
        true_data: [N, 2] tensor of the full dataset (for visualization)
    """
    data_3d, _ = make_swiss_roll(n_samples, noise=0.2)
    data_2d = data_3d[:, [0, 2]]  # x and z planes
    data_2d = (data_2d - data_2d.mean(axis=0)) / data_2d.std(axis=0)
    tensor = torch.tensor(data_2d, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    return loader, tensor


def make_fashion_mnist_dataloader(batch_size=128):
    """Fashion-MNIST 28x28, normalized to [-1, 1].

    Returns:
        dataloader: DataLoader yielding ([B, 1, 28, 28], labels) tuples
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
    ])
    dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ============================================================
# Networks
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class NoisePredictor(nn.Module):
    """MLP that predicts noise given (x, t). Works for any data_dim."""

    def __init__(self, data_dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and time embedding injection."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, t_emb):
        residual = x
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip_proj(residual)


class SimpleUNet(nn.Module):
    """U-Net for predicting noise on 28x28 grayscale images.

    Spatial:  28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
    Channels: 1 -> 32 -> 64 -> 128 -> 64 -> 32 -> 1
    """
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=128):
        super().__init__()
        ch = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.init_conv = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        self.enc_block0 = ResidualBlock(ch, ch, time_emb_dim)
        self.down0 = nn.MaxPool2d(2)

        self.enc_block1 = ResidualBlock(ch, ch * 2, time_emb_dim)
        self.down1 = nn.MaxPool2d(2)

        self.enc_block2 = ResidualBlock(ch * 2, ch * 4, time_emb_dim)

        # Bottleneck
        self.bottleneck = ResidualBlock(ch * 4, ch * 4, time_emb_dim)

        # Decoder
        self.dec_block2 = ResidualBlock(ch * 4 + ch * 4, ch * 4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)

        self.dec_block1 = ResidualBlock(ch * 2 + ch * 2, ch * 2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)

        self.dec_block0 = ResidualBlock(ch + ch, ch, time_emb_dim)

        # Final output
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        e0 = self.init_conv(x)
        e0 = self.enc_block0(e0, t_emb)

        e1 = self.down0(e0)
        e1 = self.enc_block1(e1, t_emb)

        e2 = self.down1(e1)
        e2 = self.enc_block2(e2, t_emb)

        # Bottleneck
        b = self.bottleneck(e2, t_emb)

        # Decoder with skip connections
        d2 = torch.cat([b, e2], dim=1)
        d2 = self.dec_block2(d2, t_emb)
        d2 = self.up2(d2)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='nearest')

        d1 = torch.cat([d2, e1], dim=1)
        d1 = self.dec_block1(d1, t_emb)
        d1 = self.up1(d1)
        if d1.shape[2:] != e0.shape[2:]:
            d1 = F.interpolate(d1, size=e0.shape[2:], mode='nearest')

        d0 = torch.cat([d1, e0], dim=1)
        d0 = self.dec_block0(d0, t_emb)

        out = self.final_act(self.final_norm(d0))
        out = self.final_conv(out)
        return out


class DiffUNetWrapper(nn.Module):
    """Wrap a deepinv DiffUNet with the same model(x, t) interface used here."""

    def __init__(self, unet, out_channels=None):
        super().__init__()
        self.unet = unet
        self.out_channels = out_channels

    def forward(self, x, t):
        out = self.unet(x, t, type_t="timestep")
        if self.out_channels is not None:
            out = out[:, :self.out_channels, ...]
        return out


# ============================================================
# Training helper
# ============================================================

def checkpoint_path(checkpoint_dir, prefix, epoch):
    """Return the path for a named training checkpoint."""
    return os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.pth")


def save_model_checkpoint(model, checkpoint_dir, prefix, epoch):
    """Save a training checkpoint and return its path."""
    path = checkpoint_path(checkpoint_dir, prefix, epoch)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")
    return path


def load_model_checkpoint(model, path, device='cpu'):
    """Load model weights from a checkpoint path."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()


def make_checkpoint_callback(checkpoint_dir, prefix):
    """Create a callback that saves checkpoints during training."""
    def callback(model, epoch, avg_loss, losses):
        del avg_loss, losses
        save_model_checkpoint(model, checkpoint_dir, prefix, epoch)

    return callback

def train_ddpm(
    model,
    dataloader,
    schedule,
    train_step_fn,
    epochs,
    lr=1e-4,
    device='cpu',
    log_interval=None,
    checkpoint_epochs=None,
    checkpoint_callback=None,
):
    """Generic training loop for DDPM.

    Args:
        model: noise prediction network
        dataloader: yields batches (tensors or (tensor, label) tuples)
        schedule: dict from make_schedule()
        train_step_fn: function(model, x, schedule) -> loss tensor
        epochs: number of training epochs
        lr: learning rate
        device: torch device
        log_interval: print every N epochs (default: epochs//10)
        checkpoint_epochs: optional iterable of 1-based epochs to snapshot
        checkpoint_callback: optional function called as
            checkpoint_callback(model, epoch, avg_loss, losses)

    Returns:
        losses: list of per-epoch average losses
    """
    if log_interval is None:
        log_interval = max(1, epochs // 10)
    checkpoint_epochs = set(checkpoint_epochs or [])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            x = batch.to(device)
            optimizer.zero_grad()
            loss = train_step_fn(model, x, schedule)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        losses.append(avg_loss)
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if checkpoint_callback is not None and (epoch + 1) in checkpoint_epochs:
            checkpoint_callback(model, epoch + 1, avg_loss, losses)

    return losses
