"""Task 3: DDPM Sampling & Noise Analysis
Load trained models, visualize the full denoising process,
and compare sampling with vs. without stochastic noise.
"""
import sys, os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(TASK_DIR, '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import torch
from ddpm import make_schedule, ddpm_sample, ddpm_sample_no_noise
from util import NoisePredictor, SimpleUNet, make_swiss_roll_dataloader
from visualization import (
    plot_2d_denoising,
    plot_image_denoising,
    animate_noise_comparison_2d,
    animate_noise_comparison_image,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

schedule = make_schedule(T=1000, device=device)

OUTPUT_DIR     = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load trained models
# ============================================================
print("\nLoading trained models...")

model_2d = NoisePredictor(data_dim=2).to(device)
model_2d.load_state_dict(torch.load(
    os.path.join(CHECKPOINT_DIR, "model_2d.pth"), map_location=device, weights_only=True
))
model_2d.eval()

model_img = SimpleUNet(in_channels=1).to(device)
model_img.load_state_dict(torch.load(
    os.path.join(CHECKPOINT_DIR, "model_image.pth"), map_location=device, weights_only=True
))
model_img.eval()

_, true_data_2d = make_swiss_roll_dataloader(n_samples=2000)

# ============================================================
# Part A: DDPM Sampling (with noise)
# ============================================================
print("\n=== 2D Swiss Roll Sampling ===")
samples_2d, traj_2d = ddpm_sample(model_2d, (1000, 2), schedule, n_snapshot_steps=50, device=device)
plot_2d_denoising(traj_2d, true_data_2d, save_path=os.path.join(OUTPUT_DIR, "3_sampling_2d.gif"))
print(f"Generated {samples_2d.shape[0]} 2D samples")

print("\n=== Fashion-MNIST Sampling ===")
samples_img, traj_img = ddpm_sample(model_img, (8, 1, 28, 28), schedule, n_snapshot_steps=20, device=device)
plot_image_denoising(traj_img, save_path=os.path.join(OUTPUT_DIR, "3_sampling_image.gif"))
print(f"Generated {samples_img.shape[0]} images")

# ============================================================
# Part B: Sampling WITHOUT noise (sigma_t = 0)
# ============================================================
print("\n=== 2D Sampling Without Noise ===")
_, traj_no_noise_2d = ddpm_sample_no_noise(model_2d, (1000, 2), schedule, n_snapshot_steps=20, device=device)

print("\n=== Image Sampling Without Noise ===")
_, traj_no_noise_img = ddpm_sample_no_noise(model_img, (8, 1, 28, 28), schedule, n_snapshot_steps=20, device=device)

# ============================================================
# Part C: Side-by-side comparison
# ============================================================
print("\n=== Noise Comparison: 2D ===")
# Re-sample with noise using same snapshot count for fair visual comparison
_, traj_with_2d = ddpm_sample(model_2d, (1000, 2), schedule, n_snapshot_steps=20, device=device)
animate_noise_comparison_2d(
    traj_with_2d, traj_no_noise_2d, true_data_2d,
    save_path=os.path.join(OUTPUT_DIR, "3_noise_comparison_2d.gif"),
)

print("\n=== Noise Comparison: Images ===")
_, traj_with_img = ddpm_sample(model_img, (8, 1, 28, 28), schedule, n_snapshot_steps=20, device=device)
animate_noise_comparison_image(
    traj_with_img, traj_no_noise_img,
    save_path=os.path.join(OUTPUT_DIR, "3_noise_comparison_image.gif"),
)

print("\nTask 3 complete!")
