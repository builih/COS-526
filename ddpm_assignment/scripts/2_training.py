"""Task 2: DDPM Training
Train noise prediction models on 2D Swiss Roll and Fashion-MNIST.
Save selected checkpoints and visualize one-step denoising progress during training.
"""
from functools import partial
import os
import sys

import torch

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(TASK_DIR, '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from ddpm import ddpm_train_step, make_schedule
from util import (
    NoisePredictor,
    SimpleUNet,
    checkpoint_path,
    make_fashion_mnist_dataloader,
    make_checkpoint_callback,
    make_swiss_roll_dataloader,
    train_ddpm,
)
from visualization import (
    plot_training_loss,
    render_noised_and_denoised_2d_frame,
    render_one_step_denoising_image_frame,
    save_training_progress_gif_from_checkpoints,
)


OUTPUT_DIR     = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

TWO_D_EPOCHS = 500
IMAGE_EPOCHS = 10
VIS_EPOCHS_2D = [1, 2, 5, 100, TWO_D_EPOCHS]
VIS_EPOCHS_IMAGE = [1, 2, 3, 7, IMAGE_EPOCHS]
VIS_TIMESTEPS_2D = (50, 100, 200, 500)
VIS_TIMESTEPS_IMAGE = (50, 100, 400, 700)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

schedule = make_schedule(T=1000, device=device)

# ============================================================
# 2D Swiss Roll
# ============================================================
print("\n=== Training 2D Swiss Roll ===")
dataloader_2d, true_data_2d = make_swiss_roll_dataloader(n_samples=10000, batch_size=256)
vis_data_2d = true_data_2d[:2000].to(device)
vis_noise_2d = torch.randn_like(vis_data_2d)
model_2d = NoisePredictor(data_dim=2).to(device)

losses_2d = train_ddpm(
    model_2d,
    dataloader_2d,
    schedule,
    ddpm_train_step,
    epochs=TWO_D_EPOCHS,
    lr=1e-3,
    device=device,
    log_interval=50,
    checkpoint_epochs=VIS_EPOCHS_2D,
    checkpoint_callback=make_checkpoint_callback(CHECKPOINT_DIR, "model_2d"),
)

plot_training_loss(losses_2d, save_path=os.path.join(OUTPUT_DIR, "2_loss_2d.png"))

torch.save(model_2d.state_dict(), os.path.join(CHECKPOINT_DIR, "model_2d_final.pth"))
torch.save(model_2d.state_dict(), os.path.join(CHECKPOINT_DIR, "model_2d.pth"))
torch.save(losses_2d, os.path.join(CHECKPOINT_DIR, "losses_2d.pt"))
print("Saved final 2D model and loss history.")

save_training_progress_gif_from_checkpoints(
    model_2d,
    [(epoch, checkpoint_path(CHECKPOINT_DIR, "model_2d", epoch)) for epoch in VIS_EPOCHS_2D],
    losses_2d,
    partial(
        render_noised_and_denoised_2d_frame,
        x_0=vis_data_2d,
        schedule=schedule,
        timesteps_to_show=VIS_TIMESTEPS_2D,
        noise=vis_noise_2d,
    ),
    save_path=os.path.join(OUTPUT_DIR, "2_training_2d.gif"),
    fps=1,
    device=device,
)


# ============================================================
# Fashion-MNIST
# ============================================================
print("\n=== Training Fashion-MNIST ===")
dataloader_img = make_fashion_mnist_dataloader(batch_size=128)
vis_images = next(iter(dataloader_img))[0][:3].to(device)
vis_image_noise_bank = {
    t_val: torch.randn(vis_images.shape[0], *vis_images.shape[1:], device=device)
    for t_val in VIS_TIMESTEPS_IMAGE
}

model_img = SimpleUNet(in_channels=1).to(device)
total_params = sum(p.numel() for p in model_img.parameters())
print(f"SimpleUNet parameters: {total_params:,}")

losses_img = train_ddpm(
    model_img,
    dataloader_img,
    schedule,
    ddpm_train_step,
    epochs=IMAGE_EPOCHS,
    lr=2e-4,
    device=device,
    log_interval=2,
    checkpoint_epochs=VIS_EPOCHS_IMAGE,
    checkpoint_callback=make_checkpoint_callback(CHECKPOINT_DIR, "model_image"),
)

plot_training_loss(losses_img, save_path=os.path.join(OUTPUT_DIR, "2_loss_image.png"))

torch.save(model_img.state_dict(), os.path.join(CHECKPOINT_DIR, "model_image_final.pth"))
torch.save(model_img.state_dict(), os.path.join(CHECKPOINT_DIR, "model_image.pth"))
torch.save(losses_img, os.path.join(CHECKPOINT_DIR, "losses_image.pt"))
print("Saved final image model and loss history.")

save_training_progress_gif_from_checkpoints(
    model_img,
    [(epoch, checkpoint_path(CHECKPOINT_DIR, "model_image", epoch)) for epoch in VIS_EPOCHS_IMAGE],
    losses_img,
    partial(
        render_one_step_denoising_image_frame,
        x_0=vis_images,
        schedule=schedule,
        timesteps_to_show=VIS_TIMESTEPS_IMAGE,
        noise_bank=vis_image_noise_bank,
    ),
    save_path=os.path.join(OUTPUT_DIR, "2_training_image.gif"),
    fps=1,
    device=device,
)

print("\nTask 2 complete!")
