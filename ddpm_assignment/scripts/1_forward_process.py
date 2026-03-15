"""Task 1: DDPM Forward Process
Visualize how noise is progressively added to data.
Runs on 1D double peak, 2D Swiss Roll, and Fashion-MNIST.
"""
import sys, os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(TASK_DIR, '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import torch
from ddpm import make_schedule
from util import make_double_peak_dataloader, make_swiss_roll_dataloader, make_fashion_mnist_dataloader
from visualization import animate_forward_process_1d, animate_forward_process_2d, animate_forward_process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

schedule = make_schedule(T=1000, device=device)
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 1D Double Peak ----
print("\n=== 1D Double Peak Forward Process ===")
_, true_data_1d = make_double_peak_dataloader(n_samples=2000)
x_1d = true_data_1d.to(device)
animate_forward_process_1d(x_1d, schedule, step_stride=20, save_path=os.path.join(OUTPUT_DIR, "1_forward_1d.gif"))

# ---- 2D Swiss Roll ----
print("\n=== 2D Swiss Roll Forward Process ===")
_, true_data_2d = make_swiss_roll_dataloader(n_samples=2000)
x_2d = true_data_2d.to(device)
animate_forward_process_2d(x_2d, schedule, step_stride=20, save_path=os.path.join(OUTPUT_DIR, "1_forward_2d.gif"))

# ---- Fashion-MNIST ----
print("\n=== Fashion-MNIST Forward Process ===")
dataloader_img = make_fashion_mnist_dataloader(batch_size=8)
x_img = next(iter(dataloader_img))[0].to(device)  # [8, 1, 28, 28]
animate_forward_process_image(x_img, schedule, step_stride=20, save_path=os.path.join(OUTPUT_DIR, "1_forward_image.gif"))

print("\nTask 1 complete!")
