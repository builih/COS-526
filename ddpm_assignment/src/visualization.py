import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from PIL import Image

from ddpm import forward_diffusion_step, q_sample
from util import load_model_checkpoint


# ============================================================
# Forward process visualization
# ============================================================

def animate_forward_process_1d(x_0, schedule, step_stride=20, save_path=None):
    """Animate how noise is progressively added to 1D data step by step.

    Args:
        x_0: [N, 1] clean 1D data
        schedule: dict from make_schedule()
        step_stride: save one frame every this many diffusion steps
        save_path: if given, save animation as gif
    """
    T = schedule['T']
    x_t = x_0.clone()
    frames = [(0, x_t.cpu().numpy().flatten())]
    for t_val in range(T):
        t = torch.full((x_0.shape[0],), t_val, device=x_0.device, dtype=torch.long)
        x_t = forward_diffusion_step(x_t, t, torch.randn_like(x_t), schedule)
        if (t_val + 1) % step_stride == 0:
            frames.append((t_val + 1, x_t.cpu().numpy().flatten()))

    fig, ax = plt.subplots(figsize=(5, 3))
    _, _, patches = ax.hist(frames[0][1], bins=60, density=True, range=(-6, 6))
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.0)
    title = ax.set_title('t = 0')
    ax.set_xlabel('x')
    fig.suptitle('Forward Diffusion: Adding Noise to 1D Data', fontsize=13)

    def update(frame):
        t_val, data = frames[frame]
        ax.cla()
        ax.hist(data, bins=60, density=True, range=(-6, 6), alpha=0.7)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.0)
        ax.set_title(f't = {t_val}')
        ax.set_xlabel('x')

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)
    plt.tight_layout()
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Saved: {save_path}")
    return anim


def animate_forward_process_2d(x_0, schedule, step_stride=20, save_path=None):
    """Animate how noise is progressively added to 2D data step by step.

    Args:
        x_0: [N, 2] clean 2D data
        schedule: dict from make_schedule()
        step_stride: save one frame every this many diffusion steps
        save_path: if given, save animation as gif
    """
    T = schedule['T']
    x_t = x_0.clone()
    frames = [(0, x_t.cpu().numpy())]
    for t_val in range(T):
        t = torch.full((x_0.shape[0],), t_val, device=x_0.device, dtype=torch.long)
        x_t = forward_diffusion_step(x_t, t, torch.randn_like(x_t), schedule)
        if (t_val + 1) % step_stride == 0:
            frames.append((t_val + 1, x_t.cpu().numpy()))

    fig, ax = plt.subplots(figsize=(5, 5))
    scat = ax.scatter(frames[0][1][:, 0], frames[0][1][:, 1], s=3, alpha=0.4, c='steelblue')
    title = ax.set_title(f't = 0')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    fig.suptitle('Forward Diffusion: Adding Noise to 2D Data', fontsize=13)

    def update(frame):
        t_val, data = frames[frame]
        scat.set_offsets(data)
        title.set_text(f't = {t_val}')
        return [scat, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    plt.tight_layout()
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Saved: {save_path}")
    return anim


def animate_forward_process_image(x_0, schedule, step_stride=20, save_path=None):
    """Animate how noise is progressively added to images step by step.

    Args:
        x_0: [B, 1, H, W] clean images, normalized to [-1, 1]
        schedule: dict from make_schedule()
        step_stride: save one frame every this many diffusion steps
        save_path: if given, save animation as gif
    """
    T = schedule['T']
    n_show = min(8, x_0.shape[0])
    x_t = x_0[:n_show].clone()
    frames = [(0, [x_t[i].squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5 for i in range(n_show)])]
    for t_val in range(T):
        t = torch.full((n_show,), t_val, device=x_0.device, dtype=torch.long)
        x_t = forward_diffusion_step(x_t, t, torch.randn_like(x_t), schedule)
        if (t_val + 1) % step_stride == 0:
            imgs = [x_t[i].squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5 for i in range(n_show)]
            frames.append((t_val + 1, imgs))

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    ims = [axes[i].imshow(frames[0][1][i], cmap='gray', vmin=0, vmax=1) for i in range(n_show)]
    title = fig.suptitle('Forward Diffusion: t = 0', fontsize=13)

    def update(frame):
        t_val, imgs = frames[frame]
        for i in range(n_show):
            ims[i].set_data(imgs[i])
        title.set_text(f'Forward Diffusion: t = {t_val}')
        return ims + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Saved: {save_path}")
    return anim


# ============================================================
# Denoising trajectory animations
# ============================================================

def plot_2d_denoising(trajectory, true_data=None, save_path=None):
    """Animate denoising trajectory for 2D data.

    Args:
        trajectory: list of (timestep, [N, 2] tensor) tuples
        true_data: optional [N, 2] tensor of ground truth data
        save_path: if given, save animation as gif
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

    # Faint ground truth in background
    if true_data is not None:
        td = true_data.cpu().numpy()
        ax.scatter(td[:, 0], td[:, 1], s=1, alpha=0.08, c='green', zorder=0)

    traj_sorted = sorted(trajectory, key=lambda x: x[0], reverse=True)

    scat = ax.scatter([], [], s=3, alpha=0.4, c='steelblue')
    title = ax.set_title("")

    def update(frame):
        step, samples = traj_sorted[frame]
        s = samples.cpu().numpy()
        scat.set_offsets(s)
        title.set_text(f"Denoising: t = {step}")
        return [scat, title]

    anim = FuncAnimation(fig, update, frames=len(traj_sorted), interval=300, blit=True)
    plt.tight_layout()
    plt.close(fig)

    if save_path:
        anim.save(save_path, writer='pillow', fps=3)
        print(f"Saved: {save_path}")
    return anim


def _to_display_image(sample, value_range=(-1, 1)):
    sample = sample.detach().cpu()
    if sample.ndim == 2:
        sample = sample.unsqueeze(0)

    lo, hi = value_range
    sample = sample.clamp(lo, hi)
    if hi > lo:
        sample = (sample - lo) / (hi - lo)

    if sample.shape[0] == 1:
        return sample.squeeze(0).numpy(), "gray"
    return sample.permute(1, 2, 0).numpy(), None


def plot_image_denoising(trajectory, save_path=None, value_range=(-1, 1)):
    """Animate denoising trajectory for image data.

    Args:
        trajectory: list of (timestep, [B, 1, H, W] tensor) tuples
        save_path: if given, save animation as gif
        value_range: tuple giving the display range for samples
    """
    n_show = min(8, trajectory[0][1].shape[0])
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows + 0.5))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")

    traj_sorted = sorted(trajectory, key=lambda x: x[0], reverse=True)

    ims = []
    for i in range(n_show):
        img, cmap = _to_display_image(traj_sorted[0][1][i], value_range=value_range)
        ims.append(axes[i].imshow(img, cmap=cmap, vmin=0, vmax=1))
    title = fig.suptitle("", fontsize=14, fontweight='bold')

    def update(frame):
        step, samples = traj_sorted[frame]
        for i in range(n_show):
            img, _ = _to_display_image(samples[i], value_range=value_range)
            ims[i].set_data(img)
        title.set_text(f"Denoising: t = {step}")

    anim = FuncAnimation(fig, update, frames=len(traj_sorted), interval=300, blit=False)
    plt.tight_layout()
    plt.close(fig)

    if save_path:
        anim.save(save_path, writer='pillow', fps=3)
        print(f"Saved: {save_path}")
    return anim


def save_image_grid(samples, save_path=None, ncols=4, value_range=(-1, 1)):
    """Save a grid of image samples."""
    n_show = samples.shape[0]
    ncols = min(ncols, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx in range(n_show):
        img, cmap = _to_display_image(samples[idx], value_range=value_range)
        axes[idx].imshow(img, cmap=cmap, vmin=0, vmax=1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Training progress visualization
# ============================================================

def plot_training_loss(losses, save_path=None):
    """Plot training loss curve.

    Args:
        losses: list of per-epoch average losses
        save_path: if given, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('DDPM Training Loss')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_training_snapshots_2d(snapshots, true_data=None, save_path=None):
    """Show sampling quality at different training epochs for 2D data.

    Args:
        snapshots: list of (epoch, [N, 2] samples) tuples
        true_data: optional [N, 2] ground truth
        save_path: if given, save figure
    """
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for i, (epoch, samples) in enumerate(snapshots):
        data = samples.cpu().numpy()
        axes[i].scatter(data[:, 0], data[:, 1], s=3, alpha=0.5)
        if true_data is not None:
            td = true_data.cpu().numpy()
            axes[i].scatter(td[:, 0], td[:, 1], s=1, alpha=0.1, c='red')
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)
        axes[i].set_aspect('equal')
        axes[i].set_title(f'Epoch {epoch}')

    fig.suptitle('Training Progress: 2D Samples', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_training_snapshots_image(snapshots, save_path=None):
    """Show sampling quality at different training epochs for images.

    Args:
        snapshots: list of (epoch, [B, 1, H, W] samples) tuples
        save_path: if given, save figure
    """
    n_epochs = len(snapshots)
    n_show = min(4, snapshots[0][1].shape[0])
    fig, axes = plt.subplots(n_epochs, n_show, figsize=(2.5 * n_show, 2.5 * n_epochs))
    if n_epochs == 1:
        axes = axes[np.newaxis, :]

    for r, (epoch, samples) in enumerate(snapshots):
        for c in range(n_show):
            img = samples[c].squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5
            axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(f'Ep {epoch}', fontsize=10, rotation=0, labelpad=35)

    fig.suptitle('Training Progress: Image Samples', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# One-step denoising visualization (Part 2v)
# ============================================================

def _format_progress_title(base_title, epoch=None, loss=None):
    parts = [base_title]
    if epoch is not None:
        parts.append(f"Epoch {epoch}")
    if loss is not None:
        parts.append(f"Loss {loss:.4f}")
    return " | ".join(parts)


def _figure_to_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


def save_gif_from_images(images, save_path, fps=1):
    """Save a list of PIL images as a GIF."""
    if not images:
        raise ValueError("images must be non-empty")
    duration_ms = max(1, int(round(1000 / fps)))
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved: {save_path}")


def save_training_progress_gif_from_checkpoints(
    model,
    checkpoint_specs,
    losses,
    frame_builder,
    save_path,
    fps=1,
    device='cpu',
):
    """Build a GIF from saved checkpoints.

    Args:
        model: model instance to reuse while loading checkpoints
        checkpoint_specs: iterable of (epoch, checkpoint_path)
        losses: per-epoch loss list
        frame_builder: function(model=model, epoch=epoch, loss=loss) -> PIL image
        save_path: output GIF path
        fps: frames per second
        device: torch device used for loading checkpoints
    """
    frames = []
    for epoch, checkpoint_path in checkpoint_specs:
        load_model_checkpoint(model, checkpoint_path, device=device)
        frames.append(frame_builder(model=model, epoch=epoch, loss=losses[epoch - 1]))
    save_gif_from_images(frames, save_path, fps=fps)


def _build_one_step_denoising_image_figure(
    model,
    x_0,
    schedule,
    losses=None,
    timesteps_to_show=(50, 100, 400, 700),
    noise_bank=None,
    title=None,
    show_loss_plot=True,
):
    n_samples = min(3, x_0.shape[0])
    n_t = len(timesteps_to_show)
    n_cols = 1 + 2 * n_t
    has_loss = show_loss_plot and losses is not None and len(losses) > 0

    fig = plt.figure(
        figsize=(1.5 * n_cols, 1.5 * n_samples + (2.5 if has_loss else 0.8)),
        facecolor='black',
    )

    if has_loss:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3, bottom=0.08)
        ax_loss = fig.add_subplot(gs[0])
        ax_loss.set_facecolor('black')
        ax_loss.plot(losses, color='deepskyblue', alpha=0.8, linewidth=0.8)
        ax_loss.set_ylabel('MSE', color='white', fontsize=9)
        ax_loss.set_xlabel('Epoch', color='white', fontsize=9)
        ax_loss.tick_params(colors='white', labelsize=7)
        for spine in ax_loss.spines.values():
            spine.set_color('white')
        gs_imgs = gs[1].subgridspec(n_samples, n_cols, wspace=0.05, hspace=0.05)
    else:
        gs_imgs = fig.add_gridspec(n_samples, n_cols, wspace=0.05, hspace=0.05, bottom=0.08)

    axes_grid = {}

    model.eval()
    with torch.no_grad():
        for r in range(n_samples):
            ax = fig.add_subplot(gs_imgs[r, 0])
            ax.set_facecolor('black')
            ax.axis('off')
            clean_img = x_0[r].squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5
            ax.imshow(clean_img, cmap='gray', vmin=0, vmax=1)
            axes_grid[(r, 0)] = ax

            for ti, t_val in enumerate(timesteps_to_show):
                t_tensor = torch.full((1,), t_val, device=x_0.device, dtype=torch.long)
                if noise_bank is None:
                    noise = torch.randn(1, *x_0.shape[1:], device=x_0.device)
                else:
                    noise = noise_bank[t_val][r:r + 1]
                x_t = q_sample(x_0[r:r + 1], t_tensor, noise, schedule)
                eps_hat = model(x_t, t_tensor)

                sqrt_alpha_bar = torch.sqrt(schedule['alphas_cumprod'][t_val])
                sqrt_one_minus = torch.sqrt(1.0 - schedule['alphas_cumprod'][t_val])
                x_0_hat = (x_t - sqrt_one_minus * eps_hat) / sqrt_alpha_bar

                col_noisy = 1 + 2 * ti
                col_denoised = col_noisy + 1

                ax_n = fig.add_subplot(gs_imgs[r, col_noisy])
                ax_n.set_facecolor('black')
                ax_n.axis('off')
                noisy_img = x_t[0].squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5
                ax_n.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
                if r == 0:
                    ax_n.text(0.5, 0.98, 'Noisy', transform=ax_n.transAxes,
                              ha='center', va='top', fontsize=7, color='lightgray')
                axes_grid[(r, col_noisy)] = ax_n

                ax_d = fig.add_subplot(gs_imgs[r, col_denoised])
                ax_d.set_facecolor('black')
                ax_d.axis('off')
                denoised_img = x_0_hat[0].squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5
                ax_d.imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
                if r == 0:
                    ax_d.text(0.5, 0.98, 'Denoised', transform=ax_d.transAxes,
                              ha='center', va='top', fontsize=7, color='lightgray')
                axes_grid[(r, col_denoised)] = ax_d

    if title:
        fig.suptitle(title, fontsize=13, color='white', y=0.98)
    elif has_loss:
        ax_loss.text(
            0.98,
            0.95,
            f'Epoch: {len(losses)}',
            transform=ax_loss.transAxes,
            ha='right',
            va='top',
            color='white',
            fontsize=11,
        )

    fig.canvas.draw()
    last_row = n_samples - 1
    pos_orig = axes_grid[(last_row, 0)].get_position()
    fig.text(
        (pos_orig.x0 + pos_orig.x1) / 2,
        pos_orig.y0 - 0.01,
        'Original',
        ha='center',
        va='top',
        color='white',
        fontsize=10,
    )

    for ti, t_val in enumerate(timesteps_to_show):
        col_n = 1 + 2 * ti
        col_d = col_n + 1
        pos_n = axes_grid[(last_row, col_n)].get_position()
        pos_d = axes_grid[(last_row, col_d)].get_position()
        fig.text(
            (pos_n.x0 + pos_d.x1) / 2,
            pos_n.y0 - 0.01,
            f't = {t_val}',
            ha='center',
            va='top',
            color='white',
            fontsize=10,
        )
    return fig


def plot_one_step_denoising_image(
    model,
    x_0,
    schedule,
    losses=None,
    timesteps_to_show=(50, 100, 400, 700),
    save_path=None,
):
    """Plot one-step denoising for image data."""
    fig = _build_one_step_denoising_image_figure(
        model,
        x_0,
        schedule,
        losses=losses,
        timesteps_to_show=timesteps_to_show,
        show_loss_plot=True,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Saved: {save_path}")
    plt.close(fig)


def render_one_step_denoising_image_frame(
    model,
    x_0,
    schedule,
    timesteps_to_show=(50, 100, 400, 700),
    noise_bank=None,
    epoch=None,
    loss=None,
):
    """Render a single training-progress frame for image one-step denoising."""
    fig = _build_one_step_denoising_image_figure(
        model,
        x_0,
        schedule,
        timesteps_to_show=timesteps_to_show,
        noise_bank=noise_bank,
        title=_format_progress_title('One-Step Denoising: Fashion-MNIST', epoch, loss),
        show_loss_plot=False,
    )
    image = _figure_to_image(fig)
    plt.close(fig)
    return image


def _build_noised_and_denoised_2d_figure(
    model,
    x_0,
    schedule,
    timesteps_to_show=(50, 100, 200, 500),
    noise=None,
    title=None,
):
    n_t = len(timesteps_to_show)
    n_cols = 1 + n_t
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

    def scatter_plot(ax, data, subplot_title, color='steelblue'):
        ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.4, c=color)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(subplot_title, fontsize=10)
        ax.tick_params(labelsize=7)

    clean = x_0.cpu().numpy()
    scatter_plot(axes[0, 0], clean, 'Original', color='steelblue')
    scatter_plot(axes[1, 0], clean, 'Original', color='steelblue')

    if noise is None:
        noise = torch.randn_like(x_0)

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(timesteps_to_show):
            t = torch.full((x_0.shape[0],), t_val, device=x_0.device, dtype=torch.long)
            x_t = q_sample(x_0, t, noise, schedule)
            eps_hat = model(x_t, t)
            sqrt_alpha_bar = torch.sqrt(schedule['alphas_cumprod'][t_val])
            sqrt_one_minus = torch.sqrt(1.0 - schedule['alphas_cumprod'][t_val])
            x_0_hat = (x_t - sqrt_one_minus * eps_hat) / sqrt_alpha_bar

            scatter_plot(axes[0, 1 + i], x_t.cpu().numpy(), f'Noisy (t={t_val})', color='steelblue')
            scatter_plot(axes[1, 1 + i], x_0_hat.cpu().numpy(), f'Denoised (t={t_val})', color='coral')

    axes[0, 0].set_ylabel('Noisy', fontsize=12)
    axes[1, 0].set_ylabel('Denoised', fontsize=12)
    fig.suptitle(title or 'One-Step Denoising: 2D Swiss Roll', fontsize=13)
    plt.tight_layout()
    return fig


def plot_noised_and_denoised_2d(
    model,
    x_0,
    schedule,
    timesteps_to_show=(50, 100, 200, 500),
    save_path=None,
):
    """Plot one-step denoising for 2D Swiss roll data."""
    fig = _build_noised_and_denoised_2d_figure(
        model,
        x_0,
        schedule,
        timesteps_to_show=timesteps_to_show,
    )
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def render_noised_and_denoised_2d_frame(
    model,
    x_0,
    schedule,
    timesteps_to_show=(50, 100, 200, 500),
    noise=None,
    epoch=None,
    loss=None,
):
    """Render a single training-progress frame for 2D one-step denoising."""
    fig = _build_noised_and_denoised_2d_figure(
        model,
        x_0,
        schedule,
        timesteps_to_show=timesteps_to_show,
        noise=noise,
        title=_format_progress_title('One-Step Denoising: 2D Swiss Roll', epoch, loss),
    )
    image = _figure_to_image(fig)
    plt.close(fig)
    return image


# ============================================================
# Noise comparison (Part 4)
# ============================================================

def animate_noise_comparison_2d(traj_with, traj_without, true_data=None, save_path=None):
    """Animate side-by-side denoising: with noise vs without noise (2D).

    Args:
        traj_with: trajectory from ddpm_sample (with noise)
        traj_without: trajectory from ddpm_sample_no_noise
        true_data: optional ground truth [N, 2]
        save_path: if given, save animation as gif
    """
    traj_w = sorted(traj_with, key=lambda x: x[0], reverse=True)
    traj_wo = sorted(traj_without, key=lambda x: x[0], reverse=True)
    n_frames = min(len(traj_w), len(traj_wo))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')

    # Faint ground truth in background
    if true_data is not None:
        td = true_data.cpu().numpy()
        for ax in axes:
            ax.scatter(td[:, 0], td[:, 1], s=1, alpha=0.08, c='green', zorder=0)

    scat_w = axes[0].scatter([], [], s=3, alpha=0.4, c='steelblue')
    scat_wo = axes[1].scatter([], [], s=3, alpha=0.4, c='steelblue')

    axes[0].set_title('With noise (DDPM)')
    axes[1].set_title('Without noise (sigma=0)')
    title = fig.suptitle('', fontsize=13)

    def update(frame):
        step_w, samples_w = traj_w[frame]
        step_wo, samples_wo = traj_wo[frame]
        s_w = samples_w.cpu().numpy()
        s_wo = samples_wo.cpu().numpy()
        scat_w.set_offsets(s_w)
        scat_wo.set_offsets(s_wo)
        title.set_text(f'Denoising: t = {step_w}')
        return [scat_w, scat_wo, title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=True)
    plt.tight_layout()
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer='pillow', fps=3)
        print(f"Saved: {save_path}")
    return anim


def animate_noise_comparison_image(traj_with, traj_without, save_path=None):
    """Animate side-by-side denoising: with noise vs without noise (images).

    Top row: with noise, bottom row: without noise.

    Args:
        traj_with: trajectory from ddpm_sample
        traj_without: trajectory from ddpm_sample_no_noise
        save_path: if given, save animation as gif
    """
    traj_w = sorted(traj_with, key=lambda x: x[0], reverse=True)
    traj_wo = sorted(traj_without, key=lambda x: x[0], reverse=True)
    n_frames = min(len(traj_w), len(traj_wo))
    n_show = min(4, traj_w[0][1].shape[0])

    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    for ax in axes.flatten():
        ax.axis('off')

    # Initialize with first frame
    def to_img(tensor):
        return tensor.squeeze().cpu().clamp(-1, 1).numpy() * 0.5 + 0.5

    ims_w = [axes[0, c].imshow(to_img(traj_w[0][1][c]), cmap='gray', vmin=0, vmax=1) for c in range(n_show)]
    ims_wo = [axes[1, c].imshow(to_img(traj_wo[0][1][c]), cmap='gray', vmin=0, vmax=1) for c in range(n_show)]

    axes[0, 0].set_ylabel('With noise', fontsize=10, rotation=0, labelpad=55)
    axes[1, 0].set_ylabel('No noise', fontsize=10, rotation=0, labelpad=55)
    title = fig.suptitle('', fontsize=13)

    def update(frame):
        step_w, samples_w = traj_w[frame]
        step_wo, samples_wo = traj_wo[frame]
        for c in range(n_show):
            ims_w[c].set_data(to_img(samples_w[c]))
            ims_wo[c].set_data(to_img(samples_wo[c]))
        title.set_text(f'Denoising: t = {step_w}')
        return ims_w + ims_wo + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=True)
    plt.tight_layout()
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer='pillow', fps=3)
        print(f"Saved: {save_path}")
    return anim
