"""
ddpm.py — Denoising Diffusion Probabilistic Models

Core implementation of the DDPM forward and reverse processes.

Three functions require your implementation (marked YOUR CODE HERE):
    forward_diffusion_step  — single-step Markov transition  q(x_t | x_{t-1})
    q_sample                — closed-form direct jump         q(x_t | x_0)
    ddpm_sample             — full reverse diffusion loop     p_theta(x_{t-1} | x_t)

All other functions are provided and should not be modified.
"""

import torch
import torch.nn.functional as F


# ============================================================
# Noise schedule  (provided)
# ============================================================

def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    """Build the linear beta schedule and pre-compute derived quantities.

    Returns a dict containing:
        T               — total number of diffusion timesteps
        betas           — noise variance at each step,           shape [T]
        alphas          — signal retention at each step (1-beta), shape [T]
        alphas_cumprod  — cumulative product of alphas (alpha_bar), shape [T]
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'T': T,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
    }


# ============================================================
# Forward process  (implement)
# ============================================================

def forward_diffusion_step(x_prev, t, noise, schedule):
    """Single Markov transition: corrupt x_{t-1} by one step to obtain x_t.

    Implements the step-wise forward distribution:

    Args:
        x_prev   : data at previous step, shape [B, ...]
        t        : timestep index for each sample, shape [B]
        noise    : pre-sampled Gaussian noise, same shape as x_prev
        schedule : noise schedule dict from make_schedule()

    Returns:
        x_t      : noised data at timestep t, same shape as x_prev
    """
    # Scale factors are reshaped to broadcast correctly over any data shape.
    dims = [1] * (x_prev.ndim - 1)
    beta = torch.sqrt(schedule['betas'][t]).view(-1, *dims)

    # ============ YOUR CODE HERE ============
    x_t = torch.sqrt(1 - schedule['betas'][t]).view(-1, *dims) * x_prev + beta * noise
    # raise NotImplementedError
    # ============ END YOUR CODE ============

    return x_t


def q_sample(x_0, t, noise, schedule):
    """Closed-form forward diffusion: jump from clean data x_0 to noised x_t.

    Exploits the reparameterization that collapses the Markov chain into a
    single operation, allowing efficient sampling at arbitrary timesteps
    during training without iterating through all intermediate steps.

    Args:
        x_0      : clean data, shape [B, ...]
        t        : timestep index for each sample, shape [B]
        noise    : pre-sampled Gaussian noise, same shape as x_0
        schedule : noise schedule dict from make_schedule()

    Returns:
        x_t      : noised data at timestep t, same shape as x_0
    """
    # alpha_bar_t = prod_{s=1}^{t} alpha_s, pre-computed in the schedule.
    dims = [1] * (x_0.ndim - 1)
    alpha_bar_t = schedule['alphas_cumprod'][t].view(-1, *dims)

    # ============ YOUR CODE HERE ============
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    # raise NotImplementedError
    # ============ END YOUR CODE ============

    return x_t


# ============================================================
# Training  (provided)
# ============================================================

def ddpm_train_step(model, x, schedule):
    """One DDPM training iteration.

    Samples a uniformly random timestep, corrupts the input batch via
    q_sample, predicts the added noise with the model, and returns the
    mean squared error between the true and predicted noise.

    Args:
        model    : noise predictor network, called as model(x_t, t)
        x        : clean data batch, shape [B, ...]
        schedule : noise schedule dict from make_schedule()

    Returns:
        loss     : scalar MSE loss tensor
    """
    T = schedule['T']
    noise = torch.randn_like(x)
    t = torch.randint(0, T, (x.shape[0],), device=x.device)

    x_t = q_sample(x, t, noise, schedule)
    predicted_noise = model(x_t, t)
    return F.mse_loss(predicted_noise, noise)


# ============================================================
# Reverse process — sampling  (implement)
# ============================================================

def ddpm_sample(model, shape, schedule, n_snapshot_steps=10, abl_no_noise=False, device='cpu'):
    """Full reverse diffusion: iteratively denoise from pure Gaussian noise.

    Runs the learned reverse process from t = T-1 down to t = 0. At each
    step the model predicts the noise in x_t, which is used to compute the
    posterior mean. Gaussian noise is added at every step except the last.

    Args:
        model            : noise predictor network, called as model(x, t)
        shape            : output shape, e.g. (16, 1, 28, 28) or (1000, 2)
        schedule         : noise schedule dict from make_schedule()
        n_snapshot_steps : number of intermediate states to record for visualization
        abl_no_noise     : if True, disable the stochastic noise term (sigma = 0)
        device           : torch device string

    Returns:
        x          : final generated samples, shape as specified
        trajectory : list of (timestep, x_snapshot) pairs for visualization
    """
    T = schedule['T']
    snapshot_interval = max(1, T // n_snapshot_steps)
    trajectory = []

    # Initialize from pure Gaussian noise — the starting point of generation.
    x = torch.randn(*shape, device=device)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Ask the model to predict the noise contained in x_t.
            predicted_noise = model(x, t_tensor)

            # Schedule values for this timestep, extracted as scalars.
            alpha_t        = schedule['alphas'][t]
            beta_t         = schedule['betas'][t]
            alpha_bar_t    = schedule['alphas_cumprod'][t]

            # ============ YOUR CODE HERE ============
            mean = torch.sqrt(1 / alpha_t) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

            if t > 0:
                sigma = torch.sqrt(beta_t)
                if abl_no_noise:
                    sigma = 0

                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

            # raise NotImplementedError
            # ============ END YOUR CODE ============

            if t % snapshot_interval == 0:
                trajectory.append((t, x.clone()))

    return x, trajectory


# ============================================================
# Convenience wrapper  (provided)
# ============================================================

def ddpm_sample_no_noise(model, shape, schedule, n_snapshot_steps=10, device='cpu'):
    """Runs ddpm_sample with the stochastic noise term disabled (sigma_t = 0).

    Provided as a convenience wrapper; delegates to your ddpm_sample implementation
    with abl_no_noise=True. Used in Task 3 to compare stochastic vs. deterministic sampling.

    Args:
        model            : noise predictor network, called as model(x, t)
        shape            : output shape, e.g. (16, 1, 28, 28) or (1000, 2)
        schedule         : noise schedule dict from make_schedule()
        n_snapshot_steps : number of intermediate states to record for visualization
        device           : torch device string

    Returns:
        x          : final generated samples, shape as specified
        trajectory : list of (timestep, x_snapshot) pairs for visualization
    """
    return ddpm_sample(model, shape, schedule,
                       n_snapshot_steps=n_snapshot_steps,
                       abl_no_noise=True,
                       device=device)
