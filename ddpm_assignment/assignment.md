# COS526/ECE576 Assignment 3：Denoising Diffusion Probabilistic Models (DDPM)


# 1. Task Overview

Diffusion models are one of the most powerful modern generative models.
Unlike GANs, which learn to directly map noise to data, diffusion models:

1. Gradually destroy data by adding Gaussian noise
2. Learn to reverse this destruction process step by step

In this assignment, you will implement the core logic of DDPM and train diffusion models on low-dimensional and image datasets.


## Directory Structure

```
├── src/
│   ├── ddpm.py              ← Your implementation goes here
│   ├── util.py              ← Networks & data loaders (provided)
│   └── visualization.py     ← Plotting utilities (provided)
├── scripts/
│   ├── 1_forward_process.py
│   ├── 2_training.py
│   ├── 3_sampling.py
├── run_all.py
└── assignment.md
```

You will implement the required functions in `src/ddpm.py`. Everything else is provided.

## Environment setup

```bash
# run the setup bash file:
bash setup.sh

# or run each command seperately:
conda create -n ddpm python=3.10 -y
conda activate ddpm
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install matplotlib pillow scikit-learn
```

## Running the code

```bash
conda activate ddpm

python scripts/1_forward_process.py
python scripts/2_training.py
python scripts/3_sampling.py
```


# 2. Implementing DDPM

This section walks through the mathematics of DDPM and connects each concept directly to the functions you will implement.


## 2.1 The Forward Process

### Markov Transition

At each timestep, data is slightly scaled down and Gaussian noise is added:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1 - \beta_t}\; x_{t-1},\; \beta_t I\right)$$

**In code:** `forward_diffusion_step(x_prev, t, noise, schedule)` implements:

$$x_t = \sqrt{1 - \beta_t}\; x_{t-1} + \sqrt{\beta_t}\; \epsilon$$

### Reparameterization

A key insight: we can jump directly to any timestep $t$ without iterating through all intermediate steps.

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\; x_0,\; (1 - \bar{\alpha}_t) I\right)$$

Using the reparameterization trick:

$$x_t = \sqrt{\bar{\alpha}_t}\; x_0 + \sqrt{1 - \bar{\alpha}_t}\; \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**In code:** `q_sample(x_0, t, noise, schedule)` implements this closed form. It is used during training to generate noisy samples at arbitrary timesteps without iterating.


## 2.2 DDPM Training

Instead of learning full distributions, DDPM simplifies training to predicting noise.

A neural network $\epsilon_\theta(x_t, t)$ is trained to predict the noise that was added. The loss is simply the mean squared error between the true and predicted noise:

$$\mathcal{L} = \mathbb{E}_{x_0,\, \epsilon,\, t}\!\left[\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|^2\right]$$

**Training procedure:**
1. Sample clean data $x_0$
2. Sample a random timestep $t \sim \text{Uniform}\{0, \ldots, T-1\}$
3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
4. Generate the noisy sample $x_t$ using `q_sample`
5. Predict the noise with the model
6. Minimize the MSE loss

**In code:** `ddpm_train_step(model, x, schedule)` implements this procedure. It is provided, but it depends on your `q_sample`.

**Neural networks provided in `util.py`:**
- `NoisePredictor` — an MLP for 1D/2D data with sinusoidal timestep embeddings
- `SimpleUNet` — a U-Net for 28×28 images with skip connections and timestep injection


## 2.3 The Revsere Process (DDPM Sampling)

To generate new data, start from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise.

At each step, the learned reverse distribution is:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(\mu_\theta(x_t, t),\; \sigma_t^2 I\right)$$

where the predicted mean is:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\; \epsilon_\theta(x_t, t) \right)$$

and the variance is:

$$\sigma_t^2 = \beta_t\$$


The full sampling step:
- If $t > 0$: &nbsp; $x_{t-1} = \mu_\theta + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$
- If $t = 0$: &nbsp; $x_0 = \mu_\theta$ &nbsp; (no noise at the final step)

**In code:** `ddpm_sample(model, shape, schedule, ...)` implements this loop from $t = T-1$ down to $0$.

To understand why the stochastic noise term matters, we also provide `ddpm_sample_no_noise`, which runs the same loop but sets $\sigma_t = 0$ at every step. This turns sampling into a deterministic process — you will visualize both and compare the results in Part 3.

### Noise Schedule

We use a **linear schedule** where $\beta_t$ increases from $10^{-4}$ to $0.02$ over $T = 1000$ steps. This controls how quickly data is destroyed — slow enough that each reverse step remains learnable.


# 3. The Scripts

## Part 1 — Forward Diffusion

Run the forward diffusion process on all three datasets and watch data dissolve into noise.

**Outputs** (`scripts/outputs/`):
- `1_forward_1d.gif` — a bimodal 1D distribution dissolving into Gaussian noise
- `1_forward_2d.gif` — a 2D Swiss Roll scattering into uniform noise
- `1_forward_image.gif` — Fashion-MNIST images progressively corrupted

Observe how the data structure disappears as $t$ increases until the samples are indistinguishable from random noise.


## Part 2 — Training

Train the noise-prediction networks on 2D and image data. The training loop iterates over the dataset, adds noise at random timesteps, and minimizes the MSE between true and predicted noise.

**Outputs** (`scripts/outputs/`):
- `2_loss_2d.png`, `2_loss_image.png` — training loss curves
- `2_training_2d.gif`, `2_training_image.gif` — one-step denoising quality across epochs

The training GIFs show how the model's ability to remove noise improves over time.


## Part 3 — Sampling & Noise Analysis

Load the trained models and run the full reverse diffusion process — starting from pure noise and denoising step by step.

The script also runs sampling *without* the stochastic noise term (setting $\sigma_t = 0$) and shows both versions side by side so you can see the effect directly.

**Outputs** (`scripts/outputs/`):
- `3_sampling_2d.gif` — reverse diffusion trajectory from noise to Swiss Roll
- `3_sampling_image.gif` — reverse diffusion trajectory from noise to images
- `3_noise_comparison_2d.gif` — side-by-side: with noise vs. without noise (2D)
- `3_noise_comparison_image.gif` — side-by-side: with noise vs. without noise (images)



# 4. Writeup Questions


**Q1 — Forward Process**
As the timestep increases in the forward diffusion process, what happens to the signal-to-noise ratio? Why does the process eventually produce samples that resemble pure Gaussian noise?


**Q2 — DDPM Training**
Based on the one-step denoising visualizations, describe how the denoising quality changes (a) across training epochs and (b) across different timesteps t. Why is one-step denoising alone insufficient for generating high-quality samples?

**Q3 — Stochasticity in the Reverse Process**
Why do we add random noise at every reverse step except when t = 0? What happens if the noise term is removed during sampling? 


# 5. Bonus task

DDPM is a general framework: the same sampling algorithm works for any dataset and data modality

Pick any dataset that interests you (anything except Fashion-MNIST and MNIST) and apply the DDPM to it.  

**You are free to use any LLM to help with this part.**


# 6. Submission

In a zipped file: 
- `src/ddpm.py` — your completed implementation
- Written answers to Q1–Q3
- Output GIFs from Parts 1–3
- (Code and report for the bonus task)