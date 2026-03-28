# How the Inverse Gaussian Style Transfer Pipeline Works

## Part 1: FLUX.1-dev Destylization (Painting → Photorealistic Image)

We are NOT doing text-to-image from scratch. We are doing **img2img** (also called **SDEdit**) -- starting from the painting and partially re-generating it toward a photorealistic target. The text prompt never "generates the image from words alone"; it **steers** the denoising toward realism.

### The 4 Phases

**Phase 1 -- Encode the text prompt into embeddings**

- You write a prompt like *"a real photograph of green rolling hills... DSLR photograph... no brushstrokes"*
- Two text encoders (T5-XXL and CLIP) convert this into dense vector embeddings
- These embeddings don't generate anything yet -- they're just a **numeric representation of the target description** that the model will use as guidance

**Phase 2 -- Encode the painting into latent space**

- The painting (pixel image) is fed through FLUX's **autoencoder (AE)**
- This compresses the 512x512 RGB image into a small latent tensor z0 (e.g. shape [1, 16, 32, 32])
- z0 is a compact representation of the painting -- it captures all the content but in a space the flow model can work with

**Phase 3 -- Add noise, then denoise toward the prompt (the core step)**

This is the **SDEdit** trick. FLUX is a **rectified flow** model, meaning it learns a straight-line path from pure noise (t=1) to clean image (t=0):

    z_noisy = (1 - strength) * z0 + strength * noise

- **strength = 0.7** means: mix 30% of the painting's latent with 70% random noise
- This **partially destroys** the painting's style (brushstrokes, color palette) while **retaining** its composition (hills here, trees there, sky above)
- Then the **flow model** (12B parameter transformer) iteratively denoises z_noisy back to a clean image, but now **conditioned on the text prompt** ("real photograph, no brushstrokes...")
- The text embeddings from Phase 1 tell the model **what kind of image to reconstruct** -- so it fills in photorealistic textures instead of recovering the original paint

**Key hyperparameters:**

| Parameter       | Value | What it does                                                                                  |
|-----------------|-------|-----------------------------------------------------------------------------------------------|
| **strength**    | 0.7   | How much noise to add. Higher = more destruction of original style, more photorealistic but less faithful to composition |
| **guidance**    | 5.5   | How strongly to follow the text prompt. Higher = more photorealistic but can look artificial   |
| **num_steps**   | 35    | Denoising iterations. More = higher quality, diminishing returns past ~30                     |

**Phase 4 -- Decode back to pixels**

- The denoised latent is passed through the AE's **decoder** to produce a 512x512 RGB image
- Saved as destylized_<name>.jpg

### FLUX Summary Diagram

    Painting → AE encode → latent z0
                             ↓
                  z_noisy = 30% painting + 70% noise
                             ↓
                  Flow model denoises, guided by text prompt
                             ↓
                  Clean latent (photorealistic content)
                             ↓
                  AE decode → Destylized photograph

The painting gives **structure/composition**, the text gives **"make it a photo"**, and strength controls the balance.

---

## Part 2: Lyra (Destylized Image → 3D Gaussian Splatting)

Lyra converts a **single 2D image** into a **3D Gaussian Splatting scene** in two stages.

### Stage 1: SDG (Synthetic Data Generation) -- Multi-View Videos

**Input**: One destylized image (e.g. destylized_green_land.jpg)

**What happens:**

1. **GEN3C** (a camera-controlled video diffusion model based on NVIDIA Cosmos) takes the single image and generates a **video** of the scene as if a camera is orbiting around it
2. It does this **6 times**, each with a **different camera trajectory** (orbit angles 0-5), producing 6 MP4 videos of ~121 frames each
3. These are "imagined" views -- GEN3C hallucinates what the scene looks like from the side, behind, above, etc., using priors learned from millions of videos

**Output**: 6 videos per scene at destylized_sdg_output/<scene>/{0..5}/rgb/<scene>.mp4

Think of it as: the model **imagines walking around** your 2D photo and filming it from 6 different angles.

### Stage 2: Reconstruction -- 3DGS from Videos

**Input**: The 6 trajectory videos (186 total frames = 6 views × 31 subsampled frames)

**What happens:**

1. Each video is tokenized into **latent space** by the Cosmos VAE (video autoencoder)
2. All 6 views are **concatenated** into one tensor: [batch=1, frames=186, channels, H, W]
3. A **transformer decoder** (trained on synthetic multi-view data) processes all 186 frames jointly -- attending across both time and viewpoints
4. Instead of outputting pixels, the decoder outputs **3D Gaussian parameters**: for each Gaussian, it predicts position (x,y,z), opacity, scale (3 values), rotation (4 quaternion values), and color (3 spherical harmonic coefficients) = **14 values per Gaussian**
5. These Gaussians are saved as a .ply file

**Output**: gaussians_orig/gaussians_0.ply -- a standard 3DGS PLY file you can render in real-time

### Lyra Summary Diagram

    Destylized image
          ↓
      GEN3C (×6 camera trajectories)
          ↓
      6 orbit videos (121 frames each)
          ↓
      Cosmos VAE encodes to latents
          ↓
      Transformer processes all 6 views jointly
          ↓
      Predicts per-Gaussian: (xyz, opacity, scale, rotation, color)
          ↓
      Exports .ply → viewable 3DGS scene

### Why destylize first?

GEN3C was trained on **real-world videos**, not paintings. Feeding it a painting with heavy brushstrokes would produce inconsistent multi-view videos (it doesn't know how brushstrokes look from the side). Destylizing first gives GEN3C a photorealistic input it can reason about geometrically, producing cleaner 3D reconstructions.

---

## Full Pipeline End-to-End

    Stylized Painting
          ↓
    [FLUX.1-dev SDEdit] — destylize to photorealistic image
          ↓
    Destylized Photograph
          ↓
    [Lyra Stage 1: GEN3C SDG] — generate 6 multi-view orbit videos
          ↓
    6 × 121-frame MP4 videos
          ↓
    [Lyra Stage 2: 3DGS Reconstruction] — transformer predicts Gaussians
          ↓
    3D Gaussian Splatting .ply file
          ↓
    [Inverse Style Transfer (next step)] — re-apply artistic style to 3DGS
          ↓
    Stylized 3D Scene
