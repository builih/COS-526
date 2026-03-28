# COS 526: Painting Destylization with FLUX.1-dev on Della

## Quick Reference


| Item               | Value                                                              |
| ------------------ | ------------------------------------------------------------------ |
| Della login node   | `della-gpu.princeton.edu`                                          |
| FLUX code on Della | `/scratch/gpfs/KAUSHIKS/yl4841/flux-main`                          |
| Local paintings    | `C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\paintings` |
| Conda environment  | `flux` (Python 3.10)                                               |
| GPU                | A100 80GB (`--constraint=gpu80`)                                   |
| Runtime            | ~41 seconds for 5 paintings at 512x512                             |


## Full Run Process

### 1. SSH into Della (PowerShell)

```
ssh -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 `
    -o Ciphers=aes256-ctr,aes192-ctr,aes128-ctr `
    -o KexAlgorithms=curve25519-sha256@libssh.org `
    yl4841@della-gpu.princeton.edu
```

### 2. Activate the flux environment (Della)

```
module purge
module load anaconda3/2025.6
conda activate flux
cd /scratch/gpfs/KAUSHIKS/$USER/flux-main
```

### 3. Upload paintings to Della (PowerShell)

```
scp -r -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\paintings" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/flux-main/
```

### 4. Upload scripts to Della (PowerShell)

```
scp -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\flux-main\run_destylize.py" `
  "C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\flux-main\run_destylize.slurm" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/flux-main/
```

### 5. Submit the job (Della)

```
sbatch run_destylize.slurm
```

### 6. Monitor the job (Della)

```
squeue -u $USER
```

### 7. Check results (Della)

```
ls -lh output_destylized/
tail -n 40 slurm-<JOBID>.out
sacct -j <JOBID> --format=JobID,Elapsed,State
```

### 8. Download results to local machine (PowerShell)

```
scp -r -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/flux-main/output_destylized/ `
  "C:\Users\yilin\COS-526\output_destylized\"
```

## Tuning Parameters

Override defaults by adding environment variables to `run_destylize.slurm` before the `python` line:


| Parameter   | Default | Effect                                                                                 |
| ----------- | ------- | -------------------------------------------------------------------------------------- |
| `STRENGTH`  | `0.70`  | How much to alter the painting (0.3 = subtle, 0.7 = aggressive, 0.8 = very aggressive) |
| `GUIDANCE`  | `5.5`   | How strongly to follow the "photorealistic" prompt                                     |
| `NUM_STEPS` | `35`    | Denoising steps (more = higher quality, slower)                                        |
| `IMG_SIZE`  | `512`   | Output resolution (512 fits in 80GB VRAM)                                              |
| `SEED`      | `42`    | Random seed for reproducibility                                                        |


Example override in Slurm script:

```bash
export STRENGTH=0.80
export GUIDANCE=7.0
python run_destylize.py
```

## Adding New Paintings

1. Place new `.jpg` files in `C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\paintings\`
2. (Optional) Add a custom prompt in `run_destylize.py` under `DEFAULT_PROMPTS` keyed by filename; otherwise the generic prompt is used
3. Re-upload paintings and scripts (steps 3-4 above)
4. Submit the job (step 5)

## Useful Della Commands

```bash
checkquota                           # check disk space
squeue -u $USER                      # check running jobs
scancel <JOBID>                      # cancel a job
sacct -u $USER --format=JobID,JobName,Elapsed,State -n | tail -10  # recent job history
sacct -j <JOBID> --format=JobID,Elapsed,State # job runtime
scontrol show job 5966900            # check job details (including scheduled time)
```

## Hugging Face Token

Set your token via environment variable (do not commit tokens to git):
```
export HF_TOKEN=<your-token-here>
```

### LYRA: Download results
```
scp -r -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/outputs/destylized_3d `
  "C:\Users\yilin\COS-526\lyra_recon_output\"
```

---

# COS 526: Lyra 3DGS Reconstruction on Della

## Quick Reference

| Item                | Value                                                                  |
| ------------------- | ---------------------------------------------------------------------- |
| Della login node    | `della-gpu.princeton.edu`                                              |
| Lyra code on Della  | `/scratch/gpfs/KAUSHIKS/yl4841/lyra-main`                              |
| Conda environment   | `lyra` (Python 3.10)                                                   |
| GPU                 | A100 80GB (`--constraint=gpu80`)                                       |
| SDG input           | Destylized images from FLUX stage in `destylized_inputs/`              |
| SDG output          | `destylized_sdg_output/<scene>/<view>/rgb/<scene>.mp4`                 |
| Recon output        | `outputs/destylized_3d/static_view_indices_fixed_5_0_1_2_3_4/`        |
| PLY files           | `<recon_output>/lyra_destylized_generated/gaussians_orig/*.ply`        |

## Overview

Lyra converts a single 2D image into a 3D Gaussian Splatting (3DGS) scene in two steps:

1. **SDG (Step 1)**: GEN3C generates 6 multi-view trajectory videos per input image
2. **Reconstruction (Step 2)**: A transformer decoder reconstructs 3DGS from the 6 videos and exports `.ply` files

## Prerequisites

All of the following are already set up on Della under `/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/`:

- Conda environment `lyra` with Apex, MoGe, Mamba, etc.
- Checkpoints: Cosmos tokenizer, GEN3C-7B, T5-11B, Lyra static weights
- HuggingFace models cached offline at `/scratch/gpfs/KAUSHIKS/yl4841/.cache/huggingface`

## Full Run Process

### 0. SSH into Della (PowerShell)

```
ssh -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 `
    -o Ciphers=aes256-ctr,aes192-ctr,aes128-ctr `
    -o KexAlgorithms=curve25519-sha256@libssh.org `
    yl4841@della-gpu.princeton.edu
```

### Step 1: SDG — Multi-View Video Generation

Each destylized image needs 6 camera trajectory videos. We run one Slurm job per image.

#### 1a. Upload destylized images (PowerShell)

```
scp -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "C:\Users\yilin\COS-526\output_destylized\*.jpg" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/destylized_inputs/
```

#### 1b. Submit SDG jobs (Della)

```bash
cd /scratch/gpfs/KAUSHIKS/yl4841/lyra-main

# Submit one job per image
for IMG in destylized_inputs/*.jpg; do
    IMG=$IMG sbatch run_lyra_sdg.slurm
done
```

Each job generates 6 trajectory videos under `destylized_sdg_output/<scene_name>/{0..5}/rgb/`.

#### 1c. Verify SDG output (Della)

```bash
# Should see 6 subdirs (0-5) per scene, each with rgb/<scene>.mp4
ls destylized_sdg_output/*/
```

### Step 2: Reconstruction — 3DGS from Multi-View Videos

#### 2a. Upload updated scripts if needed (PowerShell)

```
scp -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\lyra-main\src\models\data\radym_wrapper.py" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/src/models/data/
```

```
scp -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "C:\Users\yilin\COS-526\inverse_gaussian_style_transfer\lyra-main\run_lyra_recon.slurm" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/
```

#### 2b. Submit reconstruction job (Della)

```bash
cd /scratch/gpfs/KAUSHIKS/yl4841/lyra-main
rm -rf outputs/destylized_3d    # clean previous run (optional)
sbatch run_lyra_recon.slurm
```

#### 2c. Monitor (Della)

```bash
squeue -u $USER                          # check job status
scontrol show job <JOBID>                # detailed info & estimated start
tail -f slurm-<JOBID>.out               # watch live output
sacct -j <JOBID> --format=JobID,Elapsed,State  # after completion
```

### 3. Download Results (PowerShell)

#### PLY files (3DGS — use these for downstream style transfer)

```
mkdir C:\Users\yilin\Downloads\lyra_gaussians_orig -Force
scp -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/outputs/destylized_3d/static_view_indices_fixed_5_0_1_2_3_4/lyra_destylized_generated/gaussians_orig/*" `
  C:\Users\yilin\Downloads\lyra_gaussians_orig\
```

#### Rendered MP4s (visual inspection)

```
mkdir C:\Users\yilin\Downloads\lyra_rgb -Force
scp -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  "yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/outputs/destylized_3d/static_view_indices_fixed_5_0_1_2_3_4/lyra_destylized_generated/main_gaussians_renderings/*" `
  C:\Users\yilin\Downloads\lyra_rgb\
```

#### Everything (recursive)

```
scp -r -O `
  -o "MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1" `
  -o "Ciphers=aes256-ctr,aes192-ctr,aes128-ctr" `
  -o "KexAlgorithms=curve25519-sha256@libssh.org" `
  yl4841@della-gpu.princeton.edu:/scratch/gpfs/KAUSHIKS/yl4841/lyra-main/outputs/destylized_3d `
  "C:\Users\yilin\COS-526\lyra_recon_output\"
```

## Output Structure

```
outputs/destylized_3d/static_view_indices_fixed_5_0_1_2_3_4/lyra_destylized_generated/
├── gaussians/              # PyTorch .ply (torch.save — for code use only)
├── gaussians_orig/         # Real PLY files (open in SuperSplat / 3DGS viewers)
│   ├── gaussians_0.ply
│   ├── gaussians_1.ply
│   └── ...
├── main_gaussians_renderings/  # Rendered RGB videos from reconstructed 3DGS
│   ├── rgb_0.mp4
│   └── ...
├── full_output/            # Grid composite videos
├── raw/                    # Per-view rendered videos
└── meta/                   # Per-sample metadata JSON
```

**Important**: `gaussians/*.ply` are NOT real PLY files — they are `torch.save()` outputs with a `.ply` extension. Use `gaussians_orig/*.ply` for 3DGS viewers.

## Viewing Results

- **MP4 videos**: Open with VLC or Movies & TV on Windows
- **PLY files**: Open `gaussians_orig/*.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor) (browser, drag & drop)

## Key Config Files

| File | Purpose |
| ---- | ------- |
| `configs/demo/lyra_destylized.yaml` | Inference config (dataset, checkpoint, view indices, PLY export) |
| `configs/training/3dgs_res_704_1280_views_121_multi_6_prune.yaml` | Model config (6 views, 121 frames, resolution) |
| `src/models/data/registry.py` | Dataset registry (`lyra_destylized_generated` entry) |
| `src/models/data/radym_wrapper.py` | Multi-view data loader (modified for our nested layout) |
| `run_lyra_recon.slurm` | Slurm script for reconstruction (Step 2) |

## Adding New Scenes

1. Place new destylized `.jpg` images in `destylized_inputs/` on Della
2. Run SDG (Step 1) for each new image
3. The dataset registry auto-discovers all scenes under `destylized_sdg_output/` — no config changes needed
4. Run reconstruction (Step 2)

## Useful Della Commands

```bash
squeue -u $USER                      # check running jobs
scancel <JOBID>                      # cancel a job
squeue --start -j <JOBID>           # estimated start time
sacct -j <JOBID> --format=JobID,Elapsed,State  # runtime after completion
tail -n 40 slurm-<JOBID>.out        # check output log
checkquota                           # check disk space
```

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| `Corrupted MAC on input` during SCP | Add `-o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 -o Ciphers=aes256-ctr,aes192-ctr,aes128-ctr -o KexAlgorithms=curve25519-sha256@libssh.org` |
| `einops.EinopsError: 186 not divisible by 5` | `radym_wrapper.py` is outdated — re-upload the fixed version (dedup + camera count fix) |
| All PLY files are the same scene | `radym_wrapper.py` dedup fix not applied — re-upload and re-run |
| PLY won't open in viewer | Use `gaussians_orig/*.ply` (not `gaussians/*.ply`) in a 3DGS viewer like SuperSplat |
| `EADDRINUSE` during SDG | Random `--master_port` already handled in `run_lyra_sdg.slurm` |
| HuggingFace timeouts on compute node | `HF_HUB_OFFLINE=1` is set in Slurm scripts |

---


