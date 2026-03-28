#!/bin/bash
#SBATCH --job-name=destylize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:15:00
#SBATCH --qos=gpu-short

module purge
module load anaconda3/2025.6
conda activate flux

export HF_HOME="/scratch/gpfs/KAUSHIKS/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TORCH_HOME="/scratch/gpfs/KAUSHIKS/$USER/.cache/torch"

SNAP="/scratch/gpfs/KAUSHIKS/$USER/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
export FLUX_MODEL="$SNAP/flux1-dev.safetensors"
export FLUX_AE="$SNAP/ae.safetensors"

cd /scratch/gpfs/KAUSHIKS/$USER/flux-main

python run_destylize.py
