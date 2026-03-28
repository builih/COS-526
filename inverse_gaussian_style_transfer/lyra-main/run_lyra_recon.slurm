#!/bin/bash
#SBATCH --job-name=lyra-recon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --qos=gpu-medium

module purge
module load anaconda3/2025.6
conda activate lyra

export HF_HOME="/scratch/gpfs/KAUSHIKS/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TORCH_HOME="/scratch/gpfs/KAUSHIKS/$USER/.cache/torch"
export HF_HUB_OFFLINE=1

cd /scratch/gpfs/KAUSHIKS/$USER/lyra-main

CONFIG="${CONFIG:-configs/demo/lyra_destylized.yaml}"

accelerate launch --config_file configs/accelerate/accelerate_config_single.yaml \
    sample.py --config "$CONFIG"

echo "3DGS reconstruction complete."
