#!/bin/bash
#SBATCH --job-name=lyra-sdg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
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

MASTER_PORT=$((29500 + RANDOM % 1000))

INPUT_DIR="${INPUT_DIR:-destylized_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-destylized_sdg_output}"

if [ -n "$IMG" ]; then
    img="$INPUT_DIR/$IMG"
    if [ ! -f "$img" ]; then
        echo "ERROR: Image not found: $img"
        exit 1
    fi
    basename=$(basename "$img" | sed 's/\.[^.]*$//')
    out="$OUTPUT_DIR/$basename"
    echo "=== Processing single image: $basename ==="
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
        cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
        --checkpoint_dir checkpoints \
        --num_gpus 1 \
        --input_image_path "$img" \
        --video_save_folder "$out" \
        --foreground_masking \
        --multi_trajectory \
        --total_movement_distance_factor 1.0 \
        --offload_diffusion_transformer \
        --offload_tokenizer \
        --offload_text_encoder_model \
        --offload_prompt_upsampler \
        --offload_guardrail_models \
        --disable_guardrail \
        --disable_prompt_encoder
    echo "=== Done: $basename ==="
else
    for img in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.png; do
        [ -f "$img" ] || continue
        basename=$(basename "$img" | sed 's/\.[^.]*$//')
        out="$OUTPUT_DIR/$basename"
        if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null)" ]; then
            echo "Skipping $basename (already generated)"
            continue
        fi
        echo "=== Processing: $basename ==="
        CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
            cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
            --checkpoint_dir checkpoints \
            --num_gpus 1 \
            --input_image_path "$img" \
            --video_save_folder "$out" \
            --foreground_masking \
            --multi_trajectory \
            --total_movement_distance_factor 1.0 \
            --offload_diffusion_transformer \
            --offload_tokenizer \
            --offload_text_encoder_model \
            --offload_prompt_upsampler \
            --offload_guardrail_models \
            --disable_guardrail \
            --disable_prompt_encoder
        echo "=== Done: $basename ==="
    done
fi

echo "All SDG generation complete."
