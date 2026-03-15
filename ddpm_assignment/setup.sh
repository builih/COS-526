#!/usr/bin/env bash
# Setup script for the DDPM assignment
set -e

ENV_NAME="ddpm"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo "Installing pytorch...."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
echo "Installing other dependencies..."
pip3 install matplotlib pillow scikit-learn


