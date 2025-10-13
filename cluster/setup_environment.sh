#!/bin/bash
# Setup script for DSAI cluster environment
# Run this once to set up conda and the Python environment

set -e

echo "=========================================="
echo "Setting up environment on DSAI cluster"
echo "=========================================="

# Load anaconda module
echo "Loading anaconda module..."
module load anaconda3/2024.02-1

# Create conda environment
ENV_NAME="model-human-processing"
echo "Creating conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists. Remove it? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        conda env remove -n $ENV_NAME
    else
        echo "Skipping environment creation."
        exit 0
    fi
fi

# Create environment with Python 3.10
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace and transformers
echo "Installing HuggingFace libraries..."
pip install transformers datasets accelerate
pip install sentencepiece protobuf

# Install tuned-lens for logit lens experiments
echo "Installing tuned-lens..."
pip install tuned-lens

# Install other dependencies
echo "Installing other dependencies..."
pip install pandas numpy scipy matplotlib seaborn
pip install jupyter ipython

# Install vision dependencies
pip install timm pillow

echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment in future sessions:"
echo "  module load anaconda3/2024.02-1"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
