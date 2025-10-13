#!/bin/bash
#SBATCH --job-name=install-mamba
#SBATCH --output=logs/install_mamba_%j.out
#SBATCH --error=logs/install_mamba_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Script to install Mamba SSM kernels for faster inference
# Run with: sbatch cluster/install_mamba_kernels.sh

set -e

echo "Installing Mamba SSM kernels..."

# Load modules
module load anaconda3/2024.02-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate model-human-processing

# Install required packages
echo "Installing mamba-ssm..."
pip install mamba-ssm

echo "Installing causal-conv1d..."
pip install causal-conv1d

echo ""
echo "=========================================="
echo "Installation complete!"
echo "Verify installation by running:"
echo "python -c 'import mamba_ssm; import causal_conv1d; print(\"Success!\")'"
echo "=========================================="
