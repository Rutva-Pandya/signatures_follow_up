#!/bin/bash
#SBATCH --job-name=test-setup
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Quick test to verify environment setup

set -e

echo "=========================================="
echo "Testing cluster setup"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Load modules
module load anaconda3/2024.02-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate model-human-processing

echo ""
echo "Python version:"
python --version

echo ""
echo "Testing PyTorch and CUDA:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

echo ""
echo "Testing transformers:"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ""
echo "Testing small model load (gpt2):"
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('gpt2'); model = AutoModelForCausalLM.from_pretrained('gpt2'); print('Successfully loaded GPT-2!')"

echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "=========================================="
echo "Setup test complete!"
echo "=========================================="
