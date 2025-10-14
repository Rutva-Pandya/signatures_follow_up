#!/bin/bash
#SBATCH --job-name=mamba_step1_arch
#SBATCH --output=logs/mamba_step1_%j.out
#SBATCH --error=logs/mamba_step1_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

echo "=========================================="
echo "STEP 1: Mamba Architecture Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda module
module load anaconda3/2024.02-1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Create environment if it doesn't exist
if ! conda env list | grep -q "^mamba_exp "; then
    echo "Creating conda environment 'mamba_exp'..."
    conda create -n mamba_exp python=3.10 -y
    conda activate mamba_exp
    echo "Installing packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers nnsight pandas numpy
    pip install mamba-ssm causal-conv1d
else
    conda activate mamba_exp
fi

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Set model and directories
MODEL_NAME="state-spaces/mamba-130m-hf"
REPO_DIR="/scratch/jhu35/rpandya4/model-human-processing"

cd $REPO_DIR || exit 1
echo ""
echo "Working directory: $(pwd)"

# Run the test
echo ""
echo "=========================================="
echo "Running Architecture Test..."
echo "=========================================="

python -c "
import sys
sys.path.insert(0, 'src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = '$MODEL_NAME'
EXPECTED_N_LAYERS = 24
EXPECTED_VOCAB_SIZE = 50277

print('\\n' + '='*80)
print('TEST 1: Loading Mamba Model')
print('='*80)

try:
    print(f'Loading model: {MODEL_NAME}')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    print('✓ Model loaded successfully')
except Exception as e:
    print(f'✗ FAILED: {e}')
    print('\\nEnsure mamba-ssm and causal-conv1d are installed:')
    print('  pip install mamba-ssm causal-conv1d>=1.1.0')
    sys.exit(1)

print('\\n' + '='*80)
print('TEST 2: Model Architecture')
print('='*80)

# Check backbone
assert hasattr(model, 'backbone'), 'Model does not have backbone attribute'
print('✓ Model has backbone attribute')

# Check layers
assert hasattr(model.backbone, 'layers'), 'Backbone does not have layers'
n_layers = len(model.backbone.layers)
print(f'✓ Model has backbone.layers: {n_layers} layers')
if n_layers != EXPECTED_N_LAYERS:
    print(f'  WARNING: Expected {EXPECTED_N_LAYERS} layers, got {n_layers}')

# Check norm_f
assert hasattr(model.backbone, 'norm_f'), 'Backbone does not have norm_f'
print('✓ Model has backbone.norm_f')

# Check lm_head
assert hasattr(model, 'lm_head'), 'Model does not have lm_head'
print('✓ Model has lm_head')

print('\\n' + '='*80)
print('TEST 3: Tokenizer')
print('='*80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
vocab_size = len(tokenizer)
print(f'✓ Tokenizer loaded')
print(f'  Vocabulary size: {vocab_size}')
if vocab_size != EXPECTED_VOCAB_SIZE:
    print(f'  WARNING: Expected {EXPECTED_VOCAB_SIZE}, got {vocab_size}')

print('\\n' + '='*80)
print('TEST 4: Forward Pass')
print('='*80)

test_text = 'The capital of France is'
inputs = tokenizer(test_text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

print('✓ Forward pass successful')
print(f'  Input shape: {inputs.input_ids.shape}')
print(f'  Output logits shape: {outputs.logits.shape}')

# Get prediction
predicted_token_id = outputs.logits[0, -1, :].argmax().item()
predicted_token = tokenizer.decode([predicted_token_id])
print(f'  Next token prediction: \\'{predicted_token}\\'')

print('\\n' + '='*80)
print('ALL TESTS PASSED ✓')
print('='*80)
print(f'\\nModel: {MODEL_NAME}')
print(f'Layers: {n_layers}')
print(f'Vocab: {vocab_size}')
print('\\nNext: Submit slurm_test_mamba_step2.sh')
print('='*80)
"

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
