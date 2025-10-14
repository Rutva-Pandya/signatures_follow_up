#!/bin/bash
#SBATCH --job-name=mamba_step2_nnsight
#SBATCH --output=logs/mamba_step2_%j.out
#SBATCH --error=logs/mamba_step2_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s

echo "=========================================="
echo "STEP 2: nnsight & Hidden Extraction Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

mkdir -p logs

# Load environment
module load anaconda3/2024.02-1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Create environment if it doesn't exist
if ! conda env list | grep -q "^mamba_exp "; then
    echo "Creating conda environment 'mamba_exp'..."
    conda create -n mamba_exp python=3.10 -y
    conda activate mamba_exp
    echo "Installing packages..."


    # Install core packages with specific versions to avoid conflicts
    echo "Installing PyTorch and dependencies..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install transformers==4.36.2 pandas numpy tuned-lens

    # Install nnsight (must be after torch)
    pip install nnsight

    # Install causal-conv1d first (compatible with torch 2.1.2)
    echo "Installing causal-conv1d..."
    pip install causal-conv1d==1.1.0

    # Install mamba-ssm with exact torch version preserved
    echo "Installing mamba-ssm..."
    pip install mamba-ssm==1.2.0.post1

    # Verify torch version wasn't upgraded
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "error")
    if [[ ! "$TORCH_VERSION" =~ ^2\.1\.2 ]]; then
        echo "ERROR: Torch version changed to $TORCH_VERSION, reinstalling..."
        pip install --force-reinstall torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    fi

    # Verify all imports work
    echo "Verifying package installation..."
    python -c "import torch; print(f'torch: {torch.__version__}')"
    python -c "import transformers; import nnsight; import tuned_lens; print('✓ Core packages OK')"

    # Check mamba-ssm (may need CUDA libraries loaded)
    if ! python -c "import mamba_ssm; import causal_conv1d" 2>/dev/null; then
        echo "WARNING: mamba_ssm import failed, but may work when CUDA is loaded on GPU node"
    else
        echo "✓ mamba-ssm and causal-conv1d OK"
    fi
fi

MODEL_NAME="state-spaces/mamba-130m-hf"
REPO_DIR="/scratch/jhu35/rpandya4/model-human-processing"
cd $REPO_DIR || exit 1

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

python -c "
import sys
sys.path.insert(0, 'src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = '$MODEL_NAME'
TEST_TEXT = 'The capital of France is'

print('='*80)
print('Loading model and tokenizer...')
print('='*80)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'✓ Loaded {MODEL_NAME}')
print(f'  Layers: {len(model.backbone.layers)}')

print('\\n' + '='*80)
print('TEST 1: nnsight Import')
print('='*80)
try:
    from nnsight import LanguageModel
    print('✓ nnsight imported successfully')
except ImportError as e:
    print(f'✗ FAILED: {e}')
    print('Install with: pip install nnsight')
    sys.exit(1)

print('\\n' + '='*80)
print('TEST 2: nnsight Wrapper')
print('='*80)
try:
    nnsight_model = LanguageModel(model, tokenizer=tokenizer)
    print('✓ nnsight wrapper created')
except Exception as e:
    print(f'✗ FAILED: {e}')
    print('nnsight may not support Mamba - will try manual hooks')
    sys.exit(1)

print('\\n' + '='*80)
print('TEST 3: Hidden State Extraction')
print('='*80)
try:
    with torch.no_grad():
        with nnsight_model.trace(TEST_TEXT) as tracer:
            layers = nnsight_model.backbone.layers
            hiddens_l = [
                layer.output[0][0, :].unsqueeze(1) for layer in layers
            ]
            hiddens = torch.cat(hiddens_l, dim=1).save()

    print(f'✓ Hidden extraction successful')
    print(f'  Shape: {hiddens.shape}')
    n_tokens, n_layers, hidden_dim = hiddens.shape
    print(f'  Tokens: {n_tokens}, Layers: {n_layers}, Hidden dim: {hidden_dim}')
except Exception as e:
    print(f'✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\\n' + '='*80)
print('TEST 4: Logit Lens Compatibility')
print('='*80)
try:
    normed = model.backbone.norm_f(hiddens)
    logits = model.lm_head(normed)
    print(f'✓ Logit lens application successful')
    print(f'  Logits shape: {logits.shape}')

    logprobs = logits.log_softmax(dim=-1)
    print(f'✓ Log softmax successful')
except Exception as e:
    print(f'✗ FAILED: {e}')
    sys.exit(1)

print('\\n' + '='*80)
print('ALL TESTS PASSED ✓')
print('='*80)
print('\\nHidden extraction works!')
print('Next: Submit slurm_test_mamba_step3.sh')
print('='*80)
"

echo ""
echo "Job finished: $(date)"
