#!/bin/bash
#SBATCH --job-name=mamba_step2_nnsight
#SBATCH --output=logs/mamba_step2_%j.out
#SBATCH --error=logs/mamba_step2_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

echo "=========================================="
echo "STEP 2: nnsight & Hidden Extraction Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

mkdir -p logs

# Load environment
module load anaconda3/2024.02-1
conda activate mamba_exp

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
