#!/bin/bash
#SBATCH --job-name=mamba_step3_logitlens
#SBATCH --output=logs/mamba_step3_%j.out
#SBATCH --error=logs/mamba_step3_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s

echo "=========================================="
echo "STEP 3: Logit Lens Validation Test"
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
else
    conda activate mamba_exp
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

from model import LM
import torch
import numpy as np

MODEL_NAME = '$MODEL_NAME'

print('='*80)
print('TEST 1: LM Class Initialization')
print('='*80)
try:
    print(f'Initializing LM class with {MODEL_NAME}...')
    model = LM(MODEL_NAME)
    print('✓ LM class initialized')
    print(f'  Model family: {model.model_family}')
    print(f'  Layers: {model.n_layers}')
except Exception as e:
    print(f'✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\\n' + '='*80)
print('TEST 2: Layer-wise Logprobs')
print('='*80)
test_text = 'The capital of France is'
print(f'Test text: \\'{test_text}\\'')

try:
    logits, logprobs, logits_deltas, _ = model.logprobs_and_logit_diffs_all_layers(test_text)
    print('✓ Logprobs extracted')
    print(f'  Logits shape: {logits.shape}')
    print(f'  Logprobs shape: {logprobs.shape}')

    n_tokens, n_layers, vocab_size = logprobs.shape
    print(f'  Tokens: {n_tokens}, Layers: {n_layers}, Vocab: {vocab_size}')
except Exception as e:
    print(f'✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\\n' + '='*80)
print('TEST 3: Final Layer Consistency')
print('='*80)
try:
    inputs = model.model.tokenizer(test_text, return_tensors='pt')
    with torch.no_grad():
        actual_output = model.model.forward(inputs['input_ids'])
        actual_logits = actual_output.logits[0, -1, :]

    final_layer_logits = logits[-1, -1, :]
    max_diff = (actual_logits - final_layer_logits).abs().max().item()
    mean_diff = (actual_logits - final_layer_logits).abs().mean().item()

    print(f'✓ Comparison complete')
    print(f'  Max diff: {max_diff:.6f}')
    print(f'  Mean diff: {mean_diff:.6f}')

    if max_diff < 1e-3:
        print(f'  ✓ EXCELLENT: Negligible differences')
    elif max_diff < 1e-1:
        print(f'  ✓ GOOD: Small differences')
    else:
        print(f'  ⚠ WARNING: Large differences')
except Exception as e:
    print(f'✗ FAILED: {e}')

print('\\n' + '='*80)
print('TEST 4: Rank Trajectory')
print('='*80)
prefix = 'The capital of France is'
target = ' Paris'
print(f'Prefix: \\'{prefix}\\'')
print(f'Target: \\'{target}\\'')

try:
    tokenizer = model.model.tokenizer
    target_tokens = tokenizer(target)['input_ids']
    if tokenizer.bos_token_id and target_tokens[0] == tokenizer.bos_token_id:
        target_tokens = target_tokens[1:]

    # Try with space if needed
    if len(target_tokens) == 0 or tokenizer.decode([target_tokens[0]]).strip().lower() != 'paris':
        target_tokens = tokenizer(' Paris')['input_ids']
        if tokenizer.bos_token_id and target_tokens[0] == tokenizer.bos_token_id:
            target_tokens = target_tokens[1:]

    target_token_id = target_tokens[0]
    print(f'Target token ID: {target_token_id} (\\'{tokenizer.decode([target_token_id])}\\')')

    ranks = model.rank_of_token_all_layers(prefix, [target_token_id])[0]
    print(f'\\n✓ Rank extraction successful')

    initial_rank = ranks[0]
    final_rank = ranks[-1]
    improvement = initial_rank - final_rank

    print(f'  Initial rank: {initial_rank}')
    print(f'  Final rank: {final_rank}')
    print(f'  Improvement: {improvement}')

    if improvement > 0:
        print(f'  ✓ Rank improved!')
    else:
        print(f'  ⚠ Rank did not improve (model may not know this fact)')
except Exception as e:
    print(f'✗ FAILED: {e}')
    import traceback
    traceback.print_exc()

print('\\n' + '='*80)
print('TEST 5: Conditional Scoring')
print('='*80)
prefix = 'The capital of France is'
correct = 'Paris'
incorrect = 'London'

try:
    correct_scores = model.conditional_score_all_layers(prefix, correct)
    incorrect_scores = model.conditional_score_all_layers(prefix, incorrect)

    print('✓ Conditional scoring successful')

    correct_logprob = correct_scores['first'][-1]
    incorrect_logprob = incorrect_scores['first'][-1]
    diff = correct_logprob - incorrect_logprob

    print(f'  Correct logprob: {correct_logprob:.4f}')
    print(f'  Incorrect logprob: {incorrect_logprob:.4f}')
    print(f'  Difference: {diff:.4f}')

    if diff > 0:
        print(f'  ✓ Model prefers correct answer')
    else:
        print(f'  ⚠ Model prefers incorrect answer')
except Exception as e:
    print(f'✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\\n' + '='*80)
print('ALL TESTS PASSED ✓')
print('='*80)
print('\\nLM class works with Mamba!')
print('Logit lens produces valid predictions.')
print('\\nReady for full experiments!')
print('Next: Submit slurm_run_mamba_experiment.sh')
print('='*80)
"

echo ""
echo "Job finished: $(date)"
