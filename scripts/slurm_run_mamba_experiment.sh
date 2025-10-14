#!/bin/bash
#SBATCH --job-name=mamba_experiment
#SBATCH --output=logs/mamba_exp_%A_%a.out
#SBATCH --error=logs/mamba_exp_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --array=0-4

# This script runs Mamba experiments on all 5 tasks using job arrays
# Each array task handles one experiment

echo "=========================================="
echo "Mamba Experiment Runner"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
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
    pip install transformers==4.36.2 pandas numpy

    # Install nnsight (must be after torch)
    pip install nnsight

    # Try mamba-ssm installation methods
    echo "Attempting mamba-ssm installation..."

    # Method 1: Try pre-built wheel
    pip install mamba-ssm==1.2.0.post1 --no-deps 2>/dev/null
    if python -c "import mamba_ssm" 2>/dev/null; then
        echo "✓ Method 1 succeeded (pre-built wheel)"
        pip install causal-conv1d==1.1.1 --no-deps 2>/dev/null || pip install causal-conv1d
    else
        # Method 2: Install with dependencies, let it compile if needed
        echo "Method 1 failed. Trying Method 2 (may compile from source)..."
        pip install causal-conv1d>=1.1.0
        pip install mamba-ssm
    fi

    # Final check
    if ! python -c "import mamba_ssm; import causal_conv1d" 2>/dev/null; then
        echo "ERROR: Could not install mamba-ssm and causal-conv1d"
        echo "Trying one more time with --no-build-isolation..."
        pip install --no-build-isolation causal-conv1d mamba-ssm
    fi

    # Verify all imports work
    python -c "import torch; import transformers; import nnsight; import mamba_ssm; import causal_conv1d; print('✓ All packages installed successfully')"
else
    conda activate mamba_exp
fi

# Configuration
MODEL="state-spaces/mamba-130m-hf"
REPO_DIR="/scratch/jhu35/rpandya4/model-human-processing"
STIMULI_DIR="$REPO_DIR/data/stimuli"
OUTPUT_DIR="$REPO_DIR/data/model_output"

# Task array
TASKS=(
    "capitals-recall"
    "capitals-recognition"
    "animals"
    "gender"
    "syllogism"
)

# Get current task
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Task: $TASK"
echo "  Stimuli dir: $STIMULI_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo ""

cd $REPO_DIR || exit 1

# Run experiment
echo "=========================================="
echo "Running experiment..."
echo "=========================================="

python src/run_experiment.py \
    --model $MODEL \
    --task $TASK \
    --stimuli_dir $STIMULI_DIR \
    --output_dir $OUTPUT_DIR

exit_code=$?

echo ""
echo "=========================================="
echo "Experiment finished: $(date)"
echo "Exit code: $exit_code"
echo "Task: $TASK"
echo "=========================================="

# Check if output file was created
SAFE_MODEL_NAME=$(basename $MODEL)
OUTPUT_FILE="$OUTPUT_DIR/logit_lens/${TASK}_${SAFE_MODEL_NAME}.csv"

if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file created: $OUTPUT_FILE"
    wc -l $OUTPUT_FILE
else
    echo "✗ Output file not found: $OUTPUT_FILE"
    exit_code=1
fi

exit $exit_code
