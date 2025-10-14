#!/bin/bash
#SBATCH --job-name=mamba_experiment
#SBATCH --output=logs/mamba_exp_%A_%a.out
#SBATCH --error=logs/mamba_exp_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --array=0-1

# This script runs Mamba experiments on all 5 tasks using job arrays
# Array job 0: runs first 4 tasks sequentially (capitals-recall, capitals-recognition, animals, gender)
# Array job 1: runs syllogism task separately (it's much more intensive)
#
# Usage:
#   sbatch scripts/slurm_run_mamba_experiment.sh <MODEL_NAME>
#
# Supported models:
#   state-spaces/mamba-130m-hf   (works with 1 GPU, 128GB RAM)
#   state-spaces/mamba-370m-hf   (may OOM on syllogism - use --reduce_precision flag)
#   state-spaces/mamba-1.4b-hf   (may OOM on syllogism - use --reduce_precision flag)
#
# For larger models, consider using quantization:
#   Modify the python command below to add --reduce_precision flag

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

    # Install core packages
    echo "Installing PyTorch and dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pandas numpy tuned-lens

    # Install transformers with Mamba support (requires >= 4.39.0)
    pip install "transformers>=4.39.0"

    # Install latest nnsight
    pip install nnsight

    # Verify all imports work
    echo ""
    echo "Verifying package installation..."
    python -c "import torch; print(f'✓ torch: {torch.__version__}')"
    python -c "import transformers; print(f'✓ transformers: {transformers.__version__}')"
    python -c "import nnsight; print('✓ nnsight OK')"
    python -c "import tuned_lens; print('✓ tuned_lens OK')"
    python -c "import pandas, numpy; print('✓ pandas and numpy OK')"
    echo ""
    echo "✓ All packages installed successfully"
    echo "Note: Using transformers native Mamba support (no mamba-ssm needed)"
else
    conda activate mamba_exp
fi

# Configuration
# Model must be passed as first argument
if [ -z "$1" ]; then
    echo "ERROR: Model name required!"
    echo "Usage: sbatch scripts/slurm_run_mamba_experiment.sh <MODEL_NAME>"
    echo "Examples:"
    echo "  sbatch scripts/slurm_run_mamba_experiment.sh state-spaces/mamba-130m-hf"
    echo "  sbatch scripts/slurm_run_mamba_experiment.sh state-spaces/mamba-370m-hf"
    echo "  sbatch scripts/slurm_run_mamba_experiment.sh state-spaces/mamba-1.4b-hf"
    exit 1
fi

MODEL="$1"

# Determine if we need quantization for larger models
if [[ "$MODEL" == *"370m"* ]] || [[ "$MODEL" == *"790m"* ]] || [[ "$MODEL" == *"1.4b"* ]] || [[ "$MODEL" == *"2.8b"* ]]; then
    USE_QUANTIZATION="--reduce_precision"
    echo "Large model detected - enabling 4-bit quantization"
else
    USE_QUANTIZATION=""
    echo "Small model - no quantization needed"
fi

REPO_DIR="/scratch/jhu35/rpandya4/model-human-processing"
STIMULI_DIR="$REPO_DIR/data/stimuli"
OUTPUT_DIR="$REPO_DIR/data/model_output"

# Determine which tasks to run based on array ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Job 0: Run first 4 tasks sequentially
    TASKS=("capitals-recall" "capitals-recognition" "animals" "gender")
    echo "Running tasks sequentially: ${TASKS[@]}"
else
    # Job 1: Run syllogism separately
    TASKS=("syllogism")
    echo "Running task: syllogism (intensive)"
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Tasks: ${TASKS[@]}"
echo "  Stimuli dir: $STIMULI_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo ""

cd $REPO_DIR || exit 1

# Run experiments
overall_exit_code=0
for TASK in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running experiment: $TASK"
    echo "Started: $(date)"
    echo "=========================================="

    python src/run_experiment.py \
        --model $MODEL \
        --task $TASK \
        --stimuli_dir $STIMULI_DIR \
        --output_dir $OUTPUT_DIR \
        $USE_QUANTIZATION

    exit_code=$?

    echo ""
    echo "=========================================="
    echo "Task finished: $TASK"
    echo "Exit code: $exit_code"
    echo "Finished: $(date)"
    echo "=========================================="

    # Check if output file was created
    SAFE_MODEL_NAME=$(basename $MODEL)
    OUTPUT_FILE="$OUTPUT_DIR/logit_lens/${TASK}_${SAFE_MODEL_NAME}.csv"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Output file created: $OUTPUT_FILE"
        wc -l $OUTPUT_FILE
    else
        echo "✗ Output file not found: $OUTPUT_FILE"
        overall_exit_code=1
    fi
done

echo ""
echo "=========================================="
echo "All tasks completed: $(date)"
echo "Overall exit code: $overall_exit_code"
echo "=========================================="

exit $overall_exit_code
