#!/bin/bash
#SBATCH --job-name=rwkv_experiment
#SBATCH --output=logs/rwkv_exp_%A_%a.out
#SBATCH --error=logs/rwkv_exp_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --array=0-1

# This script runs RWKV experiments on all tasks using job arrays
# Array job 0: runs first 4 tasks sequentially (capitals-recall, capitals-recognition, animals, gender)
# Array job 1: runs syllogism task separately (it's much more intensive, uses quantization)
#
# Usage:
#   sbatch scripts/slurm_run_rwkv_experiment.sh <MODEL_NAME>
#
# Supported models:
#   RWKV/rwkv-4-169m-pile   (auto-quantizes for syllogism task only)
#   RWKV/rwkv-4-430m-pile   (auto-quantizes for all tasks)
#   RWKV/rwkv-4-1b5-pile    (auto-quantizes for all tasks)
#
# Quantization strategy:
#   - 169m: No quantization for first 4 tasks, 4-bit for syllogism
#   - ≥430m: 4-bit quantization for all tasks

echo "=========================================="
echo "RWKV Experiment Runner"
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

# Use same environment as Mamba (already has all dependencies)
conda activate mamba_exp

# Configuration
# Model must be passed as first argument
if [ -z "$1" ]; then
    echo "ERROR: Model name required!"
    echo "Usage: sbatch scripts/slurm_run_rwkv_experiment.sh <MODEL_NAME>"
    echo "Examples:"
    echo "  sbatch scripts/slurm_run_rwkv_experiment.sh RWKV/rwkv-4-169m-pile"
    echo "  sbatch scripts/slurm_run_rwkv_experiment.sh RWKV/rwkv-4-430m-pile"
    echo "  sbatch scripts/slurm_run_rwkv_experiment.sh RWKV/rwkv-4-1b5-pile"
    exit 1
fi

MODEL="$1"

# Set HuggingFace cache to scratch space to avoid filling home directory
export HF_HOME="/scratch/jhu35/rpandya4/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/jhu35/rpandya4/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/jhu35/rpandya4/huggingface_cache"
mkdir -p $HF_HOME

echo "Using HuggingFace cache at: $HF_HOME"

# Determine if we need quantization
# - Large models (>=430m): always use quantization
# - Small models (169m): only use quantization for syllogism task
if [[ "$MODEL" == *"430m"* ]] || [[ "$MODEL" == *"1b5"* ]] || [[ "$MODEL" == *"3b"* ]]; then
    USE_QUANTIZATION="--reduce_precision"
    echo "Large model detected - enabling 4-bit quantization for all tasks"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # Syllogism task OOMs even on small models
    USE_QUANTIZATION="--reduce_precision"
    echo "Syllogism task detected - enabling 4-bit quantization"
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
