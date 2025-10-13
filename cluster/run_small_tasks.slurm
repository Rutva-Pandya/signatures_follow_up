#!/bin/bash
#SBATCH --job-name=model-small-tasks
#SBATCH --output=logs/small_tasks_%j.out
#SBATCH --error=logs/small_tasks_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Usage: sbatch run_small_tasks.slurm <MODEL>
# For models that OOM on syllogism task (RWKV, Mamba)
# Runs only the 4 smaller tasks

set -e

# Get model argument
MODEL=${1:-"gpt2"}

echo "=========================================="
echo "Running SMALL tasks for model: $MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Load modules
module load anaconda3/2024.02-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate model-human-processing

# Set up paths
WORK_DIR="/scratch/jhu35/rpandya4/model-human-processing"
cd $WORK_DIR

# Set HuggingFace cache directory
export HF_HOME="/scratch/jhu35/rpandya4/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# Print GPU info
echo "GPU Information:"
nvidia-smi

# Run ONLY the small tasks (skip syllogism)
TASKS=("capitals-recall" "capitals-recognition" "animals" "gender")

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running task: $TASK"
    echo "=========================================="

    python src/run_experiment.py \
        --model "$MODEL" \
        --task "$TASK" \
        --stimuli_dir data/stimuli \
        --output_dir data/model_output \
        --cache_dir "$HF_HOME"

    echo "Task $TASK complete!"
done

echo ""
echo "=========================================="
echo "All small tasks complete!"
echo "Skipped syllogism task due to memory constraints"
echo "=========================================="
