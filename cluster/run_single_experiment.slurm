#!/bin/bash
#SBATCH --job-name=model-exp
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# Usage: sbatch run_single_experiment.slurm <MODEL> <TASK>
# Example: sbatch run_single_experiment.slurm gpt2 capitals-recall

set -e

# Get arguments
MODEL=${1:-"gpt2"}
TASK=${2:-"capitals-recall"}

echo "=========================================="
echo "Running experiment on DSAI cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "=========================================="

# Load modules
module load anaconda3/2024.02-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mamba_exp

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

# Run the experiment
echo "Starting experiment..."
python src/run_experiment.py \
    --model "$MODEL" \
    --task "$TASK" \
    --stimuli_dir data/stimuli \
    --output_dir data/model_output \
    --cache_dir "$HF_HOME" 

echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
