#!/bin/bash
# Batch submission script to reproduce paper results
# Submit multiple models for all tasks

# List of models to evaluate (from paper)
MODELS=(
    "gpt2"
    "gpt2-medium"
    "gpt2-large"
    "gpt2-xl"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "facebook/opt-125m"
    "facebook/opt-1.3b"
    "facebook/opt-6.7b"
)

echo "=========================================="
echo "Submitting batch jobs for model evaluation"
echo "=========================================="

# Submit job for each model
for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for: $MODEL"
    sbatch cluster/run_all_tasks.slurm "$MODEL"
    sleep 1  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  logs/all_tasks_*.out"
