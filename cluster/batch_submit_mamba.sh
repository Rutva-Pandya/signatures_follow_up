#!/bin/bash
# Batch submission script for Mamba models
# Submit Mamba1 and Mamba2 models at comparable sizes

# List of Mamba models to evaluate
MODELS=(
    # Mamba 1 (original architecture)
    "state-spaces/mamba-130m-hf"
    "state-spaces/mamba-790m-hf"
    "state-spaces/mamba-1.4b-hf"
    # Mamba 2 (newer architecture - 2-8x faster)
    "state-spaces/mamba2-130m-hf"
    "state-spaces/mamba2-780m-hf"
    "state-spaces/mamba2-1.3b-hf"
)

echo "=========================================="
echo "Submitting batch jobs for Mamba models"
echo "=========================================="

# Submit job for each model
for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for: $MODEL"
    sbatch cluster/run_all_tasks.slurm "$MODEL"
    sleep 1  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "=========================================="
echo "All Mamba jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  logs/all_tasks_*.out"
