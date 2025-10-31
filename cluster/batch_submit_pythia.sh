#!/bin/bash
# Batch submission script for Pythia models (Q13 - architecture vs training data)
# Submit ALL Pythia models (Transformer + Pile) for all tasks

MODELS=(
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-410m"
    "EleutherAI/pythia-1b"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "EleutherAI/pythia-6.9b"
    "EleutherAI/pythia-12b"
)

echo "=========================================="
echo "Submitting Pythia models for evaluation"
echo "Testing: Transformer + Pile training"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for: $MODEL"
    sbatch cluster/run_all_tasks.slurm "$MODEL"
    sleep 1
done

echo ""
echo "=========================================="
echo "All Pythia jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  logs/all_tasks_*.out"
