#!/bin/bash
#SBATCH --job-name=r-bayes-factors
#SBATCH --output=logs/r_bayes_factors_%j.out
#SBATCH --error=logs/r_bayes_factors_%j.err
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Run R Bayes Factor scripts for all tasks
# No GPU needed - CPU only

set -e

echo "=========================================="
echo "Running R Bayes Factor Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Load R module
module load r/4.3.0

# Set up paths
WORK_DIR="/scratch/jhu35/rpandya4/model-human-processing"
cd $WORK_DIR

# Ensure output directories exist
mkdir -p analysis/r_models/baseline_final analysis/r_models/baseline_midpoint analysis/r_outputs

# Run all R scripts for the 4 tasks
TASKS=("animals" "capitals-recall" "capitals-recognition" "syllogism")

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running Bayes Factor analysis for: $TASK"
    echo "=========================================="

    Rscript analysis/r_scripts/fit_${TASK}.R

    echo "Task $TASK complete!"
done

echo ""
echo "=========================================="
echo "All Bayes Factor analyses complete!"
echo "=========================================="
