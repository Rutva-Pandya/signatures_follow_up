#!/bin/bash
#SBATCH --job-name=rwkv_test
#SBATCH --output=logs/rwkv_test_%j.out
#SBATCH --error=logs/rwkv_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=defq

# Test RWKV architecture compatibility before running full experiments
# This verifies that RWKV models work with our framework
#
# Usage:
#   sbatch scripts/slurm_test_rwkv.sh

echo "=========================================="
echo "RWKV Architecture Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

mkdir -p logs

# Load environment
module load anaconda3/2024.02-1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Use same environment as Mamba experiments
conda activate mamba_exp

# Set HuggingFace cache
export HF_HOME="/scratch/jhu35/rpandya4/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/jhu35/rpandya4/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/jhu35/rpandya4/huggingface_cache"
mkdir -p $HF_HOME

echo "Using HuggingFace cache at: $HF_HOME"

REPO_DIR="/scratch/jhu35/rpandya4/model-human-processing"
cd $REPO_DIR || exit 1

echo ""
echo "Running RWKV architecture test..."
echo ""

# Run the test script
python test_rwkv_structure.py

exit_code=$?

echo ""
echo "=========================================="
echo "Test completed: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ RWKV test PASSED!"
    echo ""
    echo "Next steps:"
    echo "  1. Review code changes needed (printed above)"
    echo "  2. Implement changes to src/model.py and src/utils.py"
    echo "  3. Run RWKV experiments:"
    echo "     sbatch scripts/slurm_run_rwkv_experiment.sh RWKV/rwkv-4-169m-pile"
    echo "     sbatch scripts/slurm_run_rwkv_experiment.sh RWKV/rwkv-4-430m-pile"
    echo "     sbatch scripts/slurm_run_rwkv_experiment.sh RWKV/rwkv-4-1b5-pile"
else
    echo ""
    echo "✗ RWKV test FAILED"
    echo "Check the error output above for details"
fi

exit $exit_code
