#!/bin/bash
#
# Cluster Configuration Template
# ==============================
# Copy this file to cluster_config.sh and customize for your cluster
# Then source it before running SLURM scripts:
#   source scripts/cluster_config.sh
#
# DO NOT commit cluster_config.sh to git (it's in .gitignore)

# =============================================================================
# CLUSTER-SPECIFIC SETTINGS - CUSTOMIZE THESE
# =============================================================================

# 1. MODULE LOADS (if your cluster uses module system)
# ----------------------------------------------------
# Uncomment and adjust for your cluster's modules:

# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0
# module load cudnn/8.2


# 2. PYTHON ENVIRONMENT
# ---------------------
# Choose ONE of the following and uncomment:

# Option A: Conda environment
# source ~/.bashrc  # or ~/.bash_profile
# conda activate your_env_name

# Option B: Virtualenv
# source /path/to/your/venv/bin/activate

# Option C: Module-based Python (no virtual env needed)
# (just ensure module load above is correct)


# 3. SLURM PARTITION NAMES
# -------------------------
# What's your cluster's GPU partition name?
export GPU_PARTITION="gpu"  # Common names: gpu, gpu_p100, gpu_a100, etc.

# Do you have different partitions for different GPU types?
export GPU_PARTITION_A100="gpu_a100"  # Optional
export GPU_PARTITION_V100="gpu_v100"  # Optional


# 4. GPU REQUEST FORMAT
# ----------------------
# How does your cluster specify GPU requests?
# Common formats:
#   - gpu:1 (generic)
#   - gpu:a100:1 (specific type)
#   - gpu:tesla:1
export GPU_REQUEST="gpu:1"


# 5. DEFAULT RESOURCE LIMITS
# ---------------------------
export DEFAULT_TIME="04:00:00"    # Default job time limit
export DEFAULT_MEM="32G"          # Default memory
export DEFAULT_CPUS="4"           # Default CPU cores


# 6. PATHS
# --------
# Where is the repository on your cluster?
export REPO_DIR="$HOME/model-human-processing"

# Where should HuggingFace cache models? (optional, defaults to ~/.cache/huggingface)
# export HF_HOME="/scratch/$USER/huggingface_cache"
# export TRANSFORMERS_CACHE="$HF_HOME"


# 7. ADDITIONAL ENVIRONMENT VARIABLES
# ------------------------------------
# Any other environment variables needed?

# Suppress HuggingFace warnings (optional)
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Set offline mode if cluster has no internet (must pre-download models)
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1


# =============================================================================
# VERIFICATION (don't change this part)
# =============================================================================

echo "Cluster configuration loaded:"
echo "  GPU partition: $GPU_PARTITION"
echo "  GPU request format: $GPU_REQUEST"
echo "  Repository: $REPO_DIR"
echo "  Python: $(which python 2>/dev/null || echo 'not found')"

if command -v nvidia-smi &> /dev/null; then
    echo "  GPU available: yes"
else
    echo "  GPU available: not detected (may not be on GPU node yet)"
fi
