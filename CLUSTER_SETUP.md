# Cluster Environment Setup

## One-Time Setup (First Time Only)

### 1. Create Conda Environment

```bash
# On the cluster login node
cd /scratch/jhu35/rpandya4/model-human-processing

# Load anaconda module
module load anaconda3/2024.02-1

# Create conda environment
conda create -n model-human-processing python=3.10 -y

# Activate environment
conda activate model-human-processing

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers>=4.35.0
pip install pandas numpy
pip install nnsight>=0.2.0
pip install tuned-lens
pip install bitsandbytes accelerate
```

### 2. Verify Installation

```bash
# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import nnsight; print('nnsight:', nnsight.__version__)"
python -c "print('GPU available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.x.x
Transformers: 4.x.x
nnsight: 0.x.x
GPU available: True
```

### 3. Set Up HuggingFace Cache (Optional but Recommended)

```bash
# Add to your ~/.bashrc for persistence
echo 'export HF_HOME="/scratch/jhu35/rpandya4/huggingface_cache"' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="$HF_HOME/transformers"' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE="$HF_HOME/datasets"' >> ~/.bashrc
source ~/.bashrc

# Create cache directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE
```

## Running Experiments

Once the environment is set up, you can submit jobs:

```bash
# Single task
sbatch cluster/run_single_experiment.slurm "EleutherAI/pythia-70m" "animals"

# All tasks for one model
sbatch cluster/run_all_tasks.slurm "EleutherAI/pythia-70m"

# All Pythia models (40 jobs)
bash cluster/batch_submit_pythia.sh
```

## Troubleshooting

### If conda environment not found:
```bash
conda env list  # Check if environment exists
conda activate model-human-processing
```

### If GPU not available in Python:
```bash
nvidia-smi  # Should show GPUs
module load cuda  # If CUDA module needed
```

### If out of memory:
- Use smaller models first (pythia-70m, pythia-160m)
- Reduce batch size in code if needed
- Request more GPU memory in SLURM script
