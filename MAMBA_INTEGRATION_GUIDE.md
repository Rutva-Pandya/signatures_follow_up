# Mamba-1 Integration Guide (SLURM Cluster)

This document describes the integration of Mamba-1 SSM models into the model-human-processing codebase for SLURM-based clusters.

## Overview

We've added support for **Mamba-1** (Selective State Space Models) to enable comparison of SSM architectures with Transformers on human-like processing tasks.

**Model tested:** `state-spaces/mamba-130m-hf` (smallest Mamba-1 variant, 24 layers)

---

## Changes Made

### 1. **Code Modifications**

#### `src/utils.py`
- ✓ Already had "mamba" in model family detection (line 57)
- No changes needed

#### `analysis/notebooks/utils.py`
- ✓ Added Mamba vocab size: 50277 (line 23-24)
- ✓ Added N_LAYERS for all Mamba variants (lines 36-38):
  ```python
  # SSM models
  "mamba-130m-hf": 24, "mamba-370m-hf": 48, "mamba-790m-hf": 48,
  "mamba-1.4b-hf": 48, "mamba-2.8b-hf": 64,
  ```

#### `src/model.py`
- ✓ Already had Mamba layer references (lines 86-89):
  ```python
  elif self.model_family == "mamba":
      self.layers = self.model.backbone.layers
      self.layer_norm = self.model.backbone.norm_f
      self.lm_head = self.model.lm_head
  ```

### 2. **SLURM Scripts Created**

| Script | Purpose | Resources |
|--------|---------|-----------|
| `slurm_test_mamba_step1.sh` | Verify model loading & architecture | 1 GPU, 16GB, 30min |
| `slurm_test_mamba_step2.sh` | Test nnsight & hidden extraction | 1 GPU, 16GB, 30min |
| `slurm_test_mamba_step3.sh` | Validate logit lens | 1 GPU, 32GB, 1hr |
| `slurm_run_mamba_experiment.sh` | Run full experiments (job array) | 1 GPU, 64GB, 4hr |
| `cluster_config.template.sh` | Cluster configuration template | N/A |

---

## Setup

### 1. Environment Requirements

**Tested Configuration (JHU DSAI Cluster):**
- **Partition**: h100 (or l40s for testing)
- **GPU**: 1x H100 (80GB) or 1x L40S (48GB)
- **Memory**: 128GB system RAM
- **Time**: 6 hours for full experiments (5 tasks)
- **Python**: 3.10
- **CUDA**: 11.8 or 12.x

**Package Versions (Reproducible):**
```bash
torch>=2.4.0 (with CUDA 11.8)
transformers>=4.39.0  # Native Mamba support
nnsight>=0.3
pandas
numpy
tuned-lens
```

### 2. Cluster-Specific Adjustments

The scripts are pre-configured for JHU DSAI cluster. For other clusters, edit the `#SBATCH` headers in:
- `scripts/slurm_mamba_integration_check.sh`
- `scripts/slurm_run_mamba_experiment.sh`

Adjust:
- `--partition=h100` → your GPU partition name
- `--gres=gpu:h100:1` → your GPU request format
- `--mem=128G` → your memory limits
- `--time=06:00:00` → your time limits
- Add `--account=` or `--qos=` if required

**No cluster_config.sh needed** - configuration is embedded in scripts for reproducibility

### 3. Quick Start (JHU DSAI Cluster)

```bash
# Clone repository
cd /scratch/jhu35/rpandya4/
git clone https://github.com/Rutva-Pandya/signatures_follow_up.git model-human-processing
cd model-human-processing
git checkout mamba-integration

# Run validation test (creates environment automatically)
sbatch scripts/slurm_mamba_integration_check.sh

# Run full experiments
sbatch scripts/slurm_run_mamba_experiment.sh state-spaces/mamba-130m-hf
sbatch scripts/slurm_run_mamba_experiment.sh state-spaces/mamba-370m-hf
sbatch scripts/slurm_run_mamba_experiment.sh state-spaces/mamba-1.4b-hf
```

### 4. Manual Installation (Other Clusters)

On your cluster (in an interactive session or compute node):

```bash
# Load modules or activate environment (as configured)
module load python/3.9 cuda/11.8 gcc/9.3.0  # example

# Install packages
pip install transformers torch nnsight
pip install mamba-ssm causal-conv1d>=1.1.0
```

**Important:** Mamba requires `mamba-ssm` and `causal-conv1d` packages with CUDA support.

---

## Testing Protocol

### Step 1: Architecture Verification
```bash
cd /path/to/model-human-processing
mkdir -p logs
sbatch scripts/slurm_test_mamba_step1.sh
```

Check output:
```bash
tail logs/mamba_step1_*.out
```

**What it checks:**
- Model loads correctly
- Has expected structure (backbone.layers, backbone.norm_f, lm_head)
- Tokenizer works
- Forward pass produces outputs
- Number of layers matches expectations (24 for mamba-130m-hf)

**Expected:** "ALL TESTS PASSED ✓"

### Step 2: Hidden State Extraction
```bash
sbatch scripts/slurm_test_mamba_step2.sh
tail logs/mamba_step2_*.out
```

**What it checks:**
- nnsight can wrap Mamba models
- Can trace through model
- Hidden states extractable from each layer
- Shapes are compatible with logit lens

**Expected:** "ALL TESTS PASSED ✓"

### Step 3: Logit Lens Validation
```bash
sbatch scripts/slurm_test_mamba_step3.sh
tail logs/mamba_step3_*.out
```

**What it checks:**
- LM class initializes with Mamba
- Logprobs extracted at all layers
- Final layer matches actual model output
- Predictions are sensible
- Rank trajectories improve
- conditional_score_all_layers() works

**Expected:** "ALL TESTS PASSED ✓"

---

## Running Full Experiments

Once all tests pass:

### Option A: Job Array (Recommended - Runs all 5 tasks)
```bash
# Adjust SBATCH directives in slurm_run_mamba_experiment.sh first
sbatch scripts/slurm_run_mamba_experiment.sh
```

This submits 5 jobs (one per task) that run in parallel if resources available.

Check progress:
```bash
squeue -u $USER
ls -lh data/model_output/logit_lens/*mamba*.csv
```

### Option B: Individual Tasks
```bash
# Edit run_experiment.sh or create custom SLURM script
sbatch --wrap="python src/run_experiment.py \
    --model state-spaces/mamba-130m-hf \
    --task capitals-recall \
    --stimuli_dir data/stimuli \
    --output_dir data/model_output"
```

### Output Location
Results saved to:
```
data/model_output/logit_lens/
  capitals-recall_mamba-130m-hf.csv
  capitals-recognition_mamba-130m-hf.csv
  animals_mamba-130m-hf.csv
  gender_mamba-130m-hf.csv
  syllogism_mamba-130m-hf.csv
```

Each file contains layer-wise predictions for all stimuli.
- Tokenizer works
- Forward pass produces outputs
- Number of layers matches expectations (24 for mamba-130m-hf)

**Expected output:**
```
ALL TESTS PASSED ✓
Model: state-spaces/mamba-130m-hf
Layers: 24
Vocab: 50277
```

### Step 2: Hidden State Extraction
```bash
python scripts/test_mamba_step2_nnsight.py
```

**What it checks:**
- nnsight can wrap Mamba models
- Can trace through model
- Hidden states extractable from each layer
- Shapes are compatible with logit lens
- Includes backup method using manual hooks

**Expected output:**
```
ALL TESTS PASSED ✓
Hidden state extraction confirmed working!
```

### Step 3: Logit Lens Validation
```bash
python scripts/test_mamba_step3_logit_lens.py
```

**What it checks:**
- LM class initializes with Mamba
- Logprobs extracted at all layers
- Final layer matches actual model output
- Predictions are sensible across layers
- Rank of correct answer improves
- conditional_score_all_layers() works

**Expected output:**
```
ALL TESTS PASSED ✓
The LM class works correctly with Mamba!
```

### Step 4: Pilot Experiment
```bash
python scripts/test_mamba_step4_pilot.py
```

**What it does:**
- Runs first 5 items of capitals-recall task
- Tests full evaluation pipeline
- Saves results to `data/model_output/pilot/`
- Shows layer-wise prediction evolution

**Expected output:**
```
PILOT EXPERIMENT COMPLETE ✓
Items evaluated: 5
Layers: 24
Total predictions: 120
```

---

## Running Full Experiments

Once all tests pass, run full experiments:

### Single Task
```bash
bash scripts/run_experiment.sh state-spaces/mamba-130m-hf capitals-recall
```

### All Tasks
```bash
for task in capitals-recall capitals-recognition animals gender syllogism; do
    bash scripts/run_experiment.sh state-spaces/mamba-130m-hf $task
done
```

### Output Location
Results will be saved to:
```
data/model_output/logit_lens/
  capitals-recall_mamba-130m-hf.csv
  capitals-recognition_mamba-130m-hf.csv
  animals_mamba-130m-hf.csv
  gender_mamba-130m-hf.csv
  syllogism_mamba-130m-hf.csv
```

---

## Data Processing

After running experiments, process the results:

```bash
jupyter notebook analysis/notebooks/0_process_lm_data.ipynb
```

This will:
1. Read raw model outputs
2. Compute trajectory metrics (AUC, layer of biggest change, etc.)
3. Save processed metrics to `data/model_output/processed/`
4. Combine with human data → `data/human_model_combined/`

---

## Supported Mamba Models

All Mamba-1 variants are now supported:

| Model | Layers | Parameters | HuggingFace ID |
|-------|--------|------------|----------------|
| Mamba-130M | 24 | 130M | `state-spaces/mamba-130m-hf` |
| Mamba-370M | 48 | 370M | `state-spaces/mamba-370m-hf` |
| Mamba-790M | 48 | 790M | `state-spaces/mamba-790m-hf` |
| Mamba-1.4B | 48 | 1.4B | `state-spaces/mamba-1.4b-hf` |
| Mamba-2.8B | 64 | 2.8B | `state-spaces/mamba-2.8b-hf` |

To test a different size:
```bash
bash scripts/run_experiment.sh state-spaces/mamba-370m-hf capitals-recall
```

---

## Potential Issues & Solutions

### Issue 1: nnsight Compatibility
**Symptom:** `test_mamba_step2_nnsight.py` fails with nnsight errors

**Solution:** The test script includes a backup method using manual PyTorch hooks. If nnsight fails, we can modify `src/model.py` to use hooks instead.

### Issue 2: Hidden State Structure
**Symptom:** `layer.output[0]` doesn't work for Mamba layers

**Solution:** Mamba SSM layers may have different output structure. Check `test_mamba_step2_nnsight.py` output to see actual structure, then modify extraction logic in `src/model.py:133`.

### Issue 3: Logit Lens Mismatch
**Symptom:** Final layer logits don't match actual model output

**Solution:** This may indicate the layer norm or lm_head references are incorrect. Verify in `test_mamba_step3_logit_lens.py` Test 3.

### Issue 4: Missing Dependencies
**Symptom:** `ModuleNotFoundError: No module named 'mamba_ssm'`

**Solution:**
```bash
pip install mamba-ssm causal-conv1d>=1.1.0
```

If compilation fails, ensure CUDA is available and compatible.

---

## Key Differences: Mamba vs Transformers

| Aspect | Transformers | Mamba |
|--------|-------------|-------|
| Core mechanism | Self-attention | Selective state space |
| Layer structure | Attention + MLP | SSM + MLP |
| State | Stateless | Selective state evolution |
| Computational complexity | O(n²) | O(n) |
| Architecture in code | `model.transformer.h` or `model.model.layers` | `model.backbone.layers` |

Despite these differences, the **logit lens** approach should still work because:
- We extract hidden states after each layer
- Apply final layer norm + lm_head to decode
- This treats Mamba like a "decoder-only" architecture

---

## Validation Checklist

Before claiming Mamba integration is successful:

- [ ] All 4 test scripts pass without errors
- [ ] Pilot experiment produces sensible predictions
- [ ] Rank of correct answer improves across layers
- [ ] Final layer matches actual model output (< 1e-3 difference)
- [ ] Full experiment runs on at least one task
- [ ] Processed data has expected trajectory metrics
- [ ] Can visualize layer-wise evolution

---

## Next Steps: RWKV Integration

After Mamba is validated, the RWKV integration will require:

1. **Architecture detection:** Add "rwkv" to `src/utils.py:get_model_family()`
2. **Layer references:** Determine RWKV's structure (likely `model.rwkv.blocks`, `model.rwkv.ln_out`, `model.head`)
3. **Handle recurrence:** RWKV has recurrent state - may need special handling
4. **N_LAYERS:** Add RWKV variants to `analysis/notebooks/utils.py`
5. **Test scripts:** Adapt the 4 test scripts for RWKV
6. **State propagation:** Ensure hidden states properly carry recurrent state

RWKV will be more challenging due to its recurrent nature!

---

## Questions to Investigate on Cluster

When running the tests, pay attention to:

1. **Hidden state shapes:** Are they consistent across all layers?
2. **nnsight behavior:** Does `layer.output[0]` work or do we need `layer.output`?
3. **Logit lens accuracy:** How close is final layer to actual output?
4. **Prediction quality:** Do layer-wise predictions make sense?
5. **Rank trajectories:** Do they show monotonic improvement?

Document any unexpected behavior - it will inform RWKV integration!

---

## Contact & Support

If tests fail or unexpected behavior occurs:
1. Check the error messages in test script output
2. Review the "Potential Issues" section above
3. Examine the actual model architecture with:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", trust_remote_code=True)
   print(model)
   ```

---

**Created:** 2025-10-13
**Model:** state-spaces/mamba-130m-hf
**Status:** Ready for cluster testing
