# RWKV Integration: Quick Summary

## What is RWKV?

**RWKV** (Receptance Weighted Key Value) is a **linear attention RNN** - another recurrent alternative to transformers, like Mamba.

| Feature | Transformers | Mamba (SSM) | RWKV (Linear Attn) |
|---------|-------------|-------------|-------------------|
| Complexity | O(n²) | O(n) | O(n) |
| Recurrent | No | Yes | Yes |
| Mechanism | Self-attention | State spaces | Linear attention |

## Why Integrate RWKV?

**Scientific Question**: Is inverse scaling a property of:
- **Just Mamba** (state space models)?
- **All recurrent architectures** (Mamba + RWKV)?

If RWKV also shows inverse scaling → general phenomenon of recurrent models
If RWKV doesn't → Mamba-specific behavior

## Models Available

**RWKV-4 Series** (Pile-trained, recommended):
```
RWKV/rwkv-4-169m-pile   (24 layers)
RWKV/rwkv-4-430m-pile   (24 layers)
RWKV/rwkv-4-1b5-pile    (24 layers)
```

**Comparison with Mamba**:
```
Mamba: 130M, 370M, 790M, 1.4B
RWKV:  169M, 430M, 1.5B
```

Perfect alignment for comparison!

## Code Changes Required

**Total: 5 locations, ~30 lines of code**

### 1. `src/utils.py` (line 57) - Add RWKV to family detection
```python
families = ["llama", "olmo", "gemma", "gpt", "falcon", "mamba", "rwkv"]
```

### 2. `src/model.py` (line 53) - Enable trust_remote_code
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,  # ADD THIS
    **load_kwargs
)
```

### 3. `src/model.py` (lines 89-93) - Add RWKV layer references
```python
elif self.model_family == "rwkv":
    self.layers = self.model.rwkv.blocks
    self.layer_norm = self.model.rwkv.ln_out
    self.lm_head = self.model.head
```

### 4. `src/model.py` (lines 173-180) - Hidden state extraction (may not need changes)
Current code should work, but may need to test with nnsight

### 5. `analysis/notebooks/utils.py` - Add RWKV to model lists
- Add to `MODELS`
- Add to `N_LAYERS` (24 for all three)
- Add to `VOCAB_SIZE` (50277, same as GPT-2)
- Add to `MODEL_FAMILY_MAP`
- Add to `MODEL_PAL` (color palette)

## Testing Strategy

**BEFORE any code changes**:
```bash
python scripts/test_rwkv_structure.py
```

This will:
1. Load RWKV-169M model
2. Verify layer names are correct
3. Test hidden state extraction
4. Test tokenizer compatibility
5. Tell you exactly what code changes to make

**Estimated time**: 5 minutes

## Implementation Timeline

| Phase | Task | Time | Blocker? |
|-------|------|------|----------|
| 0 | **Test architecture** | 5 min | **Run this first** |
| 1 | Update code (5 locations) | 15 min | Test must pass |
| 2 | Run inference (3 models × 4 tasks) | 2-3 hours | Compute time |
| 3 | Process data | Auto | Should work as-is |
| 4 | Run Experiment 2 analysis | 10 min | - |
| 5 | Create comparison plots | 20 min | - |
| 6 | Run validation | 5 min | - |

**Total active time**: ~1 hour
**Total compute time**: 2-3 hours

## Risk Assessment

**Low Risk** ✓
- Very similar to Mamba integration (proven pattern)
- Minimal code changes
- Validation scripts ready
- Test script verifies compatibility first

**Potential Issues**:
- Hidden state format might differ (test will catch this)
- May need `trust_remote_code=True` (easy fix)
- nnsight might handle RWKV differently (can test)

## Expected Results

If RWKV shows inverse scaling (like Mamba):
```
RWKV-169M:  CoM ~15-20 (high instability)
RWKV-430M:  CoM ~15-20 (still unstable)
RWKV-1.5B:  CoM ~5-10  (dramatic stabilization)
```

If RWKV scales normally (like Transformers):
```
RWKV-169M:  CoM ~3-5
RWKV-430M:  CoM ~3-5
RWKV-1.5B:  CoM ~3-5
```

## Next Steps

1. ✅ **Wait for Mamba 790M results** (in progress)
2. Create new branch: `git checkout -b rwkv-integration`
3. Run architecture test: `python scripts/test_rwkv_structure.py`
4. If test passes → implement code changes
5. Run inference on RWKV models
6. Analyze and compare results

## Scientific Value

**If both Mamba and RWKV show inverse scaling**:
- Inverse scaling is a general property of recurrent architectures
- Suggests fundamental limitation of small-scale recurrent models
- Important for understanding efficient model design

**If only Mamba shows inverse scaling**:
- Behavior is specific to state space models
- Linear attention (RWKV) may be more stable at small scale
- Could inform architecture choice for efficient models

**Either way**: Strong contribution to understanding scaling laws across architectures

## Files Created

1. **`analysis/RWKV_INTEGRATION_ANALYSIS.md`** - Full technical details
2. **`scripts/test_rwkv_structure.py`** - Architecture verification test
3. **`analysis/RWKV_INTEGRATION_SUMMARY.md`** - This document

All ready for when Mamba 790M finishes!
