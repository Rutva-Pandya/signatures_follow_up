# RWKV Integration: Technical Analysis

**Status**: Planning Phase
**Date**: October 15, 2025

---

## Executive Summary

This document analyzes the technical requirements for integrating RWKV (Receptance Weighted Key Value) models into the dual-processing analysis framework. RWKV is a linear attention RNN architecture that serves as an alternative to transformers, similar to Mamba.

---

## What is RWKV?

**Architecture Type**: Linear Attention RNN
**Key Feature**: Time-mixing and channel-mixing blocks instead of attention
**Computational Complexity**: O(n) instead of O(n²) for transformers
**Recurrent**: Yes - maintains hidden state across tokens

**Relation to Mamba**:
- Both: Recurrent alternatives to transformers
- Both: Linear complexity vs quadratic
- Different: RWKV uses linear attention; Mamba uses state space models

---

## Available Models on HuggingFace

### RWKV-4 Series (Pile-trained)
```
RWKV/rwkv-4-169m-pile     (24 layers, hidden_size 768)
RWKV/rwkv-4-430m-pile     (24 layers, hidden_size 1024)
RWKV/rwkv-4-1b5-pile      (24 layers, hidden_size 2048)
RWKV/rwkv-4-3b-pile       (32 layers, hidden_size 2560)
RWKV/rwkv-4-7b-pile       (32 layers, hidden_size 4096)
RWKV/rwkv-4-14b-pile      (40 layers, hidden_size 5120)
```

### RWKV-5 Series (World-trained, multilingual)
```
RWKV/rwkv-5-world-169m    (26 layers)
RWKV/rwkv-5-world-1b5     (32 layers)
RWKV/rwkv-5-world-3b      (32 layers)
```

### RWKV-6 Series (Latest, v6.0)
```
RWKV/v6-Finch-1B6-HF      (24 layers)
RWKV/v6-Finch-3B-HF       (32 layers)
RWKV/v6-Finch-7B-HF       (32 layers)
```

**Recommended for Testing**:
- RWKV-4 series (Pile-trained) - most stable, best documented
- 3 models to match Mamba: 169M, 430M, 1.5B

---

## Architecture Structure Investigation

### Critical Question: What are the internal layer names?

Based on HuggingFace transformers library structure, RWKV models typically have:

```python
RwkvForCausalLM(
  (rwkv): RwkvModel(
    (embeddings): Embedding(...)
    (blocks): ModuleList(
      (0-N): RwkvBlock(
        (attention): RwkvSelfAttention(...)
        (feed_forward): RwkvFeedForward(...)
        (pre_ln): LayerNorm(...)
        (ln1): LayerNorm(...)
        (ln2): LayerNorm(...)
      )
    )
    (ln_out): LayerNorm(...)
  )
  (head): Linear(...)
)
```

**Expected Layer References**:
- Layers: `model.rwkv.blocks`
- Final layer norm: `model.rwkv.ln_out`
- LM head: `model.head`

---

## Code Changes Required

### 1. `src/utils.py` (Line 57)

**Current**:
```python
def get_model_family(model_name):
    model_name = model_name.lower()
    families = ["llama", "olmo", "gemma", "gpt", "falcon", "mamba"]
    for family in families:
        if family in model_name:
            return family
    raise ValueError(f"Unrecognized model family for {model_name}")
```

**Change**:
```python
def get_model_family(model_name):
    model_name = model_name.lower()
    families = ["llama", "olmo", "gemma", "gpt", "falcon", "mamba", "rwkv"]
    for family in families:
        if family in model_name:
            return family
    raise ValueError(f"Unrecognized model family for {model_name}")
```

### 2. `src/model.py` (Lines 81-94)

**Current**:
```python
# Define references to internal layers and functions for logit lens.
if self.model_family in ["llama", "olmo", "gemma", "falcon"]:
    self.layers = self.model.model.layers
    self.layer_norm = self.model.model.norm
    self.lm_head = self.model.lm_head
elif self.model_family == "gpt":
    self.layers = self.model.transformer.h
    self.layer_norm = self.model.transformer.ln_f
    self.lm_head = self.model.lm_head
elif self.model_family == "mamba":
    self.layers = self.model.backbone.layers
    self.layer_norm = self.model.backbone.norm_f
    self.lm_head = self.model.lm_head
else:
    raise ValueError(f"Unsupported model family: {self.model_family}")
```

**Change** (add after Mamba):
```python
elif self.model_family == "rwkv":
    self.layers = self.model.rwkv.blocks
    self.layer_norm = self.model.rwkv.ln_out
    self.lm_head = self.model.head
```

### 3. `src/model.py` (Lines 173-180) - Hidden State Extraction

**Current**:
```python
# Get hidden representations.
# For Mamba, layer.output is a tuple where [0] is the hidden state
if self.model_family == "mamba":
    hiddens_l = [
        layer.output[0, :, :].unsqueeze(1) for layer in self.layers
    ]
else:
    hiddens_l = [
        layer.output[0][0, :].unsqueeze(1) for layer in self.layers
    ]
```

**Potential Change** (TBD - need to test):
```python
if self.model_family == "mamba":
    hiddens_l = [
        layer.output[0, :, :].unsqueeze(1) for layer in self.layers
    ]
elif self.model_family == "rwkv":
    # RWKV blocks output hidden states directly (need to verify)
    hiddens_l = [
        layer.output[0, :].unsqueeze(1) for layer in self.layers
    ]
else:
    hiddens_l = [
        layer.output[0][0, :].unsqueeze(1) for layer in self.layers
    ]
```

**⚠️ Critical**: Need to test RWKV output format - may differ from transformers/Mamba

### 4. `analysis/notebooks/utils.py` - Model Configuration

Need to add RWKV to:

**a) Model Lists**:
```python
MODELS = [
    # ... existing models ...
    "rwkv-4-169m-pile",
    "rwkv-4-430m-pile",
    "rwkv-4-1b5-pile",
]
```

**b) Layer Counts**:
```python
N_LAYERS = {
    # ... existing ...
    "rwkv-4-169m-pile": 24,
    "rwkv-4-430m-pile": 24,
    "rwkv-4-1b5-pile": 24,
}
```

**c) Vocab Sizes** (need to verify):
```python
VOCAB_SIZE = {
    # ... existing ...
    "rwkv-4-169m-pile": 50277,  # Same as GPT-2/Mamba
    "rwkv-4-430m-pile": 50277,
    "rwkv-4-1b5-pile": 50277,
}
```

**d) Model Family Map**:
```python
MODEL_FAMILY_MAP = {
    # ... existing ...
    "rwkv-4-169m-pile": "RWKV",
    "rwkv-4-430m-pile": "RWKV",
    "rwkv-4-1b5-pile": "RWKV",
}
```

**e) Color Palette**:
```python
# Add RWKV colors (suggest: shades of green/teal to distinguish from Mamba grey)
MODEL_PAL = {
    # ... existing ...
    "rwkv-4-169m-pile": "#2ECC71",   # Green
    "rwkv-4-430m-pile": "#27AE60",   # Dark green
    "rwkv-4-1b5-pile": "#16A085",    # Teal
}
```

---

## Potential Issues & Unknowns

### 1. **Hidden State Output Format** ⚠️ CRITICAL
**Issue**: Need to verify how RWKV blocks output hidden states
**Impact**: May break if output format differs from standard transformers
**Solution**: Test with a small script first

**Test Script**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

print(f"Number of layers: {len(hidden_states)}")
print(f"Hidden state shape: {hidden_states[0].shape}")
print(f"Output format: {type(hidden_states[0])}")
```

### 2. **Tokenizer Compatibility**
**Issue**: RWKV might use different tokenizer than GPT-2/Mamba
**Impact**: Text might tokenize differently, affecting results
**Solution**: Check vocab size and test tokenization

### 3. **Trust Remote Code**
**Issue**: RWKV models require `trust_remote_code=True`
**Impact**: Need to add this flag to model loading
**Solution**: Add to `AutoModelForCausalLM.from_pretrained()` call

**Location**: `src/model.py` line 53
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,  # Add this
    **load_kwargs
)
```

### 4. **Memory Requirements**
**Issue**: RWKV memory usage might differ from Mamba
**Impact**: May need different quantization strategy
**Solution**: Test on small model first, add quantization if needed

### 5. **Layer Norm Behavior**
**Issue**: RWKV has multiple layer norms (pre_ln, ln1, ln2, ln_out)
**Impact**: Using wrong norm could affect logit lens results
**Solution**: Verify `ln_out` is correct for final decoding

---

## Implementation Plan

### Phase 1: Architecture Verification (30 min)
1. Write test script to load RWKV-169M
2. Inspect model structure (`print(model)`)
3. Verify layer names match expectations
4. Test hidden state extraction format
5. Check tokenizer compatibility

### Phase 2: Code Integration (30 min)
1. Update `src/utils.py`: Add "rwkv" to families
2. Update `src/model.py`: Add RWKV layer references
3. Update `src/model.py`: Add `trust_remote_code=True`
4. Update `src/model.py`: Add hidden state extraction logic (if needed)
5. Update `analysis/notebooks/utils.py`: Add RWKV models to all lists

### Phase 3: Inference (2-3 hours)
1. Create SLURM script (copy from Mamba, adjust memory if needed)
2. Run inference on 3 models: 169M, 430M, 1.5B
3. Run on same 4 tasks: capitals-recall, capitals-recognition, animals, syllogism
4. Verify output files are created correctly

### Phase 4: Processing & Analysis (1 hour)
1. Run processing pipeline (should work automatically)
2. Run Experiment 2 R² analysis
3. Create comparison plots (RWKV vs Mamba vs Transformers)
4. Run validation scripts

### Phase 5: Validation (30 min)
1. Run `validate_trajectory_computation.py` on RWKV data
2. Verify CoM/TTD calculations are correct (should be 0.000000 error)
3. Check summary statistics make sense

**Total Estimated Time**: ~4-5 hours (mostly compute time)

---

## Testing Strategy

### Minimal Test Before Full Integration

Create `test_rwkv_structure.py`:
```python
#!/usr/bin/env python3
"""Test RWKV model structure before full integration."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("="*80)
print("RWKV ARCHITECTURE TEST")
print("="*80)

model_name = "RWKV/rwkv-4-169m-pile"
print(f"\nLoading: {model_name}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"\n✓ Model loaded successfully")
    print(f"  Model class: {model.__class__.__name__}")

    # Check structure
    print(f"\n  Layer references:")
    print(f"    - Blocks: {hasattr(model, 'rwkv')} → model.rwkv.blocks")
    print(f"    - Ln_out: {hasattr(model.rwkv, 'ln_out')} → model.rwkv.ln_out")
    print(f"    - Head: {hasattr(model, 'head')} → model.head")

    # Count layers
    n_layers = len(model.rwkv.blocks)
    print(f"    - Number of layers: {n_layers}")

    # Test forward pass
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    print(f"\n  Hidden states:")
    print(f"    - Number of hidden states: {len(outputs.hidden_states)}")
    print(f"    - Shape: {outputs.hidden_states[0].shape}")
    print(f"    - Type: {type(outputs.hidden_states[0])}")

    print(f"\n  Tokenizer:")
    print(f"    - Vocab size: {len(tokenizer)}")
    print(f"    - BOS token: {tokenizer.bos_token}")
    print(f"    - EOS token: {tokenizer.eos_token}")
    print(f"    - PAD token: {tokenizer.pad_token}")

    print(f"\n✓ All checks passed!")
    print(f"\nExpected code changes:")
    print(f"  1. src/utils.py: Add 'rwkv' to families list")
    print(f"  2. src/model.py: Add trust_remote_code=True")
    print(f"  3. src/model.py: Add RWKV layer references:")
    print(f"       self.layers = self.model.rwkv.blocks")
    print(f"       self.layer_norm = self.model.rwkv.ln_out")
    print(f"       self.lm_head = self.model.head")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
```

---

## Scientific Questions

If RWKV integration succeeds, we can answer:

1. **Does RWKV show inverse scaling like Mamba?**
   - If yes → general property of recurrent architectures
   - If no → specific to state space models

2. **How does RWKV processing differ from Mamba and transformers?**
   - U-shaped trajectories?
   - Different TTD patterns?

3. **Which architecture is most "human-like"?**
   - R² comparison across Transformers, Mamba, RWKV

4. **Is there a universal pattern for efficient architectures?**
   - Both Mamba and RWKV are O(n) complexity
   - Do they share processing characteristics?

---

## Risk Assessment

**Low Risk**:
- Code changes are minimal (5 locations)
- Following exact pattern from Mamba integration
- Validation scripts are ready

**Medium Risk**:
- Hidden state format might differ (can test easily)
- May need `trust_remote_code=True` (easy fix)

**High Risk**:
- RWKV might be fundamentally incompatible with logit lens approach (unlikely)
- Models might not exist or be broken on HuggingFace (can verify quickly)

**Overall Risk**: **Low-Medium** ✓

---

## Next Steps

**Before starting**:
1. ✅ Wait for Mamba 790M results
2. Create new git branch: `rwkv-integration`
3. Run `test_rwkv_structure.py` to verify architecture

**If test passes**:
4. Implement code changes
5. Run inference on RWKV-4 models (169M, 430M, 1.5B)
6. Process data and analyze
7. Compare RWKV vs Mamba vs Transformers

**If test fails**:
8. Document incompatibility
9. Assess if RWKV integration is feasible
10. Consider alternative approaches
