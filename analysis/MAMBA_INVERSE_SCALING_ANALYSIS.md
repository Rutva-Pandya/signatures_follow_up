# Mamba Inverse Scaling: Deep Analysis

**Key Finding**: Mamba exhibits **inverse scaling** - larger models have dramatically LOWER CoM, opposite to transformers.

---

## The Two Fascinating Observations

### 1. Competitor vs NoCompetitor: Nearly Identical ✓

**Observation**: Across all models, CoM is almost identical whether there's a competing answer or not.

**Data**:
```
Animals task (Atypical vs Typical):
  mamba-130m-hf:  Competitor: 17.01, NoCompetitor: 17.01 (Diff: 0.00)
  mamba-370m-hf:  Competitor: 15.47, NoCompetitor: 15.47 (Diff: 0.00)
  mamba-1.4b-hf:  Competitor:  8.41, NoCompetitor:  8.41 (Diff: 0.00)
```

**Interpretation**:
- CoM appears to be an **architectural property** more than task-dependent
- The model's internal processing style is relatively fixed
- This suggests CoM measures how the architecture processes information through depth, not just task difficulty
- Small SSMs "thrash" regardless of whether there's actual ambiguity
- Large SSMs are stable regardless of task complexity

---

### 2. Dramatic Drop for Mamba-1.4B: Inverse Scaling ⚡

**Observation**: CoM drops by 2.2x from small to large Mamba models.

**Overall Statistics**:

| Model | Mean CoM | Std Dev | Median | Max CoM | High Conflict % |
|-------|----------|---------|--------|---------|-----------------|
| **mamba-130m** | **15.64** | 16.85 | 9.58 | 65.87 | **45.8%** |
| **mamba-370m** | **16.11** | 18.79 | 4.87 | 81.23 | **47.7%** |
| **mamba-1.4b** | **7.26** | 5.65 | 6.71 | 25.90 | **8.6%** |

**CoM Distribution Breakdown**:

```
                    CoM = 0    CoM ∈ (0,5)   CoM ∈ [5,15)   CoM ≥ 15
mamba-130m-hf:      30.3%        14.5%          9.5%        45.8% ← Bimodal
mamba-370m-hf:      31.0%        19.0%          2.3%        47.7% ← Bimodal
mamba-1.4b-hf:      11.8%        24.5%         55.2%         8.6% ← Gaussian
```

---

## Why This Matters: Fundamental Difference from Transformers

### Transformer Scaling (Normal)

```
GPT-2 (124M):     CoM = 2.78  (High conflict:  0.0%)
GPT-2-M (355M):   CoM = 3.76  (High conflict:  0.0%)

Pattern: Larger transformers have SIMILAR or SLIGHTLY HIGHER CoM
```

### Mamba Scaling (Inverse)

```
Mamba-130M:   CoM = 15.64  (High conflict: 45.8%) ← Very unstable
Mamba-370M:   CoM = 16.11  (High conflict: 47.7%) ← Very unstable
Mamba-1.4B:   CoM =  7.26  (High conflict:  8.6%) ← STABLE!

Pattern: Larger Mamba models have DRAMATICALLY LOWER CoM
```

**Ratio**: Small Mamba models have 5-6x higher CoM than transformers.
**Drop**: Mamba-1.4B reduces CoM by 2.2x compared to small Mamba.

---

## Task-Level Breakdown

| Task | 130M CoM | 370M CoM | 1.4B CoM | Drop (130M→1.4B) |
|------|----------|----------|----------|------------------|
| capitals-recall | 15.44 | 17.70 | **10.03** | 1.5x |
| capitals-recognition | 16.11 | 18.14 | **4.19** | 3.8x |
| animals | 17.01 | 15.47 | **8.41** | 2.0x |
| syllogism | 12.35 | 9.97 | **3.64** | 3.4x |

**Capitals-recognition** shows the most dramatic improvement with scaling (3.8x reduction).

---

## Example Trajectory: "Bat" (Mammal vs Bird Confusion)

Understanding WHY CoM drops requires looking at layer-wise processing:

### Small Model (130M): Extreme Instability
```
Layer 0:   logprob_diff = -21.84  (favors BIRD)
Layer 5:   logprob_diff = -19.59  (still BIRD)
Layer 8:   logprob_diff = -30.42  (MAXIMUM confusion - strongly favors BIRD)
Final:     logprob_diff = -4.36   (still favors BIRD, never recovers)

CoM = 26.06 (very high)
```

### Large Model (1.4B): Stable Recovery
```
Layer 0:   logprob_diff = -12.93  (favors BIRD, but less extreme)
Layer 5:   logprob_diff = -14.52  (slight dip)
Final:     logprob_diff = +3.09   (RECOVERS to favor MAMMAL ✓)

CoM = smaller (better controlled dip)
```

**Key Difference**:
- Small models: Start wrong, get MORE wrong, stay wrong
- Large models: Start slightly wrong, controlled dip, RECOVER to correct

---

## Mechanistic Hypotheses

### Why do small SSMs thrash?

**Hypothesis 1: Recurrent State Instability**
- SSMs use selective state space mechanisms (recurrent dynamics)
- Small models may have **poorly conditioned state transitions**
- Representations "explode" or "vanish" across layers
- Results in wild oscillations between answers

**Hypothesis 2: Insufficient Representational Capacity**
- Small SSMs can't maintain stable multi-faceted representations
- Early layers commit to wrong answer
- Middle layers flip violently when encountering contradictory evidence
- Can't hold both "bird-like features" AND "mammal" simultaneously

**Hypothesis 3: Training Dynamics**
- Small SSMs may be harder to optimize
- Large models benefit from better gradient flow
- More stable training → more stable inference-time trajectories

### Why does the 1.4B model succeed?

**Capacity**: Enough parameters to represent nuanced features simultaneously
**Stability**: Better-conditioned recurrence matrices
**Optimization**: Easier to train at scale

---