# Pythia Model Setup (Q13 - Architecture vs Training Data)

## Research Question
**Does architecture or training data drive the inverse scaling in Mamba?**
- GPT-2 (Transformer + WebText) → Inverted U-shape CoM
- Mamba (SSM + Pile) → Inverse scaling CoM
- **Pythia (Transformer + Pile) → ?**

If Pythia shows inverted U-shape → architecture matters
If Pythia shows inverse scaling → training data matters

## Models (Full Pythia Suite - 8 models)
| Model | Size | HuggingFace ID | Comparison |
|-------|------|----------------|------------|
| Pythia-70M | 70M | `EleutherAI/pythia-70m` | Smaller than all existing models |
| Pythia-160M | 160M | `EleutherAI/pythia-160m` | Matches GPT-2 (124M), Mamba-130M, RWKV-169M |
| Pythia-410M | 410M | `EleutherAI/pythia-410m` | Matches GPT-2-medium (355M), Mamba-370M, RWKV-430M |
| Pythia-1B | 1B | `EleutherAI/pythia-1b` | Between medium and large |
| Pythia-1.4B | 1.4B | `EleutherAI/pythia-1.4b` | Matches Mamba-1.4B, GPT-2-xl (1.5B), RWKV-1.5B |
| Pythia-2.8B | 2.8B | `EleutherAI/pythia-2.8b` | Beyond existing models |
| Pythia-6.9B | 6.9B | `EleutherAI/pythia-6.9b` | Large scale |
| Pythia-12B | 12B | `EleutherAI/pythia-12b` | Very large scale |

**Total: 8 models × 5 tasks = 40 jobs**

## Tasks
All 5 tasks:
- `capitals-recall`
- `capitals-recognition`
- `animals`
- `gender`
- `syllogism`

## Cluster Execution

### 1. Make script executable
```bash
chmod +x cluster/batch_submit_pythia.sh
```

### 2. Submit all Pythia models
```bash
bash cluster/batch_submit_pythia.sh
```

### 3. Monitor jobs
```bash
squeue -u $USER
```

### 4. Check logs
```bash
tail -f logs/all_tasks_*.out
```

## Expected Output Files
```
data/model_output/logit_lens/
├── animals_pythia-70m.csv
├── animals_pythia-160m.csv
├── animals_pythia-410m.csv
├── animals_pythia-1b.csv
├── animals_pythia-1.4b.csv
├── animals_pythia-2.8b.csv
├── animals_pythia-6.9b.csv
├── animals_pythia-12b.csv
├── capitals-recall_pythia-70m.csv
├── capitals-recall_pythia-160m.csv
... (40 files total: 8 models × 5 tasks)
```

## Next Steps After Data Collection
1. Run processing: `python analysis/scripts/process_logit_lens.py`
2. Update analysis scripts to include Pythia
3. Compare CoM patterns: Pythia vs GPT-2 vs Mamba vs RWKV
4. Create visualization showing architecture vs training data effects
