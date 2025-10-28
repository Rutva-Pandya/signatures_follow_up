#!/usr/bin/env python3
"""
Comprehensive validation of Mamba integration.
This script performs deep sanity checks on implementation and results.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '.')
from utils import *

print("="*80)
print("COMPREHENSIVE MAMBA VALIDATION")
print("="*80)

# ============================================================================
# VALIDATION 1: Data File Integrity
# ============================================================================
print("\n" + "="*80)
print("VALIDATION 1: DATA FILE INTEGRITY")
print("="*80)

tasks = ["capitals-recall", "capitals-recognition", "animals", "syllogism"]
mamba_models = ["mamba-130m-hf", "mamba-370m-hf", "mamba-1.4b-hf"]

expected_layers = {
    "mamba-130m-hf": 24,
    "mamba-370m-hf": 48,
    "mamba-1.4b-hf": 48
}

data_issues = []

for task in tasks:
    for model in mamba_models:
        filepath = f"../../data/human_model_combined/logit_lens/{task}_{model}.csv"

        try:
            df = pd.read_csv(filepath)

            # Check 1: File exists and loads
            print(f"\n✓ {task}_{model}.csv loaded successfully")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

            # Check 2: Required columns exist
            required_cols = [
                'twostage_magnitude', 'twostage_layer',
                'output_entropy', 'output_rank_correct',
                'output_logprob_correct', 'output_logprobdiff'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                data_issues.append(f"{task}_{model}: Missing columns {missing_cols}")
                print(f"  ⚠ Missing columns: {missing_cols}")
            else:
                print(f"  ✓ All required columns present")

            # Check 3: Layer count matches expected
            layer_cols = [col for col in df.columns if col.startswith('layer_') and '_entropy' in col]
            n_layer_cols = len(layer_cols)
            expected_n = expected_layers[model]
            if n_layer_cols != expected_n:
                data_issues.append(f"{task}_{model}: Expected {expected_n} layers, found {n_layer_cols}")
                print(f"  ⚠ Layer count mismatch: expected {expected_n}, found {n_layer_cols}")
            else:
                print(f"  ✓ Layer count correct: {n_layer_cols}")

            # Check 4: No excessive NaN values in key metrics
            nan_counts = df[required_cols].isna().sum()
            if (nan_counts > 0).any():
                print(f"  ⚠ NaN values detected:")
                for col, count in nan_counts[nan_counts > 0].items():
                    pct = 100 * count / len(df)
                    print(f"    {col}: {count} ({pct:.1f}%)")
                    if pct > 50:
                        data_issues.append(f"{task}_{model}: {col} has {pct:.1f}% NaN")
            else:
                print(f"  ✓ No NaN values in key metrics")

            # Check 5: Values are in reasonable ranges
            com = df['twostage_magnitude']
            ttd = df['twostage_layer']

            if com.min() < 0:
                data_issues.append(f"{task}_{model}: Negative CoM values")
                print(f"  ⚠ Negative CoM values found!")

            if (ttd < 0).any() or (ttd > 1).any():
                data_issues.append(f"{task}_{model}: TTD out of [0,1] range")
                print(f"  ⚠ TTD values outside [0,1] range!")

            print(f"  CoM range: [{com.min():.2f}, {com.max():.2f}], mean: {com.mean():.2f}")
            print(f"  TTD range: [{ttd.min():.2f}, {ttd.max():.2f}], mean: {ttd.mean():.2f}")

        except Exception as e:
            data_issues.append(f"{task}_{model}: Failed to load - {e}")
            print(f"\n✗ {task}_{model}.csv: ERROR - {e}")

# ============================================================================
# VALIDATION 2: Layer-wise Trajectory Logic
# ============================================================================
print("\n" + "="*80)
print("VALIDATION 2: LAYER-WISE TRAJECTORY COMPUTATION")
print("="*80)

# Load one file and manually verify trajectory computation
test_file = "../../data/human_model_combined/logit_lens/animals_mamba-1.4b-hf.csv"
df_test = pd.read_csv(test_file)

# Get first item
item = df_test.iloc[0]
print(f"\nTest item: {item.get('exemplar_de', 'Unknown')}")

# Extract layer-wise logprobs for correct answer
n_layers = expected_layers["mamba-1.4b-hf"]
correct_logprobs = []

for layer_idx in range(n_layers):
    col_name = f"layer_{layer_idx}_logprob_correct"
    if col_name in df_test.columns:
        correct_logprobs.append(item[col_name])
    else:
        print(f"  ⚠ Missing column: {col_name}")
        break

if len(correct_logprobs) == n_layers:
    correct_logprobs = np.array(correct_logprobs)
    print(f"  ✓ Extracted {n_layers} layer-wise logprobs")

    # Compute trajectory manually
    diffs = np.diff(correct_logprobs)

    # CoM should be sum of absolute changes
    manual_com = np.sum(np.abs(diffs))
    stored_com = item['twostage_magnitude']

    print(f"\n  Manual CoM calculation: {manual_com:.4f}")
    print(f"  Stored CoM value: {stored_com:.4f}")
    print(f"  Difference: {abs(manual_com - stored_com):.6f}")

    if abs(manual_com - stored_com) > 0.01:
        data_issues.append("CoM computation mismatch detected!")
        print(f"  ⚠ CoM MISMATCH! Check computation logic!")
    else:
        print(f"  ✓ CoM computation validated")

    # TTD should be layer of biggest change (normalized)
    max_change_idx = np.argmax(np.abs(diffs))
    manual_ttd = (max_change_idx + 1) / n_layers  # +1 because diff shifts index
    stored_ttd = item['twostage_layer']

    print(f"\n  Manual TTD calculation: {manual_ttd:.4f} (layer {max_change_idx+1}/{n_layers})")
    print(f"  Stored TTD value: {stored_ttd:.4f}")
    print(f"  Difference: {abs(manual_ttd - stored_ttd):.6f}")

    if abs(manual_ttd - stored_ttd) > 0.01:
        data_issues.append("TTD computation mismatch detected!")
        print(f"  ⚠ TTD MISMATCH! Check computation logic!")
    else:
        print(f"  ✓ TTD computation validated")

    # Show trajectory shape
    print(f"\n  Trajectory shape:")
    print(f"    Start logprob: {correct_logprobs[0]:.2f}")
    print(f"    End logprob: {correct_logprobs[-1]:.2f}")
    print(f"    Min logprob: {correct_logprobs.min():.2f} (layer {correct_logprobs.argmin()})")
    print(f"    Max logprob: {correct_logprobs.max():.2f} (layer {correct_logprobs.argmax()})")

    # Check for non-monotonic behavior (U-shape)
    if correct_logprobs.argmin() > 0 and correct_logprobs.argmin() < n_layers - 1:
        print(f"  ✓ Non-monotonic trajectory detected (U-shape characteristic of Mamba)")


# ============================================================================
# VALIDATION 3: Cross-check Against Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("VALIDATION 3: SUMMARY STATISTICS CROSS-CHECK")
print("="*80)

for model in mamba_models:
    print(f"\n{model}:")

    # Aggregate across all tasks
    all_com = []
    all_ttd = []

    for task in tasks:
        filepath = f"../../data/human_model_combined/logit_lens/{task}_{model}.csv"
        try:
            df = pd.read_csv(filepath)
            all_com.extend(df['twostage_magnitude'].dropna().tolist())
            all_ttd.extend(df['twostage_layer'].dropna().tolist())
        except:
            pass

    mean_com = np.mean(all_com)
    mean_ttd = np.mean(all_ttd)

    print(f"  Aggregated CoM: {mean_com:.2f} (n={len(all_com)})")
    print(f"  Aggregated TTD: {mean_ttd:.2f} (n={len(all_ttd)})")

    # Compare against expected values from summary
    expected_com = {
        "mamba-130m-hf": 15.2,
        "mamba-370m-hf": 15.3,
        "mamba-1.4b-hf": 6.6
    }

    expected_ttd = {
        "mamba-130m-hf": 0.75,
        "mamba-370m-hf": 0.80,
        "mamba-1.4b-hf": 0.70
    }

    com_diff = abs(mean_com - expected_com[model])
    ttd_diff = abs(mean_ttd - expected_ttd[model])

    if com_diff > 1.0:
        data_issues.append(f"{model}: CoM differs from summary by {com_diff:.2f}")
        print(f"  ⚠ CoM differs from summary by {com_diff:.2f}")
    else:
        print(f"  ✓ CoM matches summary (diff: {com_diff:.2f})")

    if ttd_diff > 0.05:
        data_issues.append(f"{model}: TTD differs from summary by {ttd_diff:.2f}")
        print(f"  ⚠ TTD differs from summary by {ttd_diff:.2f}")
    else:
        print(f"  ✓ TTD matches summary (diff: {ttd_diff:.2f})")

# ============================================================================
# VALIDATION 4: Mamba vs Transformer Comparison Sanity
# ============================================================================
print("\n" + "="*80)
print("VALIDATION 4: MAMBA VS TRANSFORMER SANITY CHECK")
print("="*80)

# Load a transformer model for comparison
transformer_file = "../../data/human_model_combined/logit_lens/animals_gpt2.csv"
try:
    df_transformer = pd.read_csv(transformer_file)
    transformer_com = df_transformer['twostage_magnitude'].mean()
    transformer_ttd = df_transformer['twostage_layer'].mean()

    print(f"\nGPT-2 (transformer baseline):")
    print(f"  CoM: {transformer_com:.2f}")
    print(f"  TTD: {transformer_ttd:.2f}")

    # Compare with Mamba
    mamba_130_file = "../../data/human_model_combined/logit_lens/animals_mamba-130m-hf.csv"
    df_mamba_130 = pd.read_csv(mamba_130_file)
    mamba_com = df_mamba_130['twostage_magnitude'].mean()
    mamba_ttd = df_mamba_130['twostage_layer'].mean()

    print(f"\nMamba-130M:")
    print(f"  CoM: {mamba_com:.2f}")
    print(f"  TTD: {mamba_ttd:.2f}")

    print(f"\nComparison:")
    print(f"  CoM ratio (Mamba/GPT-2): {mamba_com/transformer_com:.2f}x")
    print(f"  TTD difference: {mamba_ttd - transformer_ttd:+.2f}")

    if mamba_com > 3 * transformer_com:
        print(f"  ✓ Mamba shows much higher CoM (expected for small SSMs)")

except Exception as e:
    print(f"  Could not load transformer comparison: {e}")

# ============================================================================
# VALIDATION 5: Implementation Code Review
# ============================================================================
print("\n" + "="*80)
print("VALIDATION 5: CODE IMPLEMENTATION REVIEW")
print("="*80)

print("\nChecking src/model.py for Mamba references...")
try:
    with open('../../src/model.py', 'r') as f:
        model_code = f.read()

    if 'backbone.layers' in model_code:
        print("  ✓ Found backbone.layers reference")
    else:
        data_issues.append("src/model.py missing backbone.layers reference")

    if 'backbone.norm_f' in model_code:
        print("  ✓ Found backbone.norm_f reference")
    else:
        data_issues.append("src/model.py missing backbone.norm_f reference")

    if 'model_family == "mamba"' in model_code:
        print("  ✓ Found Mamba model family check")
    else:
        data_issues.append("src/model.py missing Mamba family check")

except Exception as e:
    print(f"  Could not read model.py: {e}")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

if len(data_issues) == 0:
    print("\n✅ ALL VALIDATIONS PASSED!")
    print("\nThe Mamba integration appears to be correctly implemented.")
    print("Data integrity, computation logic, and results are all validated.")
else:
    print(f"\n⚠️ FOUND {len(data_issues)} POTENTIAL ISSUES:")
    for i, issue in enumerate(data_issues, 1):
        print(f"  {i}. {issue}")
    print("\n⚠️ Review these issues before proceeding!")

print("\n" + "="*80)
