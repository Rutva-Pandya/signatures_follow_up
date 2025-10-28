#!/usr/bin/env python3
"""
Validate CoM and TTD computation by manually calculating from raw data.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '.')
from utils import *

print("="*80)
print("TRAJECTORY COMPUTATION VALIDATION")
print("="*80)

# Test cases: one file per model size
test_cases = [
    ("animals", "mamba-130m-hf", 24),
    ("animals", "mamba-370m-hf", 48),
    ("animals", "mamba-1.4b-hf", 48),
]

all_passed = True

for task, model, n_layers in test_cases:
    print(f"\n{'='*80}")
    print(f"Testing: {task}_{model}")
    print(f"{'='*80}")

    # Load raw data (layer-by-layer predictions)
    raw_file = f"../../data/model_output/logit_lens/{task}_{model}.csv"
    processed_file = f"../../data/human_model_combined/logit_lens/{task}_{model}.csv"

    raw_df = pd.read_csv(raw_file)
    processed_df = pd.read_csv(processed_file)

    print(f"\nRaw data: {len(raw_df)} rows (should be ~{len(processed_df)} items × {n_layers} layers)")
    print(f"Processed data: {len(processed_df)} rows")

    # Get unique items
    unique_items = raw_df['item_id'].unique()
    print(f"Found {len(unique_items)} unique items")

    # Get unique CoM/TTD values from processed file (one per exemplar)
    processed_unique = processed_df.groupby('exemplar_de').first().reset_index()
    print(f"Unique exemplars in processed file: {len(processed_unique)}")

    # Compute CoM and TTD for all items in raw data
    manual_coms = []
    manual_ttds = []

    test_items = unique_items[:min(5, len(unique_items))]

    max_com_error = 0
    max_ttd_error = 0

    first_item_details = None

    for item_id in unique_items:
        # Extract layer-wise predictions for this item
        item_raw = raw_df[raw_df['item_id'] == item_id].sort_values('layer_idx')

        if len(item_raw) != n_layers:
            print(f"\n⚠️  Item {item_id}: Expected {n_layers} layers, got {len(item_raw)}")
            all_passed = False
            continue

        # Get layer-wise logprobs
        logprob_correct = item_raw['first_logprob_correct'].values
        logprob_incorrect = item_raw['first_logprob_incorrect'].values
        logprob_diff = logprob_correct - logprob_incorrect

        # Manual CoM computation: magnitude of dip below 0
        m = logprob_diff.min()
        final_diff = logprob_diff[-1]
        if m >= 0:
            manual_com = 0
        else:
            manual_com = min(0, final_diff) - m

        # Manual TTD computation: layer where all subsequent values are positive
        def last_neg_index(vals):
            for i in range(len(vals)-1, -1, -1):
                if vals[i] < 0:
                    return i
            return None

        def layer_stay_pos(vals):
            """Layer where all subsequent values (including current) are positive."""
            last_neg = last_neg_index(vals)
            if last_neg is None:
                # Always positive
                return 0
            elif last_neg == len(vals) - 1:
                # Always negative
                return len(vals) - 1
            else:
                # Layer after last negative
                return last_neg + 1

        ttd_layer_idx = layer_stay_pos(logprob_diff)
        manual_ttd = (ttd_layer_idx + 1) / n_layers

        manual_coms.append(manual_com)
        manual_ttds.append(manual_ttd)

        # Store first item details for reporting
        if first_item_details is None:
            exemplar = item_raw.iloc[0]['exemplar']
            first_item_details = {
                'item_id': item_id,
                'exemplar': exemplar,
                'logprob_diff': logprob_diff,
                'manual_com': manual_com,
                'manual_ttd': manual_ttd,
                'ttd_layer_idx': ttd_layer_idx,
                'min_diff': m,
                'final_diff': final_diff
            }

    # Get stored CoM/TTD values from processed file
    stored_coms = processed_unique['twostage_magnitude'].values
    stored_ttds = processed_unique['twostage_layer'].values

    # Sort both for comparison (since names don't match)
    manual_coms_sorted = sorted(manual_coms)
    stored_coms_sorted = sorted(stored_coms)
    manual_ttds_sorted = sorted(manual_ttds)
    stored_ttds_sorted = sorted(stored_ttds)

    # Compute errors
    com_errors = [abs(m - s) for m, s in zip(manual_coms_sorted, stored_coms_sorted)]
    ttd_errors = [abs(m - s) for m, s in zip(manual_ttds_sorted, stored_ttds_sorted)]

    max_com_error = max(com_errors)
    max_ttd_error = max(ttd_errors)

    # Print details for first item
    d = first_item_details
    print(f"\nSample item: {d['exemplar']} (item_id={d['item_id']})")
    print(f"  Logprob_diff trajectory: {d['logprob_diff'][0]:.2f} → {d['logprob_diff'][-1]:.2f}")
    print(f"  Min logprob_diff: {d['min_diff']:.2f} (dip below 0)")
    print(f"  Final logprob_diff: {d['final_diff']:.2f}")
    print(f"\n  Manual CoM: {d['manual_com']:.4f} (= min(0, {d['final_diff']:.2f}) - {d['min_diff']:.2f})")
    print(f"  Manual TTD: {d['manual_ttd']:.4f} (layer {d['ttd_layer_idx']+1}/{n_layers})")

    print(f"\nComparison across all {len(manual_coms)} items:")
    print(f"  Manual CoM range: [{min(manual_coms):.2f}, {max(manual_coms):.2f}]")
    print(f"  Stored CoM range: [{min(stored_coms):.2f}, {max(stored_coms):.2f}]")
    print(f"  Manual TTD range: [{min(manual_ttds):.2f}, {max(manual_ttds):.2f}]")
    print(f"  Stored TTD range: [{min(stored_ttds):.2f}, {max(stored_ttds):.2f}]")

    if max_com_error > 0.01:
        print(f"  ✗ CoM MISMATCH! Max error: {max_com_error:.4f}")
        all_passed = False
    if max_ttd_error > 0.01:
        print(f"  ✗ TTD MISMATCH! Max error: {max_ttd_error:.4f}")
        all_passed = False

    # Summary for this model
    print(f"\n{model} summary:")
    print(f"  Max CoM error across {len(test_items)} items: {max_com_error:.6f}")
    print(f"  Max TTD error across {len(test_items)} items: {max_ttd_error:.6f}")

    if max_com_error < 0.01 and max_ttd_error < 0.01:
        print(f"  ✓ All computations validated!")
    else:
        print(f"  ✗ Computation errors detected!")
        all_passed = False

print(f"\n{'='*80}")
print("FINAL RESULT")
print(f"{'='*80}")

if all_passed:
    print("\n✅ ALL TRAJECTORY COMPUTATIONS VALIDATED!")
    print("\nCoM and TTD are correctly computed from layer-wise predictions.")
    print("The implementation is mathematically sound.")
else:
    print("\n❌ VALIDATION FAILED!")
    print("\nSome computations do not match. Review the processing pipeline.")

print(f"\n{'='*80}\n")
