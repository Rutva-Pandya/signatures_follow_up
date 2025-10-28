"""
Comprehensive analysis of Change of Mind (CoM) in Mamba, RWKV, and Transformer models.

This script analyzes two-stage processing signatures across architectures:
- CoM magnitude: How much predictions flip between layers
- CoM timing: When (which layer) the change occurs
- CoM patterns: Differences by task, accuracy, and model family
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
DATA_DIR = Path("../data/model_output/processed")
OUTPUT_DIR = Path("./com_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define model families
MAMBA_MODELS = [
    "mamba-130m-hf",
    "mamba-370m-hf",
    "mamba-790m-hf",
    "mamba-1.4b-hf"
]

RWKV_MODELS = [
    "rwkv-4-169m-pile",
    "rwkv-4-430m-pile",
    "rwkv-4-1b5-pile"
]

TRANSFORMER_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "OLMo-1B-0724-hf"
]

TASKS = ["capitals-recall", "capitals-recognition", "animals", "syllogism", "gender"]

def extract_model_size(model_name):
    """Extract model size in millions of parameters for sorting."""
    if "130m" in model_name or "169m" in model_name:
        return 130
    elif "370m" in model_name or "430m" in model_name:
        return 400
    elif "790m" in model_name:
        return 790
    elif "1.4b" in model_name or "1.5b" in model_name or "1b5" in model_name:
        return 1400
    elif "1B" in model_name:
        return 1000
    elif "3B" in model_name or "3.2" in model_name:
        return 3000
    elif "gpt2-xl" in model_name:
        return 1500
    elif "gpt2-large" in model_name:
        return 774
    elif "gpt2-medium" in model_name:
        return 355
    elif "gpt2" == model_name:
        return 124
    else:
        return 0

def get_model_family(model_name):
    """Get model family for a given model name."""
    if "mamba" in model_name:
        return "Mamba"
    elif "rwkv" in model_name:
        return "RWKV"
    elif "llama" in model_name.lower():
        return "Llama"
    elif "olmo" in model_name.lower():
        return "OLMo"
    elif "gpt2" in model_name:
        return "GPT-2"
    else:
        return "Other"

def load_all_data():
    """Load all processed data and compute accuracy."""
    all_data = []

    all_models = MAMBA_MODELS + RWKV_MODELS + TRANSFORMER_MODELS

    for task in TASKS:
        file_path = DATA_DIR / f"{task}_metrics_logit_lens.csv"

        try:
            df = pd.read_csv(file_path)

            # Add task column
            df['task'] = task

            # Compute accuracy (model correct if output_logprobdiff > 0)
            df['is_correct'] = df['output_logprobdiff'] > 0

            # Filter to models we care about
            df = df[df['model'].isin(all_models)]

            # Add model metadata
            df['model_family'] = df['model'].apply(get_model_family)
            df['model_size'] = df['model'].apply(extract_model_size)

            all_data.append(df)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            continue

    combined = pd.concat(all_data, ignore_index=True)
    return combined

def main():
    """Main analysis function."""

    print("Loading data...")
    df = load_all_data()

    print(f"\nLoaded {len(df)} total predictions across {df['model'].nunique()} models and {df['task'].nunique()} tasks")

    # Save raw data
    output_file = OUTPUT_DIR / "com_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved raw data to {output_file}")

    # Create visualizations
    create_com_visualizations(df)

    # Print summary statistics
    print_com_statistics(df)

    return df

def create_com_visualizations(df):
    """Create comprehensive CoM visualizations."""

    # Set up color palette
    palette = {
        'Mamba': '#808080',  # Grey
        'RWKV': '#4169E1',   # Royal Blue
        'GPT-2': '#90EE90',  # Light Green
        'Llama': '#FFA500',  # Orange
        'OLMo': '#DDA0DD'    # Plum
    }

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # ==================== PLOT 1: CoM by task and architecture ====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        task_df = df[df['task'] == task].copy()

        # Aggregate by model
        model_stats = task_df.groupby(['model', 'model_family', 'model_size']).agg({
            'twostage_magnitude': 'mean'
        }).reset_index()

        # Plot each family
        for family in ['Mamba', 'RWKV', 'GPT-2', 'Llama', 'OLMo']:
            family_df = model_stats[model_stats['model_family'] == family]
            if len(family_df) > 0:
                family_df = family_df.sort_values('model_size')
                ax.plot(family_df['model_size'], family_df['twostage_magnitude'],
                       marker='o', label=family, color=palette.get(family, 'gray'),
                       linewidth=2.5, markersize=10, alpha=0.8)

        ax.set_xlabel('Model Size (M parameters)', fontsize=11, fontweight='bold')
        ax.set_ylabel('CoM Magnitude', fontsize=11, fontweight='bold')
        ax.set_title(f'{task.replace("-", " ").title()}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

    # Hide the last subplot
    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'com_by_task.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'com_by_task.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {OUTPUT_DIR / 'com_by_task.pdf'}")
    plt.close()

    # ==================== PLOT 2: Average CoM across all tasks ====================
    fig, ax = plt.subplots(figsize=(12, 7))

    # Aggregate across all tasks
    model_stats = df.groupby(['model', 'model_family', 'model_size']).agg({
        'twostage_magnitude': 'mean'
    }).reset_index()

    for family in ['Mamba', 'RWKV', 'GPT-2', 'Llama', 'OLMo']:
        family_df = model_stats[model_stats['model_family'] == family]
        if len(family_df) > 0:
            family_df = family_df.sort_values('model_size')
            ax.plot(family_df['model_size'], family_df['twostage_magnitude'],
                   marker='o', label=family, color=palette.get(family, 'gray'),
                   linewidth=3, markersize=12, alpha=0.8)

    ax.set_xlabel('Model Size (M parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average CoM Magnitude', fontsize=14, fontweight='bold')
    ax.set_title('Average Change of Mind Magnitude Across Tasks',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=13, framealpha=0.9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'com_average.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'com_average.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_DIR / 'com_average.pdf'}")
    plt.close()

    # ==================== PLOT 3: CoM for correct vs incorrect predictions ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, family in enumerate(['GPT-2', 'Mamba', 'RWKV']):
        ax = axes[idx]
        family_df = df[df['model_family'] == family].copy()

        # Aggregate by model and correctness
        stats = family_df.groupby(['model', 'model_size', 'is_correct']).agg({
            'twostage_magnitude': 'mean'
        }).reset_index()

        # Plot correct vs incorrect
        for is_correct in [True, False]:
            subset = stats[stats['is_correct'] == is_correct].sort_values('model_size')
            label = 'Correct' if is_correct else 'Incorrect'
            linestyle = '-' if is_correct else '--'
            ax.plot(subset['model_size'], subset['twostage_magnitude'],
                   marker='o', label=label, linestyle=linestyle,
                   linewidth=2.5, markersize=10, alpha=0.8)

        ax.set_xlabel('Model Size (M parameters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CoM Magnitude', fontsize=12, fontweight='bold')
        ax.set_title(f'{family}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

    plt.suptitle('CoM Magnitude: Correct vs Incorrect Predictions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'com_correct_vs_incorrect.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'com_correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_DIR / 'com_correct_vs_incorrect.pdf'}")
    plt.close()

def print_com_statistics(df):
    """Print summary statistics about CoM."""

    print("\n" + "="*80)
    print("CHANGE OF MIND STATISTICS")
    print("="*80)

    # Overall statistics by family
    print("\nAverage CoM magnitude by architecture:")
    family_stats = df.groupby('model_family').agg({
        'twostage_magnitude': ['mean', 'std', 'median'],
        'twostage_layer': 'mean'
    }).round(3)
    print(family_stats)

    # CoM by correctness
    print("\nCoM magnitude: Correct vs Incorrect predictions:")
    correctness_stats = df.groupby(['model_family', 'is_correct']).agg({
        'twostage_magnitude': ['mean', 'std']
    }).round(3)
    print(correctness_stats)

    # Percentage of predictions with CoM > 0
    print("\nPercentage of predictions with Change of Mind (CoM > 0):")
    com_pct = df.groupby('model_family').apply(
        lambda x: (x['twostage_magnitude'] > 0).mean() * 100
    ).round(1)
    print(com_pct)

    # CoM by task
    print("\nCoM magnitude by task:")
    task_stats = df.groupby(['task', 'model_family']).agg({
        'twostage_magnitude': 'mean'
    }).round(3)
    print(task_stats)

    # Correlation between CoM and accuracy
    print("\nCorrelation between CoM magnitude and accuracy by family:")
    for family in ['Mamba', 'RWKV', 'GPT-2', 'Llama', 'OLMo']:
        family_df = df[df['model_family'] == family]
        if len(family_df) > 0:
            corr = family_df['twostage_magnitude'].corr(family_df['is_correct'].astype(float))
            print(f"  {family}: {corr:.3f}")

if __name__ == "__main__":
    results_df = main()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
