"""
Compute and compare accuracy of Mamba, RWKV, and Transformer models on existing tasks.

This script calculates final-layer accuracy for all models across tasks using the
processed model outputs (output_logprobdiff = first_logprob_correct - first_logprob_incorrect).
This matches the metric used throughout the analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
DATA_DIR = Path("../data/model_output/processed")
OUTPUT_DIR = Path("./accuracy_analysis")
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

# Transformer models to compare against
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

def compute_accuracy(file_path, model_name):
    """
    Compute accuracy from processed model metrics file.

    Args:
        file_path: Path to processed metrics CSV file
        model_name: Name of the model to filter for

    Returns:
        Dictionary with accuracy statistics
    """
    try:
        df = pd.read_csv(file_path)

        # Filter to specific model
        model_df = df[df['model'] == model_name]

        if len(model_df) == 0:
            return None

        # Compute accuracy: model is correct if output_logprobdiff > 0
        # (i.e., logprob(correct) > logprob(incorrect))
        accuracy = (model_df['output_logprobdiff'] > 0).mean()
        n_items = len(model_df)

        return {
            'accuracy': accuracy,
            'n_items': n_items
        }
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing {file_path} for {model_name}: {e}")
        return None

def main():
    """Main function to compute and compare accuracies."""

    results = []

    # Combine all models
    all_models = MAMBA_MODELS + RWKV_MODELS + TRANSFORMER_MODELS

    # Compute accuracy for each model-task combination
    for task in TASKS:
        file_path = DATA_DIR / f"{task}_metrics_logit_lens.csv"

        for model in all_models:
            accuracy_stats = compute_accuracy(file_path, model)

            if accuracy_stats is not None:
                results.append({
                    'model': model,
                    'model_family': get_model_family(model),
                    'model_size': extract_model_size(model),
                    'task': task,
                    'accuracy': accuracy_stats['accuracy'],
                    'n_items': accuracy_stats['n_items']
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by model family and size
    results_df = results_df.sort_values(['model_family', 'model_size'])

    # Save raw results
    output_file = OUTPUT_DIR / "model_accuracies.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved accuracy results to {output_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("ACCURACY SUMMARY BY MODEL FAMILY AND TASK")
    print("="*80)

    for task in TASKS:
        print(f"\n{task.upper()}:")
        print("-" * 80)
        task_df = results_df[results_df['task'] == task]

        for family in ['Mamba', 'RWKV', 'GPT-2', 'Llama', 'OLMo']:
            family_df = task_df[task_df['model_family'] == family]
            if len(family_df) > 0:
                print(f"\n  {family}:")
                for _, row in family_df.iterrows():
                    print(f"    {row['model']:30s} | Accuracy: {row['accuracy']:.3f} | N={row['n_items']}")

    # Create visualization comparing architectures
    create_accuracy_plots(results_df)

    return results_df

def create_accuracy_plots(df):
    """Create plots comparing accuracies across architectures."""

    # Set up color palette
    palette = {
        'Mamba': '#808080',  # Grey
        'RWKV': '#4169E1',   # Royal Blue
        'GPT-2': '#90EE90',  # Light Green
        'Llama': '#FFA500',  # Orange
        'OLMo': '#DDA0DD'    # Plum
    }

    # Plot 1: Accuracy by model size for each task
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        task_df = df[df['task'] == task]

        for family in ['Mamba', 'RWKV', 'GPT-2', 'Llama', 'OLMo']:
            family_df = task_df[task_df['model_family'] == family]
            if len(family_df) > 0:
                ax.plot(family_df['model_size'], family_df['accuracy'],
                       marker='o', label=family, color=palette.get(family, 'gray'),
                       linewidth=2, markersize=8)

        ax.set_xlabel('Model Size (M parameters)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{task.replace("-", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # Hide the last subplot if we have fewer tasks than subplots
    if len(TASKS) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_by_size.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'accuracy_by_size.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {OUTPUT_DIR / 'accuracy_by_size.pdf'}")
    plt.close()

    # Plot 2: Heatmap of accuracies
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pivot table for heatmap
    pivot_df = df.pivot_table(
        values='accuracy',
        index='model',
        columns='task',
        aggfunc='mean'
    )

    # Sort by model family and size
    model_order = df.sort_values(['model_family', 'model_size'])['model'].unique()
    pivot_df = pivot_df.reindex(model_order)

    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'},
                ax=ax, linewidths=0.5)

    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Across Tasks', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {OUTPUT_DIR / 'accuracy_heatmap.pdf'}")
    plt.close()

    # Plot 3: Average accuracy across all tasks by model family
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_accuracy = df.groupby(['model', 'model_family', 'model_size'])['accuracy'].mean().reset_index()
    avg_accuracy = avg_accuracy.sort_values('model_size')

    for family in ['Mamba', 'RWKV', 'GPT-2', 'Llama', 'OLMo']:
        family_df = avg_accuracy[avg_accuracy['model_family'] == family]
        if len(family_df) > 0:
            ax.plot(family_df['model_size'], family_df['accuracy'],
                   marker='o', label=family, color=palette.get(family, 'gray'),
                   linewidth=2, markersize=10)

    ax.set_xlabel('Model Size (M parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Average Accuracy Across All Tasks', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'average_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'average_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"Saved average accuracy plot to {OUTPUT_DIR / 'average_accuracy.pdf'}")
    plt.close()

if __name__ == "__main__":
    results_df = main()

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Compare Mamba vs RWKV vs Transformers at similar sizes
    print("\nAverage accuracy by architecture (all tasks combined):")
    avg_by_family = results_df.groupby('model_family')['accuracy'].agg(['mean', 'std', 'count'])
    print(avg_by_family)
