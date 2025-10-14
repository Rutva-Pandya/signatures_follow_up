#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper functions to be used across notebooks for analysis/visualizations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import seaborn as sns

######################################################
# Model-related helper functions and global variables
######################################################
def get_vocab_size(model):
    if "gpt2" in model:
        return 50257
    elif "llama-2" in model.lower():
        return 32000
    elif "llama-3.1" in model.lower():
        return 128000
    elif "olmo-2" in model.lower():
        return 50000
    elif "gemma-2" in model.lower():
        return 256000
    elif "falcon3" in model.lower():
        return 131000
    elif "mamba" in model.lower():
        return 50277
    elif "vit" in model.lower():
        return 16

N_LAYERS = {
    # language models
    "gpt2": 12, "gpt2-medium": 24, "gpt2-xl": 48,
    "Llama-2-7b-hf": 32, "Llama-2-13b-hf": 40, "Llama-2-70b-hf": 80,
    "Llama-3.1-8B": 32, "Llama-3.1-70B": 80, "Llama-3.1-405B": 126,
    "gemma-2-2b": 26, "gemma-2-9b": 42, "gemma-2-27b": 46,
    "OLMo-2-1124-7B": 32, "OLMo-2-1124-13B": 40, "OLMo-2-0325-32B": 64,
    "Falcon3-1B-Base": 18, "Falcon3-3B-Base": 22, "Falcon3-10B-Base": 40,
    # SSM models
    "mamba-130m-hf": 24, "mamba-370m-hf": 48, "mamba-790m-hf": 48,
    "mamba-1.4b-hf": 48, "mamba-2.8b-hf": 64,
    # vision models
    "vit_small_patch16_224": 12, "vit_base_patch16_224": 12
}

MODELS = [
  "gpt2", "gpt2-medium", "gpt2-xl",
  "Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf",
  "Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B",
  "gemma-2-2b", "gemma-2-9b", "gemma-2-27b",
  "OLMo-2-1124-7B", "OLMo-2-1124-13B", "OLMo-2-0325-32B",
  "Falcon3-1B-Base", "Falcon3-3B-Base", "Falcon3-10B-Base"
]
VISION_MODELS = ["vit_small_patch16_224", "vit_base_patch16_224"]
MODEL_FAMILY_MAP = {
    "gpt2": "GPT-2",
    "llama-2": "Llama-2",
    "llama-3.1": "Llama-3.1",
    "olmo-2": "OLMo-2",
    "gemma-2": "Gemma-2",
    "falcon3": "Falcon-3"
}
MODEL_FAMILIES = [
    MODEL_FAMILY_MAP[f] for f in [
        "gpt2", "llama-2", "llama-3.1", "olmo-2", "gemma-2", "falcon3"
    ]
]

def get_model_size(model):
    if model == "gpt2":
        return 0.124
    elif model == "gpt2-medium":
        return 0.355
    elif model == "gpt2-xl":
        return 1.5
    elif model == "vit_small_patch16_224":
        return 0.022
    elif model == "vit_base_patch16_224":
        return 0.086
    else:
        splits = model.lower().split("-")
        for s in splits:
            if s.endswith("b"):
                return int(s.replace("b", ""))
            
def get_model_family(model):
    for orig_name, pretty_name in MODEL_FAMILY_MAP.items():
        if orig_name in model.lower():
            return pretty_name

######################################################
# Task-related helper functions and global variables
######################################################
TASKS = ["capitals-recall", "capitals-recognition", "animals", "gender", "syllogism"]
TASK_NAMES = {
    "capitals-recall": "Capitals recall", 
    "capitals-recognition": "Capitals recognition", 
    "animals": "Animal categories", 
    "gender": "Gender bias",
    "syllogism": "Syllogisms"
}

# Define list of meta variables for the tasks.
TASK_META_VAR_MAP = {
    "capitals-recall": ["entity", "correct", "incorrect"],
    "capitals-recognition": ["entity", "correct", "incorrect"],
    "animals": ["condition", "exemplar", "correct", "incorrect"],
    "gender": ["profession", "correct", "incorrect"],
    "syllogism": ["correct", "incorrect"],
    "vision": []
}
ITEM_META_VAR_MAP = {
    "capitals-recall": ["model", "item_id", "condition"],
    "capitals-recognition": ["model", "item_id", "condition", "correct_first"],
    "animals": ["model", "item_id"],
    "gender": ["model", "item_id", "condition"],
    "syllogism": ["model", "unique_id", "syllogism_name", "order_first", "is_valid", "is_realistic", "is_consistent"],
    "vision": ["model", "item_id", "dataset_name"]
}

TASK_DVS = {
    "capitals-recall": [
        "response_correct_strict", 
        "response_correct_gpt4", 
        "rt", 
        "time_stroke_after_last_empty_trial", 
        "n_keystrokes_len_norm",
        "n_backspace"
    ],
    "capitals-recognition": ["response_correct", "rt"],
    "animals": ["response_correct", "RT", "MAD", "AUC", "xpos_flips", "acc_max_time"],
    "syllogism": ["response_correct", "rt"],
    "vision": ["response_correct", "rt"]
}
DV_NAMES = {
    "response_correct_strict": "Accuracy (strict)",
    "response_correct_gpt4": "Accuracy (lenient)",
    "rt": "RT",
    "RT": "RT",
    "n_keystrokes_len_norm": "# Presses / len(final answer)",
    "n_backspace": "# Backspace presses",
    "time_stroke_after_last_empty_trial": "Time of first press after last empty",
    "response_correct": "Accuracy",
    "MAD": "MAD",
    "AUC": "AUC",
    "xpos_flips": "Flips",
    "acc_max_time": "T(max acc)"
}
DV_TYPES = {
    "response_correct_strict": "Accuracy",
    "response_correct_gpt4": "Accuracy",
    "response_correct": "Accuracy",
    "rt": "RT",
    "RT": "RT",
    "n_keystrokes_len_norm": "Other processing DVs",
    "n_backspace": "Other processing DVs",
    "time_stroke_after_last_empty_trial": "Other processing DVs",
    "MAD": "Other processing DVs",
    "AUC": "Other processing DVs",
    "xpos_flips": "Other processing DVs",
    "acc_max_time": "Other processing DVs"
}
def get_dv_groups():
    dv_groups = [
        (dv_group, [dv for dv, dv_type in DV_TYPES.items() if dv_type==dv_group])
        for dv_group in ["Accuracy", "RT", "Other processing DVs"]
    ]
    return dv_groups

#########################################################
# Variable-related helper functions and global variables
#########################################################
# Define output IVs (from the final layer).
OUTPUT_IV_MAP = {
    "entropy_first_token": "output_entropy",
    "reciprocal_rank_correct": "output_rank_correct",
    "first_logprob_correct": "output_logprob_correct",
    "first_logprob_diff": "output_logprobdiff"
}
OLD_OUTPUT_IVS = ["entropy_first_token", "reciprocal_rank_correct", "first_logprob_correct", "first_logprob_diff"]
OUTPUT_IVS = [OUTPUT_IV_MAP[i] for i in OLD_OUTPUT_IVS]
print(f"OUTPUT measures ({len(OUTPUT_IVS)}):", OUTPUT_IVS)

# Define process IVs based purely on probabilities.
PROB_IVS = []
for output_iv in OUTPUT_IVS:
    # Get base name of metric (e.g., "entropy")
    iv = output_iv.replace("output_", "")
    # Add AUC measure(s).
    if iv == "logprobdiff":
        PROB_IVS.append(f"auc_{iv}_pos")
        PROB_IVS.append(f"auc_{iv}_neg")
    else:
        PROB_IVS.append(f"auc_{iv}")
    # Add biggest change measure.
    PROB_IVS.append(f"layer_biggest_change_{iv}")

# Define process IVS based on scalar projections.
BOOST_IVS = [
    "auc_boosting_pos",
    "auc_boosting_neg",
    "layer_argmax_boosting"
]

# Define "change of mind" (two-stage processing) IVs.
TWOSTAGE_IVS = [
    "twostage_magnitude",
    "twostage_magnitude_latter34",
    "twostage_layer"
]

PROCESS_IVS = PROB_IVS + BOOST_IVS + TWOSTAGE_IVS
print(f"PROCESS measures ({len(PROCESS_IVS)}):", PROCESS_IVS)

def get_metric_group(iv):
    if "entropy" in iv:
        return "Uncertainty"
    elif "rank" in iv or "logprob_correct" in iv:
        return "Confidence"
    elif "logprobdiff" in iv:
        return "Relative confidence"
    elif "boosting" in iv:
        return "Boosting"
    elif "twostage" in iv:
        return "Two-Stage"
    else:
        return iv
    
def get_quantity_type(iv):
    if "auc" in iv:
        return "AUC"
    elif "layer" in iv:
        return "MaxDeltaLayer"
    elif "twostage" in iv:
        return "TwoStage"
    elif iv == "baseline":
        return iv
    else:
        return None

#########################################################
# Plotting-related helper functions and global variables
#########################################################

METRIC_GROUP_PAL = {
    "Uncertainty": "#FF7676FF",
    "Confidence": "#F9D662FF",
    "Relative confidence": "#7CAB7DFF",
    "Boosting": "#75B7D1FF",
    "Two-Stage": "mediumpurple"
}
DV_GROUP_PAL = {
    "Accuracy": "#72874EFF",
    "RT": "#A4BED5FF",
    "Other processing DVs": "#FED789FF"
}
TWOSTAGE_PAL = {
    "Competitor": sns.color_palette("Set3")[3],
    "NoCompetitor": sns.color_palette("Accent")[1]
}

blues = sns.color_palette("Blues")
oranges = sns.color_palette("Oranges")
greens = sns.color_palette("Greens")
reds = sns.color_palette("Reds")
purples = sns.color_palette("Purples")
browns = sns.color_palette("BrBG")
MODEL_PAL = {
    "gpt2": blues[0], "gpt2-medium": blues[1], "gpt2-xl": blues[2], 
    "Llama-2-7b-hf": oranges[0], "Llama-2-13b-hf": oranges[1], "Llama-2-70b-hf": oranges[2],
    "Llama-3.1-8B": greens[0], "Llama-3.1-70B": greens[1], "Llama-3.1-405B": greens[2],
    "gemma-2-2b": reds[0], "gemma-2-9b": reds[1], "gemma-2-27b": reds[2],
    "OLMo-2-1124-7B": purples[0], "OLMo-2-1124-13B": purples[1], "OLMo-2-0325-32B": purples[2],
    "Falcon3-1B-Base": browns[2], "Falcon3-3B-Base": browns[1], "Falcon3-10B-Base": browns[0],
    "vit_small_patch16_224": "lightpink", "vit_base_patch16_224": "palevioletred",
    "Human": "k"
}
MODEL_MAP = {
    "gpt2": "GPT-2", "gpt2-medium": "GPT-2 Med", "gpt2-xl": "GPT-2 XL",
    "Llama-2-7b-hf": "Llama-2 7B", "Llama-2-13b-hf": "Llama-2 13B", "Llama-2-70b-hf": "Llama-2 70B",
    "Llama-3.1-8B": "Llama-3.1 8B", "Llama-3.1-70B": "Llama-3.1 70B", "Llama-3.1-405B": "Llama-3.1 405B",
    "gemma-2-2b": "Gemma-2 2B", "gemma-2-9b": "Gemma-2 9B", "gemma-2-27b": "Gemma-2 27B",
    "OLMo-2-1124-7B": "OLMo-2 7B", "OLMo-2-1124-13B": "OLMo-2 13B", "OLMo-2-0325-32B": "OLMo-2 32B",
    "Falcon3-1B-Base": "Falcon-3 1B", "Falcon3-3B-Base": "Falcon-3 3B", "Falcon3-10B-Base": "Falcon-3 10B",
    "vit_small_patch16_224": "ViT Small", "vit_base_patch16_224": "ViT Base", 
    "Human": "Human"
}

CONTROL_STYLE = ":"
MAIN_STYLE = "-"
