# File: utils.py
# Description: helper functions used during LM evaluation

import os
import numpy as np


HF_TOKEN = os.getenv("HF_TOKEN")
TASKS = [
    "capitals-recall", 
    "capitals-recognition",
    "animals",
    "gender",
    "syllogism"
]
TL_MODELS = [
    "gpt2", "gpt2-xl", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"
]

def flatten(xss: list) -> list:
    """Helper function for flattening a list."""
    return [x for xs in xss for x in xs]

def get_reduction_fn(reduction: str):
    """Returns an anonymous function that applies a reduction to a Tensor."""
    if reduction == "mean":
        reduction_fn = lambda x: x.mean(0).item()
    elif reduction == "sum":
        reduction_fn = lambda x: x.sum(0).item()
    elif reduction == "mean_and_sum":
        reduction_fn = lambda x: (x.mean(0).item(), x.sum(0).item())
    else:
        raise ValueError("`reduction` should be 'mean', 'sum', or 'mean_and_sum")
    return reduction_fn

def get_file_safe_model_name(model: str) -> str:
    """
    Returns a file-safe version of a Huggingface model identifier by
    only keeping the model name after a forward slash (/).
    Example: meta-llama/Llama-2-7b-hf --> Llama-2-7b-hf
    """
    safe_model_name = model.split("/")[-1] if "/" in model else model
    return safe_model_name

def get_rank(x, indices, one_indexed=True):
    """
    Adapted from https://stephantul.github.io/python/pytorch/2020/09/18/fast_topk/
    """
    vals = x[range(len(x)), indices]
    rank = (x > vals[:, None]).long().sum(1)
    if one_indexed:
        rank += 1
    return rank

def get_model_family(model_name):
    model_name = model_name.lower()
    families = ["llama", "olmo", "gemma", "pythia", "gpt", "falcon", "mamba", "rwkv"]
    for family in families:
        if family in model_name:
            return family
    raise ValueError(f"Unrecognized model family for {model_name}")

def get_vals_of_tokens(vals, tokens):
    """
    Helper function for taking a Tensor of vocabulary-related values `vals`
    of shape (n_tokens x n_layers x vocab_size), extracting the values 
    corresponding to each token ID in `tokens`, and reshaping to be of shape
    (n_layers x n_tokens)
    """
    n_layers = vals.size(1)
    all_layer_vals = np.array([
        [
            vals[i][layer_id][token_id].detach().cpu()
            for i, token_id in enumerate(tokens)
        ]
        for layer_id in range(n_layers)
    ])
    return all_layer_vals

def get_first_token_of_answers(
    tokenizer,
    prefix,
    answer_texts,
    sep: str = " "
):
    # Construct full texts by combined prefix with answer texts.
    full_texts = [prefix + sep + answer for answer in answer_texts]
    full_tokens = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
    # "Prefix" tokens by combining prefix and continuation separately.
    prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    # Answer tokens correspond to everything after the prefix.
    answer_tokens = [t[len(prefix_tokens):] for t in full_tokens]
    # Get first token of each answer.
    first_tokens = [tokenized[0] for tokenized in answer_tokens]
    return first_tokens

def get_conditions_for_capitals_recognition_experiment(stim_row) -> list[dict]:
    """
    Return list of dictionaries defining conditions for evaluation
    in the capitals recognition (forced choice) experiment.
    """
    conditions = []
    for correct_first in [True, False]:
        if correct_first:
            options = [stim_row["correct"], stim_row["incorrect"]]
        else:
            options = [stim_row["incorrect"], stim_row["correct"]]
        query = f"""The capital of {stim_row.entity} is either {options[0]} or {options[1]}. In fact,"""
        conditions.append(dict(
            correct_first=correct_first,
            query=query
        ))
    return conditions
