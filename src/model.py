import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tuned_lens import TunedLens

from utils import get_rank, get_model_family, get_vals_of_tokens


def initialize_lm(
    model_name, 
    reduce_precision=False,
    cache_dir=None,
    **kwargs
):
    """Initializes language model for experiments."""
    if reduce_precision:
        print("Using quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None
    model = LM(
        model_name, 
        device_map="auto",
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        **kwargs
    )
    return model

class LM():
    """Base model class for LMs evaluated in our experiments."""
    def __init__(
        self,
        model_name: str,
        use_tuned_lens: bool = False,
        cache_dir: str = None,
        **load_kwargs
    ) -> None:
        # Store basic meta data about the model.
        self.model_name = model_name
        self.model_family = get_model_family(model_name)

        # Check if quantization is being used
        self.quantization_config = load_kwargs.get('quantization_config', None)

        # Load model.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,  # Required for RWKV models
            **load_kwargs
        )
        print(model)

        # Store device for later use
        self.device = next(model.parameters()).device

        # Load pretrained tuned lens.
        self.use_tuned_lens = use_tuned_lens
        if use_tuned_lens:
            print("Initializing pretrained tuned lens")
            self.tuned_lens = TunedLens.from_model_and_pretrained(
                model,
                cache_dir=cache_dir
            ).to(self.device)
        else:
            print("Using standard logit lens")

        # Load tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        if tokenizer.pad_token is None:
            print("No pad token found; setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize nnsight wrapper.
        self.model = LanguageModel(model, tokenizer=tokenizer)

        # Define references to internal layers and functions for logit lens.
        if self.model_family in ["llama", "olmo", "gemma", "falcon"]:
            self.layers = self.model.model.layers
            self.layer_norm = self.model.model.norm
            self.lm_head = self.model.lm_head
        elif self.model_family == "gpt":
            self.layers = self.model.transformer.h
            self.layer_norm = self.model.transformer.ln_f
            self.lm_head = self.model.lm_head
        elif self.model_family == "pythia":
            self.layers = self.model.gpt_neox.layers
            self.layer_norm = self.model.gpt_neox.final_layer_norm
            self.lm_head = self.model.embed_out
        elif self.model_family == "mamba":
            self.layers = self.model.backbone.layers
            self.layer_norm = self.model.backbone.norm_f
            self.lm_head = self.model.lm_head
        elif self.model_family == "rwkv":
            self.layers = self.model.rwkv.blocks
            self.layer_norm = self.model.rwkv.ln_out
            self.lm_head = self.model.head
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

        self.n_layers = len(self.layers)

    def apply_lens(self, hiddens):
        """
        Decode hiddens into logits and log probabilities,
        with optional affine embedding using tuned lens.

        Args:
        * hiddens: Tensor (n_tokens x n_layers x hidden_size)

        Returns:
        * logits: Tensor (n_tokens x n_layers x vocab_size)
        * logprobs: Tensor (n_tokens x n_layers x vocab_size)
        """
        if self.use_tuned_lens:
            # Cast hiddens to the appropriate dtype for quantized models
            if self.quantization_config is not None:
                target_dtype = self.lm_head.weight.dtype
                hiddens = hiddens.to(target_dtype)

            logits = torch.stack([
                torch.stack([
                    self.tuned_lens(h, i) for i, h in enumerate(tok_hiddens)
                ])
                for tok_hiddens in hiddens.to(self.device)
            ])
            logprobs = logits.log_softmax(dim=-1)
        else:
            # Use standard logit lens
            n_tokens, n_layers, hidden_size = hiddens.shape

            # Only use layer-by-layer processing for quantized models to save memory
            # For regular models, process all at once on GPU (much faster)
            if self.quantization_config is not None:
                # Memory-efficient layer-by-layer processing for quantized models
                logits_list = []
                logprobs_list = []
                target_dtype = self.lm_head.weight.dtype

                for layer_idx in range(n_layers):
                    layer_hidden = hiddens[:, layer_idx, :].to(self.device).to(target_dtype)
                    layer_logits = self.lm_head(self.layer_norm(layer_hidden))
                    layer_logprobs = layer_logits.log_softmax(dim=-1)
                    logits_list.append(layer_logits.cpu())
                    logprobs_list.append(layer_logprobs.cpu())
                    del layer_hidden, layer_logits, layer_logprobs
                    torch.cuda.empty_cache()

                logits = torch.stack(logits_list, dim=1)
                logprobs = torch.stack(logprobs_list, dim=1)
            else:
                # Fast path: process all layers at once on GPU
                hiddens_gpu = hiddens.to(self.device)  # (n_tokens, n_layers, hidden_size)

                # Apply layer norm and lm_head to all layers
                # Reshape to process all layers together
                hiddens_reshaped = hiddens_gpu.reshape(-1, hidden_size)  # (n_tokens * n_layers, hidden_size)
                logits_flat = self.lm_head(self.layer_norm(hiddens_reshaped))  # (n_tokens * n_layers, vocab_size)
                logits = logits_flat.reshape(n_tokens, n_layers, -1)  # (n_tokens, n_layers, vocab_size)
                logprobs = logits.log_softmax(dim=-1)

                # Move back to CPU to match expected behavior
                logits = logits.cpu()
                logprobs = logprobs.cpu()

        return logits, logprobs

    def logprobs_and_logit_diffs_all_layers(
        self,
        text: str
    ) -> torch.Tensor:
        """
        Main function for obtaining logits and logprobs at each layer,
        given a context stimulus.
        """
        self.model.eval()
        with torch.no_grad():
            with self.model.trace(text) as tracer:
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
                # Get "raw" hiddens and "deltas" between hiddens (i-->i+1).
                hiddens = torch.cat(hiddens_l, dim=1).save()
                hidden_deltas = (hiddens[:, 1:, :]  - hiddens[:, :-1, :]).save()

        # Move hiddens to CPU to save GPU memory during logit computation
        hiddens = hiddens.cpu()
        hidden_deltas = hidden_deltas.cpu()

        # Get logits and logprobs using logit lens or tuned lens.
        logits, logprobs = self.apply_lens(hiddens)
        del hiddens

        # Now process hidden_deltas
        logits_deltas, logprobs_deltas = self.apply_lens(hidden_deltas)
        del hidden_deltas

        return logits, logprobs, logits_deltas, logprobs_deltas

    def rank_of_token_all_layers(
        self, 
        prefix: str,
        token_ids: list[int],
        one_indexed: bool = True
    ):
        """
        Get ranks of specified tokens at each layer.
        Returns a list of N lists, where N is the length of `token_ids`.
        Each nested list has one rank per layer.
        """
        _, logprobs, _, _ = self.logprobs_and_logit_diffs_all_layers(prefix)

        # Just look at the predictions at the last position
        # (look at the logits at position i-1 for target token i).
        final_logprobs = logprobs[-1, :, :] # n_layers x vocab_size

        ranks = []
        for token_id in token_ids:
            rank_of_token = get_rank(
                final_logprobs, 
                token_id, 
                one_indexed=one_indexed
            ).tolist()
            ranks.append(rank_of_token)
        return ranks

    def conditional_score_all_layers(
        self, 
        prefix: str, 
        continuation: str, 
        sep: str = " ", 
        check_tokenization: bool = True
    ) -> dict:
        """
        Get scores of the continuation conditioned on the prefix at each layer.
        Scores are defined as a reduction of the raw token-level log probabilities.

        Args:
         * prefix: str (what you want to condition on)
         * continuation: str (what you want to compute probability of)
        """
        text = prefix + sep + continuation

        logits, logprobs, logits_deltas, _ = \
            self.logprobs_and_logit_diffs_all_layers(text)

        # Tokenize the continuation separately so we know which ones to keep.
        continuation_tokens = self.model.tokenizer(continuation)["input_ids"]

        # Remove BOS token if necessary, since it might have been
        # automatically prepended when tokenizing the continuation separately.
        if self.model.tokenizer.bos_token:
            continuation_tokens = [
                t for t in continuation_tokens 
                if t != self.model.tokenizer.bos_token_id
            ]

        # OPTIONAL: Double check that we are "reconstructing" tokens correctly.
        # Skipping this step might slightly speed up the evaluation process.
        if check_tokenization:
            # "Original" tokens.
            inputs = self.model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            text_tokens = inputs["input_ids"][0].tolist()

            # "New" tokens by combining prefix and continuation separately.
            prefix_tokens = self.model.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            new_text_tokens = prefix_tokens + continuation_tokens

            # Check if the tokens all match up.
            match = all(n == t for n, t in zip(new_text_tokens, text_tokens))

            if not match:
                # If there was a mismatch, make another attempt by adding
                # a space in front of the continuation.
                continuation_tokens = self.model.tokenizer(
                    " " + continuation, add_special_tokens=False
                )["input_ids"]
                new_text_tokens = prefix_tokens + continuation_tokens
                match = all(n == t for n, t in zip(new_text_tokens, text_tokens))
                if not match:
                    # If there is still a mismatch, give up.
                    raise ValueError("Could not isolate continuation tokens!")

        # Only keep the logits corresponding to the continuation.
        # We look at the logits at position i-1 for target token i, so we need to
        # shift logits and logprobs to the left by 1.
        n_continuation_tokens = len(continuation_tokens)
        continuation_logprobs = logprobs[-n_continuation_tokens-1:-1, :, :]
        continuation_logits = logits[-n_continuation_tokens-1:-1, :, :]
        continuation_logits_deltas = logits_deltas[-n_continuation_tokens-1:-1, :, :]

        # Compute entropy at first token of continuation for each layer.
        # FINAL SHAPE: n_layers
        first_token_of_continuation_logprobs = continuation_logprobs[0, :, :]
        all_layer_entropy = (
            # Exponentiate logprobs to get probs
            -first_token_of_continuation_logprobs.exp() * \
            first_token_of_continuation_logprobs
        ).sum(dim=1).detach().cpu().numpy()

        # Get logits and logprobs corresponding to each token in the continuation.
        # "Raw" layers: n_layers x n_continuation_tokens
        all_layer_logprobs = get_vals_of_tokens(
            continuation_logprobs, 
            continuation_tokens
        )
        all_layer_logits = get_vals_of_tokens(
            continuation_logits, 
            continuation_tokens
        )
        # Layer "deltas": (n_layers-1) x n_continuation_tokens
        all_layer_logits_deltas = get_vals_of_tokens(
            continuation_logits_deltas,
            continuation_tokens
        )

        # Perform reduction on token-level logprobs to get final scores.
        sum_logprobs = np.sum(all_layer_logprobs, axis=1)
        mean_logprobs = np.mean(all_layer_logprobs, axis=1)
        first_logprobs = all_layer_logprobs[:, 0]
        return {
            "entropy": all_layer_entropy,
            "sum": sum_logprobs, 
            "mean": mean_logprobs, 
            "first": first_logprobs, 
            "logits": all_layer_logits,
            "logits_deltas": all_layer_logits_deltas
        }
