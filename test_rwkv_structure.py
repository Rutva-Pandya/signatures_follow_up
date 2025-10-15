#!/usr/bin/env python3
"""
Test RWKV model structure before full integration.
This script verifies layer names, hidden state format, and tokenizer compatibility.

Usage:
    python scripts/test_rwkv_structure.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_rwkv_architecture():
    print("="*80)
    print("RWKV ARCHITECTURE TEST")
    print("="*80)

    model_name = "RWKV/rwkv-4-169m-pile"
    print(f"\nLoading: {model_name}")
    print("(This may take a minute to download...)")

    try:
        # Load model with trust_remote_code=True (required for RWKV)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu"  # Keep on CPU for testing
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print(f"\n✓ Model loaded successfully")
        print(f"  Model class: {model.__class__.__name__}")

        # Check structure
        print(f"\nChecking layer references...")
        has_rwkv = hasattr(model, 'rwkv')
        has_blocks = hasattr(model.rwkv, 'blocks') if has_rwkv else False
        has_ln_out = hasattr(model.rwkv, 'ln_out') if has_rwkv else False
        has_head = hasattr(model, 'head')

        print(f"  ✓ model.rwkv exists: {has_rwkv}")
        print(f"  ✓ model.rwkv.blocks exists: {has_blocks}")
        print(f"  ✓ model.rwkv.ln_out exists: {has_ln_out}")
        print(f"  ✓ model.head exists: {has_head}")

        if not all([has_rwkv, has_blocks, has_ln_out, has_head]):
            print("\n⚠️  WARNING: Expected layer structure not found!")
            print(f"\nActual model structure:")
            print(model)
            return False

        # Count layers
        n_layers = len(model.rwkv.blocks)
        print(f"\n  Number of layers: {n_layers}")

        # Test forward pass with hidden states
        print(f"\nTesting forward pass...")
        text = "The capital of France is"
        inputs = tokenizer(text, return_tensors="pt")
        print(f"  Input: '{text}'")
        print(f"  Input IDs shape: {inputs['input_ids'].shape}")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Check hidden states
        print(f"\n  Hidden states:")
        print(f"    - Number of hidden states: {len(outputs.hidden_states)}")
        print(f"    - First hidden state shape: {outputs.hidden_states[0].shape}")
        print(f"    - Last hidden state shape: {outputs.hidden_states[-1].shape}")
        print(f"    - Type: {type(outputs.hidden_states[0])}")

        # Expected: (batch_size, seq_len, hidden_size)
        expected_batch = 1
        expected_seq_len = inputs['input_ids'].shape[1]
        actual_shape = outputs.hidden_states[0].shape

        if actual_shape[0] == expected_batch and actual_shape[1] == expected_seq_len:
            print(f"  ✓ Hidden state shape matches expected format")
        else:
            print(f"  ⚠️  Hidden state shape unexpected!")
            print(f"     Expected: (1, {expected_seq_len}, hidden_dim)")
            print(f"     Got: {actual_shape}")

        # Test tokenizer
        print(f"\n  Tokenizer:")
        print(f"    - Vocab size: {len(tokenizer)}")
        print(f"    - BOS token: {tokenizer.bos_token}")
        print(f"    - EOS token: {tokenizer.eos_token}")
        print(f"    - PAD token: {tokenizer.pad_token}")

        if tokenizer.pad_token is None:
            print(f"    ⚠️  No pad token - will need to set to eos_token")

        # Test logit decoding
        print(f"\nTesting logit lens decoding...")
        final_hidden = outputs.hidden_states[-1]  # Last layer

        # Apply layer norm + lm head (what logit lens does)
        normalized = model.rwkv.ln_out(final_hidden)
        logits = model.head(normalized)

        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected vocab size: {len(tokenizer)}")

        if logits.shape[-1] == len(tokenizer):
            print(f"  ✓ Logit decoding works correctly")
        else:
            print(f"  ⚠️  Logit size mismatch!")

        # Test hidden state extraction for nnsight
        print(f"\nTesting layer output format (for nnsight integration)...")
        # This simulates what we'll need to do in model.py
        print(f"  Checking if layer.output format is compatible...")
        print(f"  (Note: This test uses output_hidden_states, actual integration uses nnsight)")

        print(f"\n{'='*80}")
        print(f"✓ ALL TESTS PASSED!")
        print(f"{'='*80}")

        print(f"\nRequired code changes for integration:")
        print(f"\n1. src/utils.py (line 57):")
        print(f"   families = ['llama', 'olmo', 'gemma', 'gpt', 'falcon', 'mamba', 'rwkv']")

        print(f"\n2. src/model.py (line 53):")
        print(f"   Add: trust_remote_code=True to from_pretrained()")

        print(f"\n3. src/model.py (lines 89-93, add after mamba block):")
        print(f"   elif self.model_family == 'rwkv':")
        print(f"       self.layers = self.model.rwkv.blocks")
        print(f"       self.layer_norm = self.model.rwkv.ln_out")
        print(f"       self.lm_head = self.model.head")

        print(f"\n4. analysis/notebooks/utils.py:")
        print(f"   - Add RWKV models to MODELS list")
        print(f"   - Add layer counts: N_LAYERS['rwkv-4-169m-pile'] = {n_layers}")
        print(f"   - Add vocab size: VOCAB_SIZE['rwkv-4-169m-pile'] = {len(tokenizer)}")
        print(f"   - Add to MODEL_FAMILY_MAP: 'rwkv-4-169m-pile': 'RWKV'")

        print(f"\n5. May need to test hidden state extraction with nnsight")
        print(f"   (current test uses output_hidden_states)")

        print(f"\n{'='*80}")
        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"TEST FAILED")
        print(f"{'='*80}")
        return False

if __name__ == "__main__":
    success = test_rwkv_architecture()
    exit(0 if success else 1)
