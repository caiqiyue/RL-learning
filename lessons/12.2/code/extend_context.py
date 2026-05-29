"""
Extend Model Context Window

This script demonstrates how to extend a model's context window using RoPE
interpolation techniques. It covers:
1. Loading a model with modified position embeddings
2. Applying different interpolation strategies
3. Verifying the extended context works correctly
"""

import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any


def modify_model_config(
    model_path: str, new_max_pos: int, interpolation_method: str = "ntk-aware"
) -> AutoConfig:
    """
    Create a modified config for extended context.

    Args:
        model_path: Path or identifier of the base model
        new_max_pos: Target maximum position embeddings
        interpolation_method: One of 'linear', 'ntk-aware', 'yarn'

    Returns:
        Modified configuration object
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    original_max_pos = config.max_position_embeddings
    print(f"Original max_position_embeddings: {original_max_pos}")
    print(f"Target max_position_embeddings: {new_max_pos}")
    print(f"Interpolation method: {interpolation_method}")

    config.max_position_embeddings = new_max_pos

    if interpolation_method == "linear":
        config.rope_scaling = {
            "type": "linear",
            "factor": original_max_pos / new_max_pos,
        }
    elif interpolation_method == "ntk-aware":
        config.rope_scaling = {
            "type": "dynamic",
            "original_max_position_embeddings": original_max_pos,
        }
    elif interpolation_method == "yarn":
        config.rope_scaling = {
            "type": "yarn",
            "original_max_position_embeddings": original_max_pos,
            "scale": original_max_pos / new_max_pos,
        }

    return config


def load_model_with_extended_context(
    model_path: str,
    new_max_pos: int,
    interpolation_method: str = "ntk-aware",
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> AutoModelForCausalLM:
    """
    Load a model with extended context window.

    Args:
        model_path: Path or identifier of the base model
        new_max_pos: Target maximum position embeddings
        interpolation_method: One of 'linear', 'ntk-aware', 'yarn'
        device: Device to load the model on
        torch_dtype: Data type for model weights

    Returns:
        Model with extended context window
    """
    config = modify_model_config(model_path, new_max_pos, interpolation_method)

    model_kwargs = {
        "config": config,
        "trust_remote_code": True,
    }

    if device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype or torch.float16
        model_kwargs["device_map"] = None

    if torch_dtype:
        model_kwargs["torch_dtype"] = torch_dtype

    print(f"\nLoading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    print("Model loaded successfully!")

    return model


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """
    Load the tokenizer for the model.

    Args:
        model_path: Path or identifier of the model

    Returns:
        Tokenizer object
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_extended_context(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_text: str,
    max_new_tokens: int = 100,
) -> Dict[str, Any]:
    """
    Test the model with an extended context input.

    Args:
        model: The model to test
        tokenizer: The tokenizer
        test_text: Input text to test with
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with results
    """
    print(f"\n--- Testing Extended Context ---")
    print(f"Input text length: {len(test_text)} characters")

    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"Tokenized sequence length: {seq_len} tokens")
    print(f"Model max_position_embeddings: {model.config.max_position_embeddings}")

    if seq_len > model.config.max_position_embeddings:
        print(
            f"WARNING: Sequence length {seq_len} exceeds max {model.config.max_position_embeddings}"
        )
        return {"error": "Sequence too long"}

    if "cuda" in str(model.device):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "input_length": len(test_text),
        "tokenized_length": seq_len,
        "generated_length": len(generated_text),
        "generated_text": generated_text,
    }


def create_long_context_prompt(
    num_paragraphs: int = 10, paragraphs_per_topic: int = 2
) -> str:
    """
    Create a synthetic long-context prompt for testing.

    Args:
        num_paragraphs: Number of paragraphs to generate
        paragraphs_per_topic: Paragraphs per topic cluster

    Returns:
        Long context prompt string
    """
    topics = [
        "artificial intelligence",
        "climate change",
        "space exploration",
        "quantum computing",
        "biotechnology",
        "renewable energy",
        "neural networks",
        "sustainable development",
        "medical research",
    ]

    sentences = [
        "This is an important development in the field.",
        "Researchers have made significant progress recently.",
        "The implications of this are far-reaching.",
        "Experts believe this could transform how we approach the topic.",
        "Further studies are needed to fully understand the consequences.",
        "Initial results show promising outcomes.",
        "The methodology employed represents a novel approach.",
        "Collaboration between institutions has accelerated progress.",
    ]

    paragraphs = []
    for i in range(num_paragraphs):
        topic_idx = (i // paragraphs_per_topic) % len(topics)
        topic = topics[topic_idx]
        selected_sentences = sentences[
            (i * 2) % len(sentences) : ((i * 2) % len(sentences)) + 4
        ]
        paragraph = f"Regarding {topic}: " + " ".join(selected_sentences)
        paragraphs.append(paragraph)

    prompt = "Here is a comprehensive overview:\n\n" + "\n\n".join(paragraphs)
    prompt += "\n\nBased on this information, please provide a detailed summary of the main themes."

    return prompt


def estimate_kv_cache_memory(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_param: int = 2,
) -> float:
    """
    Estimate KV cache memory usage.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        seq_len: Sequence length
        bytes_per_param: Bytes per parameter (2 for fp16, 4 for fp32)

    Returns:
        Memory in GB
    """
    kv_params = 2 * num_layers * num_heads * head_dim * seq_len * bytes_per_param
    kv_gb = kv_params / (1024**3)
    return kv_gb


def main():
    parser = argparse.ArgumentParser(description="Extend model context window")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path"
    )
    parser.add_argument(
        "--new_max_pos", type=int, default=32768, help="New max position embeddings"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ntk-aware",
        choices=["linear", "ntk-aware", "yarn"],
        help="Interpolation method",
    )
    parser.add_argument(
        "--test_length", type=int, default=4096, help="Test sequence length in tokens"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--estimate_memory", action="store_true", help="Estimate KV cache memory"
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Context Window Extension Script")
    print("=" * 60)

    if args.estimate_memory:
        print("\n--- KV Cache Memory Estimation ---")
        num_layers = 32
        num_heads = 32
        head_dim = 128

        for seq_len in [4096, 8192, 16384, 32768, 65536]:
            mem_gb = estimate_kv_cache_memory(num_layers, num_heads, head_dim, seq_len)
            print(f"Seq len {seq_len:6d}: {mem_gb:.2f} GB (fp16)")

    config = modify_model_config(args.model, args.new_max_pos, args.method)

    print("\n" + "=" * 60)
    print("Note: This script demonstrates the configuration and")
    print("estimation steps. Actual model loading requires:")
    print("1. A valid HuggingFace token for gated models")
    print("2. Sufficient GPU memory for the model")
    print("3. The model to support RoPE scaling (most modern LLMs do)")
    print("=" * 60)

    long_prompt = create_long_context_prompt(num_paragraphs=15)
    print(f"\nGenerated test prompt ({len(long_prompt)} chars)")


if __name__ == "__main__":
    main()
