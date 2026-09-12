import rich
from vllm import SamplingParams

from gen.generate import GenerationConfig, init_vllm
from config import DEFAULT_MAX_TOKENS


def test_default_config(model_name: str):
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Scaling: None (default)")
    print("=" * 60)

    config = GenerationConfig(
        model=model_name,
        gpu_memory_utilization=0.9,
        tp=1,
        devices=None,
        scaling_factor=None,
    )

    model, tokenizer = init_vllm(config)

    actual_max_len = model.llm_engine.model_config.max_model_len
    model_config_max = model.llm_engine.model_config.hf_config.max_position_embeddings

    rich.print(f"Model's max_position_embeddings: {model_config_max}")
    rich.print(f"Actual max_model_len:            {actual_max_len}")
    assert (
        actual_max_len == model_config_max
    ), f"Expected default: {model_config_max}, got: {actual_max_len}"

    rope_scaling = model.llm_engine.model_config.hf_config.rope_scaling
    rich.print(f"Rope scaling config: {rope_scaling}")

    rich.print("PASS")


def test_rope_scaling(model_name: str, scaling_factor: float):
    rich.print(f"\n{'='*60}")
    rich.print(f"Model: {model_name} | Scaling: {scaling_factor}")
    rich.print(f"{'='*60}")

    config = GenerationConfig(
        model=model_name,
        gpu_memory_utilization=0.9,
        tp=1,
        devices=None,
        scaling_factor=scaling_factor,
    )

    model, tokenizer = init_vllm(config)

    actual_max_len = model.llm_engine.model_config.max_model_len
    expected_max_len = int(DEFAULT_MAX_TOKENS * scaling_factor)

    rich.print(f"Expected max_model_len: {expected_max_len}")
    rich.print(f"Actual max_model_len:   {actual_max_len}")
    assert (
        actual_max_len == expected_max_len
    ), f"Mismatch: {actual_max_len} != {expected_max_len}"

    test_token_count = expected_max_len - 100
    dummy_text = "hello " * (test_token_count // 2)
    rich.print(f"Generating with prompt of ~{len(dummy_text)//4} tokens...")
    tokens = tokenizer.encode(dummy_text)[:test_token_count]
    long_prompt = tokenizer.decode(tokens)

    rich.print(f"Test prompt length: {len(tokens)} tokens")

    output = model.generate([long_prompt], SamplingParams(max_tokens=10))
    generated = output[0].outputs[0].text

    rich.print(f"Generation succeeded: {repr(generated[:50])}")
    rich.print("PASS")


def main():
    models = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    scaling_factors = [2.0]

    for model_name in models:
        test_default_config(model_name)
        for factor in scaling_factors:
            test_rope_scaling(model_name, factor)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
