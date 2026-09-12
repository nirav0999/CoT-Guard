import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from copy import deepcopy

import rich

from utils import (
    load_jsonl,
    save_jsonl,
)

from config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_ROLLOUTS,
    DEFAULT_TEMP,
)


@dataclass
class GenerationConfig:
    model: str
    backend: str = "vllm"
    temperature: float = DEFAULT_TEMP
    top_p: float = 0.95
    max_tokens: int = DEFAULT_MAX_TOKENS
    n: int = 1
    seed: int = 42
    gpu_memory_utilization: float = 0.95
    tp: int = 1
    bs: int = 32
    thinking: bool = True
    devices: str = "0"
    scaling_factor: float | None = None
    top_k: int = -1
    min_p: int = 0
    presence_penalty: float = 0.0


def init_vllm(config: GenerationConfig) -> tuple[Any, Any]:
    from vllm import LLM

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.devices)

    try:
        llm_kwargs = {
            "model": config.model,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tp,
        }
        if config.scaling_factor is not None:
            rich.print(f"Applying rope scaling factor: {config.scaling_factor}")
            llm_kwargs["rope_scaling"] = {
                "rope_type": "dynamic",
                "factor": config.scaling_factor,
            }
            llm_kwargs["max_model_len"] = int(DEFAULT_MAX_TOKENS * config.scaling_factor)
        model = LLM(**llm_kwargs)
    except:
        llm_kwargs = {
            "model": config.model,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tp,
        }
        if config.scaling_factor is not None:
            rich.print(f"Applying rope scaling factor: {config.scaling_factor}")
            llm_kwargs["hf_overrides"] = {
                    "rope_parameters": {
                        "rope_type": "dynamic",
                        "factor": config.scaling_factor,
                        # "rope_type": "yarn",
                        # "factor": 2.0,
                        # "original_max_position_embeddings": 32768,
                    },
                }
            llm_kwargs["max_model_len"] = int(DEFAULT_MAX_TOKENS * config.scaling_factor)
        model = LLM(**llm_kwargs)
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def format_prompts(
    messages_list: List[List[Dict[str, str]]], tokenizer: Any
) -> List[str]:
    formatted = []
    for messages in messages_list:
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = user_messages[-1]["content"]
        formatted.append(prompt)
    return formatted


def create_sampling_params(config: GenerationConfig) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        n=config.n,
        seed=config.seed,
        top_k=config.top_k,
        min_p=config.min_p,
        presence_penalty=config.presence_penalty,
    )


def cleanup_vllm(model: Any) -> None:
    import gc

    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def extract_responses(outputs: List[Any]) -> List[List[str]]:
    results = []
    for output in outputs:
        responses = [completion.text for completion in output.outputs]
        results.append(responses)
    return results


def validate_messages(data: List[Dict[str, Any]]) -> None:
    for row in data:
        assert "messages" in row, "Each row must have 'messages' key"
        assert "task_id" in row, "Each row must have 'task_id' key"
        for msg in row["messages"]:
            assert (
                "role" in msg and "content" in msg
            ), "Each message must have 'role' and 'content'"


def run_vllm_generation(
    input_path: str, output_path: str, gen_config: GenerationConfig
) -> str:
    rich.print(f"[cyan bold underline]📊 Generation Config[/cyan bold underline]")
    rich.print(f"[dim]Model: {gen_config.model}[/dim]")
    rich.print(f"[dim]Temperature: {gen_config.temperature}[/dim]")
    rich.print(f"[dim]Rollouts: {gen_config.n}[/dim]")
    rich.print(f"[dim]Devices: {gen_config.devices}[/dim]")
    rich.print(f"[dim]TP: {gen_config.tp}[/dim]")

    data = load_jsonl(input_path)
    validate_messages(data)
    rich.print(f"[yellow]{len(data)} samples loaded[/yellow]")

    model, tokenizer = init_vllm(gen_config)

    messages_list = [row["messages"] for row in data]
    formatted_prompts = format_prompts(messages_list, tokenizer)

    sampling_params = create_sampling_params(gen_config)
    outputs = model.generate(formatted_prompts, sampling_params, use_tqdm=True)
    responses = extract_responses(outputs)

    results = []
    for row, response_list in zip(data, responses):
        for rollout_idx, response in enumerate(response_list):
            task_id = row["task_id"]
            rollout_task_id = (
                task_id if gen_config.n == 1 else f"{task_id}:{rollout_idx}"
            )

            result = {
                "task_id": rollout_task_id,
                "messages": row["messages"]
                + [{"role": "assistant", "content": response}],
                "metadata": row.get("metadata", {}),
            }
            results.append(result)

    save_jsonl(results, output_path)
    cleanup_vllm(model)
    return output_path


def run_api_generation(
    input_path: str, output_path: str, gen_config: GenerationConfig
) -> str:
    from utils.batch_api_run import run_batched_inference, prepare_batch_requests, convert_batch_outputs_to_responses

    data = load_jsonl(input_path)
    validate_messages(data)

    batch_query_file, requests = prepare_batch_requests(input_path, data, gen_config)
    output_batch_path = run_batched_inference(batch_query_file, requests, output_path, gen_config)
    responses = convert_batch_outputs_to_responses(output_batch_path, data, gen_config)

    # print(len(responses), " | ", len(data) * gen_config.n, " | ", len(data), " | ", gen_config.n)
    assert len(responses) == len(data)
    results = []
    for idx, record in enumerate(data):
        task_id = record["task_id"]
        for rollout_idx in range(gen_config.n):
            rollout_task_id = (
                task_id if gen_config.n == 1 else f"{task_id}:{rollout_idx}"
            )
            result = {
                "task_id": rollout_task_id,
                "messages": deepcopy(record["messages"]) + [{
                    "role": "assistant",
                    "content": responses[idx][rollout_idx],
                }],
                "metadata": deepcopy(record.get("metadata", {})),
            }
            results.append(result)

    save_jsonl(results, output_path)
    return output_path


def generate_main(
    input_path: str,
    output_path: str,
    model: str = "Qwen/Qwen3-14B",
    backend: str = "vllm",
    temperature: float = DEFAULT_TEMP,
    top_p: float = 0.95,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n: int = DEFAULT_ROLLOUTS,
    seed: int = 42,
    gpu_memory_utilization: float = 0.95,
    tp: int = 1,
    bs: int = 32,
    devices: str = "0",
    scaling_factor: float | None = None,
) -> str:

    if isinstance(devices, tuple):
        devices = ",".join(map(str, devices))

    gen_config = GenerationConfig(
        model=model,
        backend=backend,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        tp=tp,
        bs=bs,
        devices=devices,
        scaling_factor=scaling_factor,
    )
    if backend == "vllm":
        return run_vllm_generation(input_path, output_path, gen_config)
    elif backend == "openai":
        return run_api_generation(input_path, output_path, gen_config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_main)
