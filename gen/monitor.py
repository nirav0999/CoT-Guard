# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from typing import Dict, List

import rich
from fire import Fire

from config import (
    DEFAULT_ATTACK_MODEL,
    DEFAULT_ATTACK_POLICY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    GLOBAL_DIRECTORY,
    TEMP_DIRECTORY,
)
from constants import FILTER_MODES
from gen.generate import GenerationConfig, run_api_generation, run_vllm_generation
from utils import (
    load_jsonl,
    save_jsonl,
)

MONITOR_TYPES = ["cot_only", "action_only", "cot_action"]


def get_monitor_input_paths(
    main_task: str,
    side_task: str,
    attack_policy: str,
    attack_model: str,
    attack_temp: float,
    attack_rollouts: int,
    filter_mode: str,
    monitor_policy: str,
    monitor_types: List[str],
    prompt_variant: str = "standard",
) -> Dict[str, str]:
    input_dir = os.path.join(
        GLOBAL_DIRECTORY,
        "datasets",
        "monitor",
        f"{main_task}.{side_task}.{attack_policy}/{attack_model.split('/')[-1]}--{attack_rollouts}/",
    )
    base = f"{filter_mode}.{monitor_policy}"
    if prompt_variant != "standard":
        base = f"{base}.{prompt_variant}"
    return {mt: os.path.join(input_dir, f"{base}.{mt}.jsonl") for mt in monitor_types}


def get_output_path_base(
    main_task: str,
    side_task: str,
    attack_policy: str,
    attack_model: str,
    attack_temp: float,
    attack_rollouts: int,
    filter_mode: str,
    monitor_policy: str,
    monitor_model: str,
    monitor_temp: float,
    model_id: str = "",
    prompt_variant: str = "standard",
) -> str:
    base = f"{attack_model.split('/')[-1]}--{attack_rollouts}"
    output_dir = os.path.join(
        GLOBAL_DIRECTORY,
        "results",
        "monitor",
        f"{main_task}.{side_task}.{attack_policy}",
        base,
    )
    monitor_model_short = model_id if model_id else monitor_model.split("/")[-1]
    base = f"{filter_mode}.{monitor_model_short}.{monitor_policy}"
    if prompt_variant != "standard":
        base = f"{base}.{prompt_variant}"
    return os.path.join(output_dir, base)


def count_existing_rollouts(output_path: str) -> Dict[str, int]:
    if not os.path.exists(output_path):
        return {}
    data = load_jsonl(output_path)
    counts: Dict[str, int] = {}
    for row in data:
        task_id = row["task_id"]
        parts = task_id.split(":")
        base_id = ":".join(parts[:-1])
        counts[base_id] = counts[base_id] + 1 if base_id in counts else 1
    return counts


def merge_inputs_with_resume(
    input_paths: Dict[str, str],
    output_path_base: str,
    num_rollouts: int,
) -> tuple[List[dict], Dict[str, int], int]:
    merged = []
    existing_counts: Dict[str, int] = {}
    max_needed = 0

    for monitor_type, input_path in input_paths.items():
        output_path = f"{output_path_base}.{monitor_type}.jsonl"
        type_existing = count_existing_rollouts(output_path)

        data = load_jsonl(input_path)
        type_need = 0
        for row in data:
            task_id = row["task_id"]
            existing = type_existing[task_id] if task_id in type_existing else 0
            needed = num_rollouts - existing

            if needed > 0:
                merged.append(row)
                existing_counts[task_id] = existing
                max_needed = max(max_needed, needed)
                type_need += 1

        total_existing = sum(type_existing.values())
        if type_need == 0:
            rich.print(
                f"[green]✓ {monitor_type}: complete ({total_existing} rollouts)[/green]"
            )
        else:
            rich.print(
                f"[yellow]{monitor_type}: {type_need}/{len(data)} need rollouts (existing: {total_existing})[/yellow]"
            )

    return merged, existing_counts, max_needed


def split_and_merge_outputs(
    results: List[dict],
    output_path_base: str,
    existing_counts: Dict[str, int],
    monitor_types: List[str],
) -> None:
    by_type: Dict[str, List[dict]] = {mt: [] for mt in monitor_types}

    for row in results:
        task_id = row["task_id"]
        parts = task_id.split(":")
        if parts[-1] in monitor_types:
            base_id = task_id
            rollout_idx = 0
            monitor_type = parts[-1]
        else:
            base_id = ":".join(parts[:-1])
            rollout_idx = int(parts[-1])
            monitor_type = parts[-2]
        assert (
            monitor_type in monitor_types
        ), f"Unknown monitor_type in task_id: {task_id}"

        existing = existing_counts[base_id] if base_id in existing_counts else 0
        new_idx = existing + rollout_idx
        row["task_id"] = f"{base_id}:{new_idx}"
        by_type[monitor_type].append(row)

    for monitor_type, new_rows in by_type.items():
        output_path = f"{output_path_base}.{monitor_type}.jsonl"

        if os.path.exists(output_path):
            existing_rows = load_jsonl(output_path)
            all_rows = existing_rows + new_rows
        else:
            all_rows = new_rows

        all_task_ids = [row["task_id"] for row in all_rows]
        assert len(all_task_ids) == len(
            set(all_task_ids)
        ), f"Duplicate task_ids in {output_path}: {[tid for tid in all_task_ids if all_task_ids.count(tid) > 1]}"

        save_jsonl(all_rows, output_path)
        rich.print(
            f"[green]✓ {monitor_type}: {len(all_rows)} total → {output_path}[/green]"
        )


def monitor_infer_main(
    main_task: str = "bigcodebench",
    side_task: str = "iodelete",
    attack_policy: str = "baseline",
    attack_model: str = DEFAULT_ATTACK_MODEL,
    attack_temp: float = DEFAULT_TEMP,
    attack_rollouts: int = 5,
    filter_mode: str = "side-all",
    monitor_policy: str = "main_aware",
    prompt_variant: str = "standard",
    monitor_model: str = "Qwen/Qwen3-14B",
    monitor_temp: float = DEFAULT_TEMP,
    monitor_type: str = "all",
    backend: str = "vllm",
    devices: str = "0",
    tp: int = 1,
    bs: int = 128,
    num_rollouts: int = 1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    scaling_factor: float = 2.0,  # For monitor models, we scale context length
    model_id: str = "",
) -> str:
    assert filter_mode in FILTER_MODES, f"filter_mode must be one of {FILTER_MODES}"

    if monitor_type == "all":
        monitor_types = MONITOR_TYPES
    else:
        assert (
            monitor_type in MONITOR_TYPES
        ), f"monitor_type must be 'all' or one of {MONITOR_TYPES}"
        monitor_types = [monitor_type]

    if attack_policy == "no_side_task":
        final_filter_mode = "main-all"
    else:
        final_filter_mode = filter_mode

    input_paths = get_monitor_input_paths(
        main_task,
        side_task,
        attack_policy,
        attack_model,
        attack_temp,
        attack_rollouts,
        final_filter_mode,
        monitor_policy,
        monitor_types,
        prompt_variant,
    )
    output_path_base = get_output_path_base(
        main_task,
        side_task,
        attack_policy,
        attack_model,
        attack_temp,
        attack_rollouts,
        final_filter_mode,
        monitor_policy,
        monitor_model,
        monitor_temp,
        model_id=model_id,
        prompt_variant=prompt_variant,
    )

    rich.print("[cyan bold underline]📊 Monitor Inference[/cyan bold underline]")
    for mt, p in input_paths.items():
        rich.print(f"[blue]→ {mt}: {p}[/blue]")
    rich.print(f"[bright_blue]← Output base: {output_path_base}[/bright_blue]")

    for path in input_paths.values():
        assert os.path.exists(path), f"Missing: {path}"

    rich.print(
        "\n[cyan bold underline]📊 Checking Existing & Merging[/cyan bold underline]"
    )
    merged, existing_counts, max_needed = merge_inputs_with_resume(
        input_paths, output_path_base, num_rollouts
    )

    if not merged:
        rich.print("\n[green]✓ All rollouts complete. Nothing to generate.[/green]")
        return output_path_base

    rich.print(
        f"\n[yellow]Total to process: {len(merged)} samples, max {max_needed} rollouts each[/yellow]"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"{TEMP_DIRECTORY}/logs/single/monitor_{monitor_model.split('/')[-1]}_{main_task}.{side_task}_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    temp_input = f"{temp_dir}/temp_input.jsonl"
    temp_output = f"{temp_dir}/temp_output.jsonl"
    save_jsonl(merged, temp_input)

    if isinstance(devices, tuple):
        devices = ",".join(map(str, devices))

    gen_config = GenerationConfig(
        model=monitor_model,
        backend=backend,
        temperature=monitor_temp,
        max_tokens=max_tokens,
        n=max_needed,
        tp=tp,
        bs=bs,
        devices=devices,
        scaling_factor=scaling_factor,
    )

    # if (
    #     monitor_model == "Qwen3-4B"
    # ):  # Specifically for Qwen3-4B, we found that presence penalty helps reduce repetition and improves quality
    #     gen_config.presence_penalty = 1.5

    rich.print("\n[cyan bold underline]📊 Running Generation[/cyan bold underline]")
    if backend == "vllm":
        run_vllm_generation(temp_input, temp_output, gen_config)
    elif backend == "api":
        run_api_generation(temp_input, temp_output, gen_config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    rich.print("\n[cyan bold underline]📊 Merging Results[/cyan bold underline]")
    results = load_jsonl(temp_output)
    split_and_merge_outputs(results, output_path_base, existing_counts, monitor_types)

    return output_path_base


def monitor_parallel(
    main_task: str = "bigcodebench",
    side_tasks: str = "iodelete",
    attack_policies: str = DEFAULT_ATTACK_POLICY,
    attack_model: str = DEFAULT_ATTACK_MODEL,
    attack_temp: float = DEFAULT_TEMP,
    attack_rollouts: int = 5,
    filter_mode: str = "side-all",
    monitor_policy: str = "main_aware",
    prompt_variant: str = "standard",
    monitor_model: str = "Qwen/Qwen3-8B",
    monitor_temp: float = DEFAULT_TEMP,
    monitor_type: str = "all",
    devices: str = "0,1,2,3,4,5,6,7",
    tp: int = 1,
    num_rollouts: int = 1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    bs: int = 128,
    scaling_factor: float = 2.0,  # For monitor models, we scale context length
    balance_by_length: bool = True,
    model_id: str = "",
) -> List[str]:
    from gen.generate_parallel import parallel_generate

    assert filter_mode in FILTER_MODES, f"filter_mode must be one of {FILTER_MODES}"

    if monitor_type == "all":
        monitor_types = MONITOR_TYPES
    else:
        assert (
            monitor_type in MONITOR_TYPES
        ), f"monitor_type must be 'all' or one of {MONITOR_TYPES}"
        monitor_types = [monitor_type]

    if isinstance(side_tasks, str):
        side_tasks_list = side_tasks.split(",")
    else:
        side_tasks_list = side_tasks

    if isinstance(attack_policies, str):
        attack_policies = attack_policies.split(",")
    else:
        attack_policies = attack_policies

    input_paths = []
    output_paths = []

    for attack_policy in attack_policies:
        for side_task in side_tasks_list:

            if attack_policy == "no_side_task":
                final_filter_mode = "main-all"
            else:
                final_filter_mode = filter_mode

            type_input_paths = get_monitor_input_paths(
                main_task,
                side_task,
                attack_policy,
                attack_model,
                attack_temp,
                attack_rollouts,
                final_filter_mode,
                monitor_policy,
                monitor_types,
                prompt_variant,
            )
            output_path_base = get_output_path_base(
                main_task,
                side_task,
                attack_policy,
                attack_model,
                attack_temp,
                attack_rollouts,
                final_filter_mode,
                monitor_policy,
                monitor_model,
                monitor_temp,
                model_id=model_id,
                prompt_variant=prompt_variant,
            )

            for mt in monitor_types:
                input_paths.append(type_input_paths[mt])
                output_paths.append(f"{output_path_base}.{mt}.jsonl")

    for inp, out in zip(input_paths, output_paths, strict=False):
        rich.print(f"[blue]→ {inp}[/blue]")
        rich.print(f"[bright_blue]← {out}[/bright_blue]")
    rich.print()

    if isinstance(devices, tuple):
        devices = ",".join(map(str, devices))

    return parallel_generate(
        input_paths=input_paths,
        output_paths=output_paths,
        model=monitor_model,
        devices=devices,
        tp=tp,
        temp=monitor_temp,
        num_rollouts=num_rollouts,
        max_tokens=max_tokens,
        bs=bs,
        scaling_factor=scaling_factor,
        balance_by_length=balance_by_length,
    )


if __name__ == "__main__":
    Fire({"single": monitor_infer_main, "parallel": monitor_parallel})
