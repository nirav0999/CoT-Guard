# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from shutil import copy
from typing import List

import rich
from fire import Fire

from gen.generate import GenerationConfig, run_api_generation, run_vllm_generation
from utils import (
    load_jsonl,
    save_jsonl,
)

from config import (
    DEFAULT_ATTACK_MODEL,
    DEFAULT_ATTACK_POLICY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_ROLLOUTS,
    DEFAULT_TEMP,
    GLOBAL_DIRECTORY,
    TEMP_DIRECTORY
)

def get_output_path(input_path: str, model: str, temp: float) -> str:
    filename = os.path.basename(input_path)
    d = filename.replace(".jsonl", "")
    output_filename = f"{model.split('/')[-1]}.jsonl"
    output_dir = os.path.join(GLOBAL_DIRECTORY, f"results/attack/{d}/")
    return f"{output_dir}/{output_filename}"


def count_existing_rollouts(output_path: str) -> dict:
    if not os.path.exists(output_path):
        return {}

    data = load_jsonl(output_path)
    rollout_counts = {}

    for row in data:
        task_id = row["task_id"]
        parts = task_id.split(":")
        base_id = ":".join(parts[:-1]) if len(parts) > 1 else task_id
        rollout_counts[base_id] = rollout_counts.get(base_id, 0) + 1

    return rollout_counts


def filter_samples_needing_rollouts(
    input_path: str, existing_counts: dict, num_rollouts: int
) -> List[dict]:
    from copy import deepcopy
    data = load_jsonl(input_path)
    samples_to_process = []

    for row in data:
        task_id = row["task_id"]
        existing = existing_counts.get(task_id, 0)
        needed = num_rollouts - existing

        if needed > 0:
            row_copy = deepcopy(row)
            row_copy["_needed_rollouts"] = needed
            samples_to_process.append(row_copy)

    return samples_to_process


def attack_infer_main(
    main_task: str = "bigcodebench",
    side_task: str = "iodelete",
    attack_policy: str = DEFAULT_ATTACK_POLICY,
    model: str = DEFAULT_ATTACK_MODEL,
    backend: str = "vllm",
    temp: float = DEFAULT_TEMP,
    bs: int = 128,
    num_rollouts: int = DEFAULT_ROLLOUTS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tp: int = 2,
    devices: str = "5,6",
) -> str:

    input_path = f"{GLOBAL_DIRECTORY}/datasets/attack/{main_task}.{side_task}.{attack_policy}.jsonl"

    output_path = get_output_path(input_path, model, temp)
    rich.print(f"[dim]Output path: {output_path}[/dim]")

    existing_counts = count_existing_rollouts(output_path)
    total_existing = sum(existing_counts.values())
    rich.print(
        f"[yellow]Existing rollouts: {total_existing} across {len(existing_counts)} tasks[/yellow]"
    )

    samples_to_process = filter_samples_needing_rollouts(
        input_path, existing_counts, num_rollouts
    )

    if not samples_to_process:
        rich.print(
            f"[green]All {num_rollouts} rollouts already exist for all tasks. Skipping generation.[/green]"
        )
        return output_path

    needed_per_sample = {
        s["task_id"]: s["_needed_rollouts"] for s in samples_to_process
    }
    max_needed = max(needed_per_sample.values())
    rich.print(f"[yellow]Samples needing rollouts: {len(samples_to_process)}[/yellow]")
    rich.print(f"[yellow]Max rollouts needed per sample: {max_needed}[/yellow]")

    for sample in samples_to_process:
        del sample["_needed_rollouts"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"{TEMP_DIRECTORY}/logs/single/attack_{model.split('/')[-1]}_{main_task}.{side_task}_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    temp_input_path = f"{temp_dir}/temp_input.jsonl"
    temp_output_path = f"{temp_dir}/temp_output.jsonl"
    save_jsonl(samples_to_process, temp_input_path)

    if isinstance(devices, tuple):
        devices = ",".join(map(str, devices))

    gen_config = GenerationConfig(
        model=model,
        backend=backend,
        temperature=temp,
        max_tokens=max_tokens,
        n=max_needed,
        tp=tp,
        bs=bs,
        devices=devices,
    )

    rich.print()
    if backend == "vllm":
        run_vllm_generation(temp_input_path, temp_output_path, gen_config)
    elif backend == "api":
        run_api_generation(temp_input_path, temp_output_path, gen_config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    new_results = load_jsonl(temp_output_path)

    filtered_results = []
    for row in new_results:
        task_id = row["task_id"]
        parts = task_id.split(":")
        base_id = ":".join(parts[:-1]) if len(parts) > 1 else task_id
        rollout_idx = int(parts[-1]) if len(parts) > 1 else 0
        needed = needed_per_sample.get(base_id, 0)
        if rollout_idx < needed:
            existing = existing_counts.get(base_id, 0)
            new_rollout_idx = existing + rollout_idx
            new_task_id = f"{base_id}:{new_rollout_idx}"
            row["task_id"] = new_task_id
            filtered_results.append(row)

    if os.path.exists(output_path):
        existing_data = load_jsonl(output_path)
        all_results = existing_data + filtered_results
    else:
        all_results = filtered_results

    all_task_ids = [row["task_id"] for row in all_results]
    assert len(all_task_ids) == len(
        set(all_task_ids)
    ), f"Duplicate task_ids in {output_path}: {[tid for tid in all_task_ids if all_task_ids.count(tid) > 1]}"

    save_jsonl(all_results, output_path)

    rich.print()
    rich.print(f"[green]Done! Total rollouts now: {len(all_results)}[/green]")
    rich.print(f"[green]Output: {output_path}[/green]")

    return output_path


def attack_parallel(
    main_task: str = "bigcodebench",
    side_tasks: str = "iodelete",
    attack_policies: str = "no_side_task",
    model: str = DEFAULT_ATTACK_MODEL,
    temp: float = DEFAULT_TEMP,
    devices: str = "0",
    tp: int = 1,
    num_rollouts: int = DEFAULT_ROLLOUTS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    bs: int = 128,
) -> List[str]:
    from gen.generate_parallel import parallel_generate

    if isinstance(side_tasks, str):
        side_tasks = side_tasks.split(",")

    if isinstance(attack_policies, str):
        attack_policies = attack_policies.split(",")

    input_paths = []
    output_paths = []

    for attack_policy in attack_policies:
        for side_task in side_tasks:
            input_path = f"{GLOBAL_DIRECTORY}/datasets/attack/{main_task}.{side_task}.{attack_policy}.jsonl"
            output_path = get_output_path(input_path, model, temp)
            input_paths.append(input_path)
            output_paths.append(output_path)

    for inp, out in zip(input_paths, output_paths):
        rich.print(f"[blue]→ {inp}[/blue]")
        rich.print(f"[bright_blue]← {out}[/bright_blue]")
    rich.print()

    return parallel_generate(
        input_paths=input_paths,
        output_paths=output_paths,
        model=model,
        devices=devices,
        tp=tp,
        temp=temp,
        num_rollouts=num_rollouts,
        max_tokens=max_tokens,
        bs=bs,
    )


if __name__ == "__main__":
    Fire({"single": attack_infer_main, "parallel": attack_parallel})
