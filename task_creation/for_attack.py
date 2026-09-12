# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List

import rich
from fire import Fire

from config import GLOBAL_DIRECTORY
from constants import DEFAULT_ATTACK_POLICIES, DEFAULT_SIDE_TASKS
from src.cot_red_handed.tasks.tasks import TaskManager
from utils import save_jsonl, validate_attack_dataset


def extract_samples_from_task(task, debug: bool = False) -> List[Dict[str, Any]]:
    samples = []
    for sample in task.dataset:
        output = {
            "task_id": sample.id,
            "messages": [{"role": "user", "content": sample.input}],
            "metadata": {
                "main_task_id": (
                    sample.id.split("_")[0] if "_" in sample.id else sample.id
                ),
                "main_test": sample.target,
                "side_test": sample.metadata.get("side_test_case", ""),
                "canonical_main_sol": sample.metadata["canonical_solution"],
                "side_task_name": sample.metadata["side_task_name"],
                "side_task_description": sample.metadata["side_task_description"],
                "attack_policy": sample.metadata["attack_policy"],
                "side_and_attack_policy": sample.metadata["side_and_attack_policy"],
                "input_types": sample.metadata["input_types"],
                "output_types": sample.metadata["output_types"],
                "code_prompt": sample.metadata["code"],
                "original_prompt": sample.metadata["prompt"],
            },
        }
        samples.append(output)

    return samples


def process_single_side_task(
    task_manager: TaskManager,
    main_task: str,
    side_task: str,
    attack_policy: str,
    use_pairings: bool,
    debug: bool,
    id_set: str,
) -> str:

    task = task_manager.generate_task(
        main_task_name=main_task,
        side_task_name=side_task,
        attack_policies=[attack_policy],
        id_set=id_set,
        task_type="cot_eval",
        use_pairings=use_pairings,
    )

    samples = extract_samples_from_task(task, debug=debug)
    rich.print(f"[yellow]Total samples for {side_task}: {len(samples)}[/yellow]")

    output_dir = os.path.join(GLOBAL_DIRECTORY, "datasets/attack/")

    output_filename = f"{main_task}.{side_task}.{attack_policy}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    save_jsonl(samples, output_path)

    return output_path


def for_attack_main(
    main_task: str = "bigcodebench",
    side_tasks: str = ",".join(DEFAULT_SIDE_TASKS),
    attack_policies: str = ",".join(DEFAULT_ATTACK_POLICIES),
    use_pairings: bool = True,
    debug: bool = False,
    id_set: str = "TEST",
) -> List[str]:

    if isinstance(side_tasks, str):
        side_tasks = side_tasks.split(",")
    if isinstance(attack_policies, str):
        attack_policies = attack_policies.split(",")
    side_task_list = [s.strip() for s in side_tasks if s.strip()]
    attack_policy_list = [p.strip() for p in attack_policies if p.strip()]
    task_manager = TaskManager()

    output_paths = []
    for side_task in side_task_list:
        for attack_policy in attack_policy_list:
            output_path = process_single_side_task(
                task_manager=task_manager,
                main_task=main_task,
                side_task=side_task,
                attack_policy=attack_policy,
                use_pairings=use_pairings,
                debug=debug,
                id_set=id_set,
            )
            output_paths.append(output_path)

    rich.print(f"[green]Done! Generated {len(output_paths)} files:[/green]")
    for path in output_paths:
        rich.print(f"  [dim]{path}[/dim]")
        validate_attack_dataset(path)

    return output_paths


if __name__ == "__main__":
    Fire(for_attack_main)
