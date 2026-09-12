# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import sys

import rich

from bigcodebench.bcb_utils import *
from src.cot_red_handed.tasks.tasks import TaskManager


def generate_tasks(args):
    task_manager = TaskManager()

    side_tasks = [task.strip() for task in args.side_tasks.split(",") if task.strip()]
    if args.sharding == 1:
        print("Side_tasks: ", side_tasks, " | Len(Side_tasks): ", len(side_tasks))

    if args.attack_policy:
        attack_policies = [
            policy.strip() for policy in args.attack_policy.split(",") if policy.strip()
        ]
    else:
        attack_policies = ["baseline"]

    task_list = []
    for side_task in side_tasks:
        task = task_manager.generate_task(
            main_task_name=args.dataset,
            side_task_name=side_task,
            attack_policies=attack_policies,
            id_set=args.bcb_id_set,
            use_pairings=args.bcb_use_pairings,
        )
        task_list.append(task)

    if args.sharding == 1:
        print(f"✓ Generated {len(task_list)} tasks")

    prompts, targets, metadata = extract_prompts_from_tasks(task_list)
    print("Prompts sample: ", prompts[:2])
    rich.print("[green]✓ Generated prompts, targets, and metadata from tasks[/green]")
    rich.print(f"[blue]Sample prompt:[/blue] {prompts[0]}")
    rich.print(f"[blue]Sample target:[/blue] {targets[0]}")
    rich.print(f"[blue]Sample metadata:[/blue] {metadata[0]}")

    rich.print(f"[yellow] Total prompts generated: {len(prompts)} [/yellow]")
    rich.print(f"[yellow] Total targets generated: {len(targets)} [/yellow]")
    rich.print(f"[yellow] Total metadata entries generated: {len(metadata)} [/yellow]")

    if "phi-4-reasoning" in args.model_name:
        # Hardcode for phi-4-reasoning
        prompts = [
            f"{prompt}\n**Please enclose the code within ```python and ``` when you provide the final solution.**"
            for prompt in prompts
        ]

    return prompts, targets, metadata
