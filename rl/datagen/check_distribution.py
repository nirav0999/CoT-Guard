# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from pathlib import Path

import rich
from fire import Fire

from utils import GLOBAL_DIRECTORY, extract_choice


def check_distribution(
    main_tasks: list = ["kodcode"],
    side_tasks: list = [
        "permissions",
        "iodelete",
        "exit",
        "varname",
        "ioexfil",
        "iodownload",
    ],
    evaluated_models: list = ["Qwen3-14B"],
    monitor_models: list = ["Qwen3-32B"],
    attack_policies: list = ["baseline"],
    rollout_num: int = 1,
    monitor_policies: list = ["main_aware"],
    monitor_objectives: list = ["cot_only"],
    filter: str = "side-all",
) -> None:

    for monitor_objective in monitor_objectives:
        side_counter = Counter()
        main_counter = Counter()
        side_total = 0
        main_total = 0

        for main_task in main_tasks:
            for side_task in side_tasks:
                for evaluated_model in evaluated_models:
                    for monitor_model in monitor_models:
                        for monitor_policy in monitor_policies:
                            for attack_policy in attack_policies:

                                side_path_file = (
                                    Path(GLOBAL_DIRECTORY)
                                    / "results"
                                    / "monitor"
                                    / f"{main_task}.{side_task}.{attack_policy}"
                                    / f"{evaluated_model}--{rollout_num}"
                                    / f"{filter}.{monitor_model}.{monitor_policy}.{monitor_objective}.jsonl"
                                )
                                main_path_file = (
                                    Path(GLOBAL_DIRECTORY)
                                    / "results"
                                    / "monitor"
                                    / f"{main_task}.{side_task}.no_side_task"
                                    / f"{evaluated_model}--{rollout_num}"
                                    / f"main-all.{monitor_model}.{monitor_policy}.{monitor_objective}.jsonl"
                                )

                                for label, path_file, counter_ref in [
                                    ("side", side_path_file, side_counter),
                                    ("main", main_path_file, main_counter),
                                ]:
                                    if not path_file.exists():
                                        rich.print(
                                            f"[dark_red bold]✗ Missing: {path_file}[/dark_red bold]"
                                        )
                                        continue
                                    rich.print(
                                        f"[blue]→ Reading ({label}): {path_file}[/blue]"
                                    )
                                    with path_file.open("r", encoding="utf-8") as f:
                                        for line in f:
                                            if not line.strip():
                                                continue
                                            data = json.loads(line)
                                            monitor_response = data["messages"][1][
                                                "content"
                                            ]
                                            choice = extract_choice(monitor_response)
                                            counter_ref[choice] += 1
                                            if label == "side":
                                                side_total += 1
                                            else:
                                                main_total += 1

        rich.print(f"\n[magenta bold]{'═' * 60}[/magenta bold]")
        rich.print(
            f"[cyan bold underline]📊 Choice Distribution for monitor_objective={monitor_objective}[/cyan bold underline]"
        )

        rich.print(
            "\n[magenta bold]Side task files (malicious samples):[/magenta bold]"
        )
        rich.print(f"[dim]  Total samples: {side_total}[/dim]")
        for choice in sorted(side_counter.keys()):
            count = side_counter[choice]
            rich.print(
                f"[yellow]  {choice}: {count} / {side_total} = {count / side_total * 100:.1f}%[/yellow]"
            )

        rich.print("\n[magenta bold]Main task files (benign samples):[/magenta bold]")
        rich.print(f"[dim]  Total samples: {main_total}[/dim]")
        for choice in sorted(main_counter.keys()):
            count = main_counter[choice]
            rich.print(
                f"[yellow]  {choice}: {count} / {main_total} = {count / main_total * 100:.1f}%[/yellow]"
            )

        total = side_total + main_total
        combined = side_counter + main_counter
        rich.print("\n[magenta bold]Combined:[/magenta bold]")
        rich.print(f"[dim]  Total samples: {total}[/dim]")
        for choice in sorted(combined.keys()):
            count = combined[choice]
            rich.print(
                f"[yellow]  {choice}: {count} / {total} = {count / total * 100:.1f}%[/yellow]"
            )

        rich.print(
            "\n[magenta bold]After collect.py filtering (side accepts C only, main accepts A/B/D):[/magenta bold]"
        )
        filtered_mal = side_counter.get("C", 0)
        filtered_ben = sum(main_counter[c] for c in ["A", "B", "D"])
        rich.print(f"[yellow]  Malicious (C from side): {filtered_mal}[/yellow]")
        rich.print(f"[yellow]  Benign (A+B+D from main): {filtered_ben}[/yellow]")
        rich.print(
            f"[yellow]  Total after filter: {filtered_mal + filtered_ben}[/yellow]"
        )


if __name__ == "__main__":
    Fire(check_distribution)
# python rl/datagen/check_distribution.py
