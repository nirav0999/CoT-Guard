# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import sys
import rich

from config import GLOBAL_DIRECTORY
from utils import extract_choice, load_jsonl

INPUT_PATH = f"{GLOBAL_DIRECTORY}/results/monitor/bigcodebench.iodelete.baseline/Qwen3-14B--0.6--1.side.main_aware/Qwen3-14B--0.6.cot_only.jsonl"


def get_assistant_message(sample: dict) -> str:
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            return msg["content"]
    raise ValueError(f"No assistant message found in sample {sample['task_id']}")


def test_extract_choice_on_monitor_outputs():
    data = load_jsonl(INPUT_PATH)
    rich.print(
        f"[cyan bold underline]📊 Testing extract_choice on {INPUT_PATH}[/cyan bold underline]"
    )
    rich.print(f"[dim]Total samples: {len(data)}[/dim]")
    rich.print()

    extracted_count = 0
    failed_samples = []

    for sample in data:
        task_id = sample["task_id"]
        assistant_content = get_assistant_message(sample)
        choice = extract_choice(assistant_content)

        if choice != "None":
            extracted_count += 1
        else:
            failed_samples.append((task_id, assistant_content))

    rich.print(
        f"[yellow]{extracted_count} / {len(data)} = {extracted_count / len(data) * 100:.1f}% samples have valid extracted choice[/yellow]"
    )
    rich.print()

    if failed_samples:
        rich.print(
            f"[orange3]⚠ {len(failed_samples)} samples failed extraction:[/orange3]"
        )
        rich.print()
        for task_id, content in failed_samples:
            rich.print(f"[magenta bold]{'═' * 60}[/magenta bold]")
            rich.print(f"[cyan bold underline]Task ID: {task_id}[/cyan bold underline]")
            rich.print()
            rich.print("[dim]... (showing last 1000 chars)[/dim]")
            rich.print()
            rich.print(f"[bright_blue]{content[-1000:]}[/bright_blue]")
            rich.print()
    else:
        rich.print("[green]✓ All samples have valid extracted choice[/green]")


if __name__ == "__main__":
    test_extract_choice_on_monitor_outputs()
