#!/usr/bin/env python3

# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import os

import rich
from fire import Fire

from gen.generate_parallel import parallel_generate
from utils import load_jsonl, save_jsonl


def generate_rollouts(
    input_path: str = "/srv/local/hanw14/icml2026/coding/clean/datasets/rl/.datagen/cot_only.kodcode.permissions_iodelete_exit_varname_ioexfil_iodownload.Qwen3-14B.Qwen3-32B.main_aware.baseline.context_window_16384.sft_check.train.jsonl",
    model: str = "Qwen/Qwen3-4B",
    n: int = 8,
    devices: str = "0,1,2,3,4,5,6,7,8",
    tp: int = 1,
    temp: float = 0.6,
    max_tokens: int = 24576,
    bs: int = 128,
    scaling_factor: float | None = 2.0,
    balance_by_length: bool = True,
) -> None:
    assert input_path.endswith(".jsonl"), f"Expected .jsonl, got: {input_path}"
    assert os.path.exists(input_path), f"Missing input_path: {input_path}"

    data = load_jsonl(input_path)
    rich.print(f"[blue]→ Loaded {len(data)} records from {input_path}[/blue]")

    seen_task_ids: dict[str, int] = {}
    duplicate_task_ids: list[str] = []
    deduped_data: list[dict] = []
    for record in data:
        task_id = record["task_id"]
        if task_id in seen_task_ids:
            duplicate_task_ids.append(task_id)
            seen_task_ids[task_id] += 1
        else:
            seen_task_ids[task_id] = 1
            deduped_data.append(record)

    if duplicate_task_ids:
        rich.print(
            f"[orange3]⚠ Found {len(duplicate_task_ids)} duplicate task_ids — removing them[/orange3]"
        )
        for task_id in duplicate_task_ids:
            rich.print(
                f"[orange3]  duplicate: {task_id} (appears {seen_task_ids[task_id]} times)[/orange3]"
            )
        rich.print(
            f"[yellow]Deduplication: {len(data)} → {len(deduped_data)} = removed {len(data) - len(deduped_data)} / {len(data)}[/yellow]"
        )
    else:
        rich.print(f"[green]✓ No duplicate task_ids found[/green]")

    data = deduped_data

    seen_messages: dict[str, str] = {}
    shared_message_count = 0
    for record in data:
        message_text = record["messages"][0]["content"]
        if message_text in seen_messages:
            shared_message_count += 1
            if shared_message_count <= 5:
                rich.print(
                    f"[orange3]  ⚠ same messages: {record['task_id']} and {seen_messages[message_text]}[/orange3]"
                )
                rich.print(
                    f"[dim]    messages (first 120 chars): {message_text[:120]}[/dim]"
                )
        else:
            seen_messages[message_text] = record["task_id"]

    if shared_message_count > 0:
        rich.print(
            f"[orange3]⚠ {shared_message_count} records share messages with an earlier record (showed first 5)[/orange3]"
        )
    else:
        rich.print(f"[green]✓ No duplicate messages found[/green]")

    gen_input_path = input_path.replace(".jsonl", ".gen_input.jsonl")
    ground_truth_path = input_path.replace(".jsonl", ".ground_truth.jsonl")
    rollout_path = input_path.replace(".jsonl", f".rollouts_n{n}.jsonl")

    gen_records: list[dict] = []
    gt_records: list[dict] = []
    for record in data:
        gen_records.append(
            {
                "task_id": record["task_id"],
                "messages": record["messages"],
                "metadata": {
                    "data_source": record["data_source"],
                },
            }
        )
        gt_records.append(
            {
                "task_id": record["task_id"],
                "ground_truth": record["reward_model"]["ground_truth"],
                "data_source": record["data_source"],
            }
        )

    save_jsonl(gen_records, gen_input_path)
    rich.print(
        f"[green]✓ Saved {len(gen_records)} gen input records to {gen_input_path}[/green]"
    )

    save_jsonl(gt_records, ground_truth_path)
    rich.print(
        f"[green]✓ Saved {len(gt_records)} ground truth records to {ground_truth_path}[/green]"
    )

    rich.print(
        f"[magenta bold]═══ Generating {n} rollouts per prompt ═══[/magenta bold]"
    )
    rich.print(f"[dim]model: {model}[/dim]")
    rich.print(f"[dim]gen_input_path: {gen_input_path}[/dim]")
    rich.print(f"[dim]rollout_path: {rollout_path}[/dim]")

    parallel_generate(
        input_paths=gen_input_path,
        output_paths=rollout_path,
        model=model,
        devices=devices,
        tp=tp,
        temp=temp,
        num_rollouts=n,
        max_tokens=max_tokens,
        bs=bs,
        scaling_factor=scaling_factor,
        balance_by_length=balance_by_length,
    )

    rich.print(f"[green]✓ Rollouts saved to {rollout_path}[/green]")


if __name__ == "__main__":
    Fire(generate_rollouts)
# python rl/datagen/rollout.py --help
