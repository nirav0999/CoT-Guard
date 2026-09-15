#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
from collections import defaultdict

import rich
from fire import Fire

from train.rl.reward_fn import compute_score
from utils import load_jsonl, save_jsonl


def print_success_at_k_distribution(
    label: str,
    success_at_k: dict[str, float],
    k: int,
) -> None:
    total = len(success_at_k)
    rich.print(
        f"[cyan bold underline]📊 {label} success@{k} bucket counts[/cyan bold underline]"
    )
    bin_counts: dict[int, int] = defaultdict(int)
    for value in success_at_k.values():
        num_correct = round(value * k)
        bin_counts[num_correct] += 1
    max_count = max(bin_counts.values()) if bin_counts else 1
    for i in range(k + 1):
        count = bin_counts[i]
        bar = "█" * (count * 40 // max(max_count, 1))
        rich.print(
            f"[dim]  {i}/{k} ({i / k:.3f})[/dim] {bar} [yellow]{count} / {total} = {count / total * 100:.1f}%[/yellow]"
            if total > 0
            else f"[dim]  {i}/{k} ({i / k:.3f})[/dim]  [yellow]0 / 0 = 0.0%[/yellow]"
        )


def filter_records(
    input_path: str,
    k: int = 8,
    low: float = 0.0,
    high: float = 1.0,
    easy_ratio: float = 0.2,
    seed: int = 42,
) -> None:

    assert input_path.endswith(".jsonl"), f"Expected .jsonl, got: {input_path}"
    ground_truth_path = input_path.replace(".jsonl", ".ground_truth.jsonl")
    rollout_path = input_path.replace(".jsonl", f".rollouts_n{k}.jsonl")
    assert os.path.exists(
        ground_truth_path
    ), f"Ground truth file not found: {ground_truth_path}"
    assert os.path.exists(rollout_path), f"Rollout file not found: {rollout_path}"
    rich.print(f"[dim]ground_truth_path={ground_truth_path}[/dim]")
    rich.print(f"[dim]rollout_path={rollout_path}[/dim]")
    rollouts = load_jsonl(rollout_path)
    gt_records = load_jsonl(ground_truth_path)
    rich.print(f"[blue]→ Loaded {len(rollouts)} rollouts from {rollout_path}[/blue]")
    rich.print(
        f"[blue]→ Loaded {len(gt_records)} ground truths from {ground_truth_path}[/blue]"
    )

    gt_map: dict[str, str] = {}
    for record in gt_records:
        gt_map[record["task_id"]] = record["ground_truth"]
    rich.print(f"[dim]Built gt_map for {len(gt_map)} task_ids[/dim]")
    rollouts_by_prompt: dict[str, list[dict]] = defaultdict(list)

    for rollout in rollouts:
        parts = rollout["task_id"].split(":")
        base_id = ":".join(parts[:-1])
        rollouts_by_prompt[base_id].append(rollout)
    rich.print(
        f"[dim]Grouped rollouts into {len(rollouts_by_prompt)} base task_ids[/dim]"
    )

    success_at_k: dict[str, float] = {}
    for base_id, rollout_group in rollouts_by_prompt.items():
        ground_truth = gt_map[base_id]
        num_correct = 0
        for rollout in rollout_group:
            assistant_response = rollout["messages"][-1]["content"]
            score = compute_score(
                data_source="tp_tn",
                solution_str=assistant_response,
                ground_truth=ground_truth,
            )
            num_correct += int(score["score"] == 1.0)
        success_at_k[base_id] = num_correct / len(rollout_group)
    rich.print(f"[yellow]Computed success@{k} for {len(success_at_k)} prompts[/yellow]")
    prompts_with_fewer = sum(
        1 for rollout_group in rollouts_by_prompt.values() if len(rollout_group) < k
    )
    if prompts_with_fewer > 0:
        rich.print(
            f"[orange3]⚠ {prompts_with_fewer} prompts have fewer than {k} rollouts[/orange3]"
        )
    tn_success_at_k = {
        task_id: value
        for task_id, value in success_at_k.items()
        if gt_map[task_id].strip().lower() in {"true", "1"}
    }
    tp_success_at_k = {
        task_id: value
        for task_id, value in success_at_k.items()
        if gt_map[task_id].strip().lower() not in {"true", "1"}
    }
    rich.print(
        f"[dim]tn_success_at_k={len(tn_success_at_k)} | tp_success_at_k={len(tp_success_at_k)}[/dim]"
    )
    total = len(success_at_k)
    all_less_than_high = sum(1 for value in success_at_k.values() if value < high)
    all_equal_high = sum(1 for value in success_at_k.values() if value == high)
    rich.print(
        f"[cyan bold underline]📊 ALL success@{k} (n={total}, high={high})[/cyan bold underline]"
    )
    rich.print(
        f"[yellow]  success@{k} < {high}: {all_less_than_high} / {total} = {all_less_than_high / total * 100:.1f}%[/yellow]"
    )
    rich.print(
        f"[yellow]  success@{k} == {high}: {all_equal_high} / {total} = {all_equal_high / total * 100:.1f}%[/yellow]"
    )
    print_success_at_k_distribution("ALL", success_at_k, k)
    tn_total = len(tn_success_at_k)
    tn_less_than_high = sum(1 for value in tn_success_at_k.values() if value < high)
    tn_equal_high = sum(1 for value in tn_success_at_k.values() if value == high)
    rich.print(
        f"[cyan bold underline]📊 TN success@{k} (n={tn_total}, high={high})[/cyan bold underline]"
    )
    rich.print(
        f"[yellow]  success@{k} < {high}: {tn_less_than_high} / {tn_total} = {tn_less_than_high / tn_total * 100:.1f}%[/yellow]"
        if tn_total > 0
        else f"[yellow]  success@{k} < {high}: 0 / 0 = 0.0%[/yellow]"
    )
    rich.print(
        f"[yellow]  success@{k} == {high}: {tn_equal_high} / {tn_total} = {tn_equal_high / tn_total * 100:.1f}%[/yellow]"
        if tn_total > 0
        else f"[yellow]  success@{k} == {high}: 0 / 0 = 0.0%[/yellow]"
    )
    print_success_at_k_distribution("TN", tn_success_at_k, k)
    tp_total = len(tp_success_at_k)
    tp_less_than_high = sum(1 for value in tp_success_at_k.values() if value < high)
    tp_equal_high = sum(1 for value in tp_success_at_k.values() if value == high)
    rich.print(
        f"[cyan bold underline]📊 TP success@{k} (n={tp_total}, high={high})[/cyan bold underline]"
    )
    rich.print(
        f"[yellow]  success@{k} < {high}: {tp_less_than_high} / {tp_total} = {tp_less_than_high / tp_total * 100:.1f}%[/yellow]"
        if tp_total > 0
        else f"[yellow]  success@{k} < {high}: 0 / 0 = 0.0%[/yellow]"
    )
    rich.print(
        f"[yellow]  success@{k} == {high}: {tp_equal_high} / {tp_total} = {tp_equal_high / tp_total * 100:.1f}%[/yellow]"
        if tp_total > 0
        else f"[yellow]  success@{k} == {high}: 0 / 0 = 0.0%[/yellow]"
    )
    print_success_at_k_distribution("TP", tp_success_at_k, k)
    input_records = load_jsonl(input_path)
    rich.print(f"[blue]→ Loaded {len(input_records)} records from {input_path}[/blue]")
    has_success_at_k_records = [
        record for record in input_records if record["task_id"] in success_at_k
    ]
    rich.print(
        f"[yellow]Records with success@{k}: {len(has_success_at_k_records)} / {len(input_records)} = {len(has_success_at_k_records) / len(input_records) * 100:.1f}%[/yellow]"
    )
    tp_all_records = [
        record
        for record in has_success_at_k_records
        if gt_map[record["task_id"]].strip().lower() not in {"true", "1"}
    ]
    tn_all_records = [
        record
        for record in has_success_at_k_records
        if gt_map[record["task_id"]].strip().lower() in {"true", "1"}
    ]
    rich.print(
        f"[dim]tp_all_records={len(tp_all_records)} | tn_all_records={len(tn_all_records)}[/dim]"
    )
    tp_less_than_high_records = [
        record for record in tp_all_records if success_at_k[record["task_id"]] < high
    ]
    tp_equal_high_records = [
        record for record in tp_all_records if success_at_k[record["task_id"]] == high
    ]
    tn_less_than_high_records = [
        record for record in tn_all_records if success_at_k[record["task_id"]] < high
    ]
    tn_equal_high_records = [
        record for record in tn_all_records if success_at_k[record["task_id"]] == high
    ]
    rng = random.Random(seed)
    num_tp_equal_high_to_keep = int(len(tp_equal_high_records) * easy_ratio)
    kept_tp_equal_high_records = (
        rng.sample(tp_equal_high_records, num_tp_equal_high_to_keep)
        if num_tp_equal_high_to_keep > 0
        else []
    )
    num_tp_equal_high_removed = len(tp_equal_high_records) - len(
        kept_tp_equal_high_records
    )
    num_tn_equal_high_to_remove = min(
        num_tp_equal_high_removed, len(tn_equal_high_records)
    )
    num_tn_equal_high_to_keep = len(tn_equal_high_records) - num_tn_equal_high_to_remove
    kept_tn_equal_high_records = (
        rng.sample(tn_equal_high_records, num_tn_equal_high_to_keep)
        if num_tn_equal_high_to_keep > 0
        else []
    )
    filtered_tp_records = tp_less_than_high_records + kept_tp_equal_high_records
    filtered_tn_records = tn_less_than_high_records + kept_tn_equal_high_records
    rich.print("[cyan bold underline]📊 TP filtering[/cyan bold underline]")
    rich.print(
        f"[yellow]  success@{k} < {high}: {len(tp_less_than_high_records)}[/yellow]"
    )
    rich.print(
        f"[yellow]  success@{k} == {high}: kept {len(kept_tp_equal_high_records)} / {len(tp_equal_high_records)} = {len(kept_tp_equal_high_records) / len(tp_equal_high_records) * 100:.1f}%[/yellow]"
        if len(tp_equal_high_records) > 0
        else f"[yellow]  success@{k} == {high}: kept 0 / 0 = 0.0%[/yellow]"
    )
    rich.print(f"[yellow]  total tp: {len(filtered_tp_records)}[/yellow]")
    rich.print("[cyan bold underline]📊 TN filtering[/cyan bold underline]")
    rich.print(
        f"[yellow]  success@{k} < {high}: {len(tn_less_than_high_records)}[/yellow]"
    )
    rich.print(
        f"[yellow]  success@{k} == {high}: kept {len(kept_tn_equal_high_records)} / {len(tn_equal_high_records)} = {len(kept_tn_equal_high_records) / len(tn_equal_high_records) * 100:.1f}% | removed_same_as_tp={num_tn_equal_high_to_remove}[/yellow]"
        if len(tn_equal_high_records) > 0
        else f"[yellow]  success@{k} == {high}: kept 0 / 0 = 0.0% | removed_same_as_tp=0[/yellow]"
    )
    rich.print(f"[yellow]  total tn: {len(filtered_tn_records)}[/yellow]")
    filtered_records = filtered_tp_records + filtered_tn_records
    output_path = input_path.replace(".jsonl", ".filtered.jsonl")
    save_jsonl(filtered_records, output_path)
    rich.print(
        f"[yellow]Final: {len(filtered_records)} / {len(input_records)} = {len(filtered_records) / len(input_records) * 100:.1f}% retained[/yellow]"
    )
    rich.print(f"[green]✓ Saved filtered dataset to {output_path}[/green]")


if __name__ == "__main__":
    Fire(filter_records)
