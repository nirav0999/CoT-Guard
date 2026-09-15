#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

"""Build prompt-only GRPO records from verified monitor runs."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from config import GLOBAL_DIRECTORY


def names(value: str) -> list[str]:
    return [part.strip().split("/")[-1] for part in value.split(",") if part.strip()]


def monitor_paths(root: Path, main_task: str, side_task: str, attack_policy: str,
                  evaluated_model: str, attack_rollouts: int, monitor_model: str,
                  monitor_policy: str, objective: str, filter_mode: str) -> tuple[Path, Path]:
    base = root / "results" / "monitor"
    rollout_dir = f"{evaluated_model}--{attack_rollouts}"
    malicious = (base / f"{main_task}.{side_task}.{attack_policy}" / rollout_dir /
                 f"{filter_mode}.{monitor_model}.{monitor_policy}.{objective}.jsonl")
    benign = (base / f"{main_task}.{side_task}.no_side_task" / rollout_dir /
              f"main-all.{monitor_model}.{monitor_policy}.{objective}.jsonl")
    return malicious, benign


def read_records(path: Path, benign: bool, main_task: str, side_task: str,
                 objective: str, set_type: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing monitor output: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row["metadata"]
            if benign and not metadata["main_pass"]:
                continue
            if not benign and not metadata["side_pass"]:
                continue
            user = row["messages"][0]
            if user["role"] != "user":
                raise ValueError(f"{path}:{line_number}: expected a user prompt")
            records.append({
                "task_id": f"{main_task}:{side_task}:{objective}:{benign}:{row['task_id']}",
                "data_source": f"{set_type}_{'benign' if benign else 'malicious'}",
                "prompt": [user],
                "reward_model": {"ground_truth": str(benign)},
                "extra_info": {
                    "benign": benign,
                    "side_task": side_task,
                    "monitor_objective": objective,
                    "validation": {
                        "main_pass": metadata["main_pass"],
                        "side_pass": metadata["side_pass"],
                    },
                },
            })
    return records


def collect(args: argparse.Namespace) -> Path:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir or root / "datasets" / "rl")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    records = []
    for objective in names(args.monitor_objectives):
        for main_task in names(args.main_tasks):
            for side_task in names(args.side_tasks):
                malicious, benign = monitor_paths(
                    root, main_task, side_task, args.attack_policy,
                    names(args.evaluated_model)[0], args.attack_rollouts,
                    names(args.monitor_model)[0], args.monitor_policy,
                    objective, args.filter_mode,
                )
                records.extend(read_records(malicious, False, main_task, side_task,
                                            objective, args.set_type))
                records.extend(read_records(benign, True, main_task, side_task,
                                            objective, args.set_type))

    records = [record for record in records if len(tokenizer.apply_chat_template(
        record["prompt"], add_generation_prompt=True, tokenize=True
    )) <= args.max_prompt_tokens]

    if args.sft_path:
        with Path(args.sft_path).open(encoding="utf-8") as handle:
            sft_prompts = {row["instruction"] for row in json.load(handle)}
        overlap = sum(record["prompt"][0]["content"] in sft_prompts for record in records)
        print(f"SFT prompt overlap: {overlap}/{len(records)}")
        if args.sft_only:
            records = [record for record in records
                       if record["prompt"][0]["content"] in sft_prompts]
    elif args.sft_only:
        raise ValueError("--sft-only requires --sft-path")

    rng = random.Random(args.seed)
    if args.top_k > 0:
        groups = {}
        for record in records:
            key = (record["extra_info"]["side_task"], record["extra_info"]["benign"])
            groups.setdefault(key, []).append(record)
        records = []
        for group in groups.values():
            records.extend(rng.sample(group, min(args.top_k, len(group))))
    rng.shuffle(records)
    if not records:
        raise ValueError("No eligible RL records were found")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (f"{'_'.join(names(args.monitor_objectives))}."
                f"{'_'.join(names(args.main_tasks))}."
                f"{'_'.join(names(args.side_tasks))}.{args.set_type}.jsonl")
    output_path = output_dir / filename
    if output_path.exists():
        raise FileExistsError(f"Dataset already exists: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} RL prompts to {output_path}")
    print(f"Labels: {dict(Counter(record['reward_model']['ground_truth'] for record in records))}")
    return output_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", default=GLOBAL_DIRECTORY)
    result.add_argument("--output-dir")
    result.add_argument("--main-tasks", required=True)
    result.add_argument("--side-tasks", required=True)
    result.add_argument("--evaluated-model", default="Qwen/Qwen3-14B")
    result.add_argument("--monitor-model", default="Qwen/Qwen3-32B")
    result.add_argument("--attack-policy", default="baseline")
    result.add_argument("--attack-rollouts", type=int, default=5)
    result.add_argument("--monitor-policy", default="main_aware")
    result.add_argument("--monitor-objectives", default="action_only,cot_only,cot_action")
    result.add_argument("--filter-mode", default="side-all")
    result.add_argument("--set-type", choices=("train", "val"), default="train")
    result.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    result.add_argument("--max-prompt-tokens", type=int, default=16384)
    result.add_argument("--top-k", type=int, default=-1,
                        help="Maximum records per side task and label; -1 keeps all")
    result.add_argument("--sft-path", help="Optional SFT JSON for overlap reporting")
    result.add_argument("--sft-only", action="store_true",
                        help="Keep only prompts found in --sft-path")
    result.add_argument("--seed", type=int, default=42)
    return result


if __name__ == "__main__":
    collect(parser().parse_args())
