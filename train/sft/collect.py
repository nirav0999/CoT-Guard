#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

"""Collect balanced SFT examples from verified monitor outputs."""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from config import GLOBAL_DIRECTORY
from utils import extract_choice


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


def selected(path: Path, benign: bool, allowed_choices: set[str]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing monitor output: {path}")
    examples = []
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
            messages = row["messages"]
            if len(messages) < 2 or messages[0]["role"] != "user":
                raise ValueError(f"{path}:{line_number}: expected user and assistant messages")
            answer = messages[1]["content"]
            if extract_choice(answer) not in allowed_choices:
                continue
            examples.append({
                "instruction": messages[0]["content"],
                "output": answer,
                "input": "",
            })
    return examples


def collect(args: argparse.Namespace) -> list[Path]:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir or root / "datasets" / "sft")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rng = random.Random(args.seed)
    benign_choices = set(names(args.benign_choices))
    saved = []
    dataset_info_path = output_dir / "dataset_info.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = json.loads(dataset_info_path.read_text()) if dataset_info_path.exists() else {}
    for objective in names(args.monitor_objectives):
        malicious_examples = []
        benign_examples = []
        for main_task in names(args.main_tasks):
            for side_task in names(args.side_tasks):
                malicious, benign = monitor_paths(
                    root, main_task, side_task, args.attack_policy,
                    names(args.evaluated_model)[0], args.attack_rollouts,
                    names(args.monitor_model)[0], args.monitor_policy,
                    objective, args.filter_mode,
                )
                malicious_examples.extend(selected(malicious, False, {"C"}))
                benign_examples.extend(selected(benign, True, benign_choices))

        def fits(example: dict) -> bool:
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ]
            return len(tokenizer.apply_chat_template(messages, tokenize=True)) <= args.max_tokens

        malicious_examples = [example for example in malicious_examples if fits(example)]
        benign_examples = [example for example in benign_examples if fits(example)]
        count = min(len(malicious_examples), len(benign_examples))
        if count == 0:
            raise ValueError(f"No balanced examples for {objective}")
        rng.shuffle(malicious_examples)
        rng.shuffle(benign_examples)
        records = malicious_examples[:count] + benign_examples[:count]
        rng.shuffle(records)
        dataset_name = (f"cot_guard_{objective}_"
                        f"{'_'.join(names(args.main_tasks))}_"
                        f"{'_'.join(names(args.side_tasks))}")
        output_path = output_dir / f"{dataset_name}.json"
        if output_path.exists():
            raise FileExistsError(f"Dataset already exists: {output_path}")
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False)
        dataset_info[dataset_name] = {"file_name": output_path.name}
        saved.append(output_path)
        print(f"{objective}: {count} malicious + {count} benign -> {output_path}")
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")
    print(f"Registered datasets in {dataset_info_path}")
    return saved


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
    result.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    result.add_argument("--max-tokens", type=int, default=32768)
    result.add_argument("--benign-choices", default="A,B,D",
                        help="Replicates the original SFT selection; use A,B for reward-aligned labels")
    result.add_argument("--seed", type=int, default=42)
    return result


if __name__ == "__main__":
    collect(parser().parse_args())
