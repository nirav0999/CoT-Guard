# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import rich
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import GLOBAL_DIRECTORY

SAVE_PATH = f"{GLOBAL_DIRECTORY}/datasets/rl/"
CONTEXT_LENGTH = 16384


def token_len_list(dataset: list[dict]) -> list[int]:
    TOKENIZER_MODEL = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )

    def num_tokens_chat(messages: list[dict[str, str]]) -> int:
        return len(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        )

    messages_list = [s["messages"] for s in dataset]

    with ThreadPoolExecutor() as executor:
        return list(
            tqdm(
                executor.map(num_tokens_chat, messages_list),
                total=len(messages_list),
                desc="Processing",
            )
        )


def create_record(
    path_file: Path,
    side_task: str,
    benign: bool,
    data_source: str,
) -> list[dict]:
    ground_truth = str(benign)
    records = []
    with path_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            main_pass = data["metadata"]["main_pass"]
            side_pass = data["metadata"]["side_pass"]

            if benign:
                if not main_pass:
                    continue
            else:
                if not side_pass:
                    continue

            records.append(
                {
                    "task_id": f"{side_task}:{data['task_id']}",
                    "data_source": data_source,
                    "prompt": [data["messages"][0]],
                    "messages": data["messages"],
                    "reward_model": {"ground_truth": ground_truth},
                    "extra_info": {
                        "benign": benign,
                        "side_task": side_task,
                        "validation": {
                            "main_pass": main_pass,
                            "side_pass": side_pass,
                        },
                    },
                }
            )
    return records


def load_sft_prompts(sft_path: str):
    if not os.path.exists(sft_path):
        rich.print("[red] SFT path does not exist")
        return []

    assert sft_path.endswith(".json"), f"SFT path must be .json, got: {sft_path}"
    rich.print(f"[blue]→ Loading SFT prompts from: {sft_path}[/blue]")
    with open(sft_path, "r") as f:
        records = json.load(f)

    assert isinstance(
        records, list
    ), f"SFT file must contain a JSON array, got: {type(records)}"
    sft_prompts = set()
    for record in records:
        sft_prompts.add(record["instruction"])

    rich.print(
        f"[dim]  SFT records: {len(records)} | Unique prompts: {len(sft_prompts)}[/dim]"
    )
    return sft_prompts


def filter_against_sft_dataset(rl_records: List[dict], sft_records):
    in_sft_dataset = []
    for r in rl_records:
        p = r["messages"][0]["content"]
        if p in sft_records:
            in_sft_dataset.append(r)

    rich.print(f"[green] Selected {len(in_sft_dataset)} out of {len(rl_records)}")
    return in_sft_dataset


def check_overlap(rl_prompts, sft_prompts):
    set_1 = set(sft_prompts)
    set_2 = set(rl_prompts)

    overlap_1_in_2 = sum(1 for prompt in sft_prompts if prompt in set_2)
    overlap_2_in_1 = sum(1 for prompt in rl_prompts if prompt in set_1)

    rich.print(
        f"[green]Number of prompts in SFT dataset in RL Dataset: {overlap_1_in_2}"
    )
    rich.print(
        f"[green]Number of prompts in RL Dataset in SFT Dataset: {overlap_2_in_1}"
    )


def collect_main(
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
    top_k: int = 256,
    set_type: str = "train",
    check_against_sft: bool = False,
    sft_path: str | None = None,
):
    records = []
    for monitor_objective in monitor_objectives:
        for main_task in main_tasks:
            for side_task in side_tasks:
                for evaluated_model in evaluated_models:
                    for monitor_model in monitor_models:
                        for monitor_policy in monitor_policies:
                            for attack_policy in attack_policies:

                                rich.print(
                                    f"[bold yellow] [INFO] Main: {main_task}, Side: {side_task}, eval_model: {evaluated_model}, monitor_model: {monitor_model}, monitor_policy: {monitor_policy}, attack_policy: {attack_policy}, monitor_objective: {monitor_objective} [/bold yellow]"
                                )

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

                                assert (
                                    side_path_file.exists()
                                ), f"[ERROR] Missing side-task monitor file: {side_path_file}"
                                assert (
                                    main_path_file.exists()
                                ), f"[ERROR] Missing no_side_task monitor file: {main_path_file}"

                                records += create_record(
                                    side_path_file,
                                    side_task,
                                    benign=False,
                                    data_source=f"{set_type}_malicious",
                                )

                                records += create_record(
                                    main_path_file,
                                    side_task,
                                    benign=True,
                                    data_source=f"{set_type}_benign",
                                )

    # base_name = f'{monitor_objective}.{"_".join(main_tasks)}.{"_".join(side_tasks)}.{"_".join(evaluated_models)}.{"_".join(monitor_models)}.{"_".join(monitor_policies)}.{"_".join(attack_policies)}'

    base_name = f'{"_".join(monitor_objectives)}.{"_".join(main_tasks)}.{"_".join(side_tasks)}.{"_".join(evaluated_models)}.{"_".join(monitor_models)}.{"_".join(monitor_policies)}.{"_".join(attack_policies)}'
    list_token_lens = token_len_list(records)

    per_side_task_malicious = {}
    benign_pool = []

    for r, num_tokens_r in zip(records, list_token_lens):
        if num_tokens_r > CONTEXT_LENGTH:
            continue

        if r["extra_info"]["benign"]:
            benign_pool.append((r, num_tokens_r))
        else:
            st = r["extra_info"]["side_task"]
            per_side_task_malicious.setdefault(st, []).append((r, num_tokens_r))

    random.seed(42)

    random.shuffle(benign_pool)
    benign_pool = sorted(benign_pool, key=lambda x: x[1], reverse=True)
    if top_k > 0:
        benign_pool = benign_pool[:top_k]

    malicious_pool = []
    for st in sorted(per_side_task_malicious.keys()):
        m_items = per_side_task_malicious[st]
        random.shuffle(m_items)
        m_items = sorted(m_items, key=lambda x: x[1], reverse=True)
        if top_k > 0:
            m_items = m_items[:top_k]
        rich.print(f"[yellow]{st}: malicious={len(m_items)}[/]")
        malicious_pool.extend(m_items)

    rich.print(f"[yellow]benign (pooled): {len(benign_pool)}[/]")

    benign_pool = [r for r, _ in benign_pool]
    malicious_pool = [r for r, _ in malicious_pool]

    mal_dist = Counter(r["extra_info"]["side_task"] for r in malicious_pool)
    rich.print("[cyan bold underline]📊 Malicious pool distribution by side_task[/]")
    for st, count in sorted(mal_dist.items()):
        rich.print(
            f"[yellow]  {st}: {count} / {len(malicious_pool)} = {count / len(malicious_pool) * 100:.1f}%[/]"
        )

    base_name = base_name + f".{set_type}"
    if top_k > 0:
        base_name += f".top_{top_k}_mal_per_side_task"

    random.shuffle(benign_pool)
    random.shuffle(malicious_pool)

    save_path = Path(SAVE_PATH) / f"{base_name}.jsonl"

    records = benign_pool + malicious_pool

    if sft_path is not None:
        sft_prompts = load_sft_prompts(sft_path=sft_path)
        if check_against_sft:
            records = filter_against_sft_dataset(
                rl_records=records, sft_records=sft_prompts
            )
        check_overlap([r["messages"][0]["content"] for r in records], sft_prompts)

    with save_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return save_path


if __name__ == "__main__":
    from fire import Fire

    Fire(collect_main)

# Sample run:
# python -m rl.datagen.collect \
#   --main_tasks '["kodcode"]' \
#   --side_tasks '["permissions","iodelete","exit","varname","ioexfil","iodownload"]' \
#   --evaluated_models '["Qwen3-14B"]' \
#   --attack_policies '["baseline"]' \
#   --num_rollouts 20 \
#   --top_k 256 \
#   --set_type train
