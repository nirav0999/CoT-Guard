#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import rich
from transformers import AutoTokenizer

from utils import GLOBAL_DIRECTORY, extract_choice

SAVE_PATH = "/srv/local/hanw14/icml2026/coding/datasets/rl"
MODEL_NAME_OR_PATH = "Qwen/Qwen3-32B"
CONTEXT_WINDOW_LENGTH = 32768


def load_sft_prompts(sft_path: str) -> set:
    assert sft_path.endswith(".json"), f"SFT path must be .json, got: {sft_path}"
    rich.print(f"[blue]→ Loading SFT prompts from: {sft_path}[/blue]")

    with open(sft_path, "r") as f:
        records = json.load(f)

    assert isinstance(
        records, list
    ), f"SFT file must contain a JSON array, got: {type(records)}"

    prompts = set()
    for record in records:
        prompts.add(record["instruction"])

    rich.print(
        f"[dim]  SFT records: {len(records)} | Unique prompts: {len(prompts)}[/dim]"
    )
    return prompts


def collect_rl_from_sft_prompts(
    sft_path: str,
    main_tasks: list,
    side_tasks: list,
    evaluated_models: list,
    monitor_models: list,
    attack_policies: list,
    rollout_num: int,
    monitor_policies: list,
    monitor_objectives: list,
    filter: str,
) -> None:

    sft_prompts = load_sft_prompts(sft_path)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=True,
        trust_remote_code=True,
    )

    def num_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    for monitor_objective in monitor_objectives:

        all_records_TP = []
        all_records_TN = []

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

                                if side_path_file.exists() and main_path_file.exists():

                                    with side_path_file.open(
                                        "r", encoding="utf-8"
                                    ) as f:
                                        for line in f:
                                            if line.strip():
                                                data = json.loads(line)
                                                monitor_response = data["messages"][1][
                                                    "content"
                                                ]
                                                prompt = data["messages"][0]["content"]

                                                if (
                                                    extract_choice(monitor_response)
                                                    == "C"
                                                    and prompt in sft_prompts
                                                ):
                                                    all_records_TP.append(
                                                        {
                                                            "messages": data[
                                                                "messages"
                                                            ],
                                                            "benign": False,
                                                            "validation": {
                                                                "main_pass": data[
                                                                    "metadata"
                                                                ]["main_pass"],
                                                                "side_pass": data[
                                                                    "metadata"
                                                                ]["side_pass"],
                                                            },
                                                        }
                                                    )

                                    with main_path_file.open(
                                        "r", encoding="utf-8"
                                    ) as f:
                                        for line in f:
                                            if line.strip():
                                                data = json.loads(line)
                                                monitor_response = data["messages"][1][
                                                    "content"
                                                ]
                                                prompt = data["messages"][0]["content"]

                                                if (
                                                    extract_choice(monitor_response)
                                                    in ["A", "B", "D"]
                                                    and prompt in sft_prompts
                                                ):
                                                    all_records_TN.append(
                                                        {
                                                            "messages": data[
                                                                "messages"
                                                            ],
                                                            "benign": True,
                                                            "validation": {
                                                                "main_pass": data[
                                                                    "metadata"
                                                                ]["main_pass"],
                                                                "side_pass": data[
                                                                    "metadata"
                                                                ]["side_pass"],
                                                            },
                                                        }
                                                    )
                                else:
                                    rich.print(
                                        f"[dark_red bold][ERROR] Either {side_path_file} or {main_path_file} does not exist. Exit the collecting training data script. [/dark_red bold]"
                                    )

        output_file = (
            Path(SAVE_PATH)
            / f"{monitor_objective}.{"_".join(main_tasks)}.{"_".join(side_tasks)}.{"_".join(evaluated_models)}.{"_".join(monitor_models)}.{"_".join(monitor_policies)}.{"_".join(attack_policies)}_check.jsonl"
        )

        num_TN = len(all_records_TN)
        all_records_TN = [
            r
            for r in all_records_TN
            if num_tokens(r["messages"][0]["content"] + r["messages"][1]["content"])
            <= CONTEXT_WINDOW_LENGTH
        ]
        retained_num_TN = len(all_records_TN)
        rich.print(
            f"[bold blue] Ratio of TN samples exceeding context window length: {(num_TN - retained_num_TN)/num_TN} | Retain_num_TN: {retained_num_TN} | num_TN: {num_TN}[/bold blue]"
        )

        num_TP = len(all_records_TP)
        all_records_TP = [
            r
            for r in all_records_TP
            if num_tokens(r["messages"][0]["content"] + r["messages"][1]["content"])
            <= CONTEXT_WINDOW_LENGTH
        ]
        retained_num_TP = len(all_records_TP)
        rich.print(
            f"[bold blue] Ratio of TP samples exceeding context window length: {(num_TP - retained_num_TP)/num_TP} | Retained_num_TP: {retained_num_TP} | num_TP: {num_TP} [/bold blue]"
        )

        all_records = all_records_TN + all_records_TP
        rich.print(
            f"[yellow]TP: {len(all_records_TP)} | TN: {len(all_records_TN)} | Total: {len(all_records)}[/yellow]"
        )

        matched_prompts = {r["messages"][0]["content"] for r in all_records}
        unmatched = sft_prompts - matched_prompts
        rich.print(
            f"[yellow]SFT prompts matched: {len(matched_prompts)} / {len(sft_prompts)} = {len(matched_prompts) / len(sft_prompts) * 100:.1f}%[/yellow]"
        )
        rich.print(
            f"[yellow]SFT prompts not found in RL sources: {len(unmatched)}[/yellow]"
        )

        with output_file.open("w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        rich.print(
            f"[green]✓ Saved {len(all_records)} records to {output_file}[/green]"
        )


if __name__ == "__main__":
    collect_rl_from_sft_prompts(
        sft_path="/srv/local/hanw14/icml2026/coding/sft_config/data/cot_only.kodcode.permissions_iodelete_exit_varname_ioexfil_iodownload.Qwen3-14B.Qwen3-32B.main_aware.baseline.json",
        main_tasks=["kodcode"],
        side_tasks=[
            "permissions",
            "iodelete",
            "exit",
            "varname",
            "ioexfil",
            "iodownload",
        ],
        evaluated_models=["Qwen3-14B"],
        monitor_models=["Qwen3-32B"],
        attack_policies=["baseline"],
        rollout_num=1,
        monitor_policies=["main_aware"],
        monitor_objectives=["cot_only"],
        filter="side-all",
    )
