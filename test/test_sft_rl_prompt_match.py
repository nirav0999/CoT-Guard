#!/usr/bin/env python3

# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import json
import sys

sys.path.insert(0, "/home/ndiwan2/projects/cot-monitor/cot_monitor_coding")

import fire
import rich


def load_sft_prompts(sft_path: str) -> set:
    assert sft_path.endswith(".json"), f"SFT path must be .json, got: {sft_path}"
    rich.print(f"[blue]→ Loading SFT data from: {sft_path}[/blue]")

    with open(sft_path, "r") as f:
        records = json.load(f)

    assert isinstance(
        records, list
    ), f"SFT file must contain a JSON array, got: {type(records)}"
    rich.print(f"[dim]  SFT records: {len(records)}[/dim]")

    prompts = set()
    for record in records:
        prompts.add(record["instruction"])
    return prompts


def load_rl_prompts(rl_path: str) -> set:
    assert rl_path.endswith(".jsonl"), f"RL path must be .jsonl, got: {rl_path}"
    rich.print(f"[blue]→ Loading RL data from: {rl_path}[/blue]")

    prompts = set()
    count = 0
    with open(rl_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                prompts.add(record["messages"][0]["content"])
                count += 1

    rich.print(f"[dim]  RL records: {count}[/dim]")
    return prompts


def test_sft_rl_prompt_match(
    sft_path: str = "/srv/local/hanw14/icml2026/coding/sft_config/data/cot_only.kodcode.permissions_iodelete_exit_varname_ioexfil_iodownload.Qwen3-14B.Qwen3-32B.main_aware.baseline.json",
    rl_path: str = "/srv/local/hanw14/icml2026/coding/datasets/rl/cot_only.kodcode.permissions_iodelete_exit_varname_ioexfil_iodownload.Qwen3-14B.Qwen3-32B.main_aware.baseline_check.jsonl",
) -> None:

    rich.print(
        f"[cyan bold underline]📊 Comparing SFT vs RL prompts[/cyan bold underline]"
    )
    rich.print()

    sft_prompts = load_sft_prompts(sft_path)
    rl_prompts = load_rl_prompts(rl_path)
    rich.print()

    rich.print(f"[yellow]Unique SFT prompts: {len(sft_prompts)}[/yellow]")
    rich.print(f"[yellow]Unique RL prompts: {len(rl_prompts)}[/yellow]")
    rich.print()

    in_sft_not_rl = sft_prompts - rl_prompts
    in_rl_not_sft = rl_prompts - sft_prompts
    in_both = sft_prompts & rl_prompts

    rich.print(
        f"[yellow]In both: {len(in_both)} / {len(sft_prompts | rl_prompts)} = {len(in_both) / len(sft_prompts | rl_prompts) * 100:.1f}%[/yellow]"
    )
    rich.print(f"[yellow]In SFT only: {len(in_sft_not_rl)}[/yellow]")
    rich.print(f"[yellow]In RL only: {len(in_rl_not_sft)}[/yellow]")
    rich.print()

    # if len(in_sft_not_rl) == 0 and len(in_rl_not_sft) == 0:
    #     rich.print(f"[green]✓ All prompts match between SFT and RL datasets[/green]")
    # else:
    #     if in_sft_not_rl:
    #         rich.print(f"[orange3]⚠ {len(in_sft_not_rl)} prompts in SFT but not in RL:[/orange3]")
    #         for prompt in list(in_sft_not_rl):
    #             rich.print(f"[magenta bold]{'═' * 60}[/magenta bold]")
    #             rich.print(f"[bright_blue]{prompt[:500]}[/bright_blue]")
    #         rich.print()

    #     if in_rl_not_sft:
    #         rich.print(f"[orange3]⚠ {len(in_rl_not_sft)} prompts in RL but not in SFT:[/orange3]")
    #         for prompt in list(in_rl_not_sft):
    #             rich.print(f"[magenta bold]{'═' * 60}[/magenta bold]")
    #             rich.print(f"[bright_blue]{prompt[:500]}[/bright_blue]")


if __name__ == "__main__":
    fire.Fire(test_sft_rl_prompt_match)
