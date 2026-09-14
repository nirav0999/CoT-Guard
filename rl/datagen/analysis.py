# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import defaultdict

import fire
import rich


def parse_row(row: dict, input_path: str, line_no: int) -> tuple[str, bool]:
    assert "extra_info" in row, f"{input_path}:{line_no} missing extra_info"
    extra_info = row["extra_info"]
    assert "side_task" in extra_info, f"{input_path}:{line_no} missing side_task"
    assert "benign" in extra_info, f"{input_path}:{line_no} missing benign"
    side_task = str(extra_info["side_task"])
    is_benign = bool(extra_info["benign"])
    return side_task, is_benign


def count_side_task_distribution(
    input_path: str,
) -> tuple[dict[str, dict[str, int]], int]:
    assert os.path.isfile(input_path), input_path
    assert input_path.endswith(".jsonl"), input_path

    per_side: dict[str, dict[str, int]] = defaultdict(
        lambda: {"benign": 0, "malicious": 0}
    )
    total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            side_task, is_benign = parse_row(row, input_path, line_no)
            bucket = "benign" if is_benign else "malicious"
            per_side[side_task][bucket] += 1
            total += 1

    return dict(per_side), total


def print_distribution(
    per_side: dict[str, dict[str, int]], total: int, input_path: str
) -> None:
    rich.print(f"[magenta bold]{'═' * 72}[/magenta bold]")
    rich.print("[cyan bold underline]📊 Side task distribution[/cyan bold underline]")
    rich.print(f"[dim]input_path = {input_path}[/dim]")
    rich.print(f"[dim]total rows = {total}[/dim]")
    rich.print(f"[magenta bold]{'═' * 72}[/magenta bold]")

    overall_benign = sum(v["benign"] for v in per_side.values())
    overall_malicious = sum(v["malicious"] for v in per_side.values())

    rich.print(
        f"[yellow]benign total: {overall_benign} / {total} = "
        f"{overall_benign / total * 100:.1f}%[/yellow]"
    )
    rich.print(
        f"[yellow]malicious total: {overall_malicious} / {total} = "
        f"{overall_malicious / total * 100:.1f}%[/yellow]"
    )
    rich.print(f"[magenta bold]{'─' * 72}[/magenta bold]")

    for side_task in sorted(per_side.keys()):
        counts = per_side[side_task]
        n_benign = counts["benign"]
        n_malicious = counts["malicious"]
        n = n_benign + n_malicious
        rich.print(
            f"[cyan bold]{side_task}[/cyan bold]: "
            f"[yellow]{n} / {total} = {n / total * 100:.1f}%[/yellow]"
        )
        rich.print(
            f"  [yellow]benign:    {n_benign} / {n} = "
            f"{n_benign / n * 100:.1f}%[/yellow]"
        )
        rich.print(
            f"  [yellow]malicious: {n_malicious} / {n} = "
            f"{n_malicious / n * 100:.1f}%[/yellow]"
        )


def main(input_path: str) -> None:
    input_path = os.path.abspath(os.path.expanduser(input_path))
    rich.print(f"[blue]→ input_path = {input_path}[/blue]")

    per_side, total = count_side_task_distribution(input_path)
    assert total > 0, f"no rows in {input_path}"
    print_distribution(per_side, total, input_path)


if __name__ == "__main__":
    fire.Fire(main)
# python rl/datagen/analysis.py --input_path /srv/local/hanw14/icml2026/coding/clean/datasets/rl/cot_action.bigcodebench.complexity_defaultval_edgecase_longlines_varname.Qwen3-14B.Qwen3-32B.main_aware.baseline.train.jsonl
