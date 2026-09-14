# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import fire
import rich

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.analysis.aggregate import aggregate_run


def print_evolving_rewards(split_name: str, aggregated_rows: list[dict]) -> None:
    rich.print(
        f"\n[cyan bold underline]📊 {split_name} — evolving rewards[/cyan bold underline]"
    )

    all_sources: list[str] = sorted(
        {src for row in aggregated_rows for src in row["per_source"]}
    )

    header_parts = [
        f"{'step':>6}",
        f"{'n':>5}",
        f"{'mean_reward':>12}",
        f"{'detect_rate':>12}",
    ]
    for src in all_sources:
        header_parts.append(f"{src + '_reward':>20}")
        header_parts.append(f"{src + '_detect':>20}")
    rich.print(f"[dim]{' | '.join(header_parts)}[/dim]")
    sep_width = sum(len(p) for p in header_parts) + 3 * (len(header_parts) - 1)
    rich.print(f"[dim]{'—' * sep_width}[/dim]")

    for row in aggregated_rows:
        step = row["step_no"]
        n = row["n"]
        overall_reward = f"{row['mean_reward'] * 100:.1f}%"
        overall_detect = f"{row['detection_rate'] * 100:.1f}%"

        parts = [
            f"{step:>6}",
            f"{n:>5}",
            f"{overall_reward:>12}",
            f"{overall_detect:>12}",
        ]

        for src in all_sources:
            if src in row["per_source"]:
                s = row["per_source"][src]
                reward_str = (
                    f"{s['total_reward']:.0f}/{s['n']}={s['mean_reward'] * 100:.1f}%"
                )
                detect_str = (
                    f"{s['n_detected']}/{s['n']}={s['detection_rate'] * 100:.1f}%"
                )
                parts.append(f"{reward_str:>20}")
                parts.append(f"{detect_str:>20}")
            else:
                parts.append(f"{'—':>20}")
                parts.append(f"{'—':>20}")

        rich.print(f"[yellow]{' | '.join(parts)}[/yellow]")

    if aggregated_rows:
        rich.print(f"[dim]{'—' * sep_width}[/dim]")
        last = aggregated_rows[-1]
        rich.print(
            f"[green]✓ last step {last['step_no']}: "
            f"mean_reward={last['mean_reward'] * 100:.1f}%, "
            f"detection_rate={last['detection_rate'] * 100:.1f}%[/green]"
        )


def main(run_dir: str, split: str = "train") -> None:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    assert os.path.isdir(run_dir), run_dir
    assert split in {
        "train",
        "val",
        "both",
    }, f"split must be train/val/both, got {split}"

    rich.print("[magenta bold]═══ Print Rewards ═══[/magenta bold]")
    rich.print(f"[blue]→ run_dir = {run_dir}[/blue]")
    rich.print(f"[blue]→ split = {split}[/blue]")

    aggregated = aggregate_run(run_dir)

    if split == "both":
        print_evolving_rewards("train", aggregated["train"])
        print_evolving_rewards("val", aggregated["val"])
    else:
        print_evolving_rewards(split, aggregated[split])


if __name__ == "__main__":
    fire.Fire(main)
