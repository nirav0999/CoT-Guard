# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import fire
import matplotlib.pyplot as plt
import rich

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.analysis.aggregate import aggregate_run

CHOICE_LABELS = ["A", "B", "C", "D", "None"]

CHOICE_COLORS = {
    "A": "#3b82f6",
    "B": "#22c55e",
    "C": "#dc2626",
    "D": "#f59e0b",
    "None": "#6b7280",
}

LINE_LABELS = ["train", "ood_main", "ood_side", "ood_main_ood_side"]

LINE_COLORS = {
    "train": "#dc2626",
    "ood_main": "#a855f7",
    "ood_side": "#16a34a",
    "ood_main_ood_side": "#f59e0b",
}


def collect_line_data(
    aggregated: dict[str, list[dict]],
    benign_or_malicious: str,
    metric_key: str,
) -> dict[str, tuple[list[int], list[float]]]:
    assert benign_or_malicious in {"benign", "malicious"}
    lines: dict[str, tuple[list[int], list[float]]] = {}

    for label in LINE_LABELS:
        steps: list[int] = []
        values: list[float] = []

        if label == "train":
            for row in aggregated["train"]:
                combined_sub = _combine_train_sub(
                    row["per_source"], benign_or_malicious
                )
                if combined_sub is None:
                    continue
                steps.append(row["step_no"])
                values.append(combined_sub[metric_key])
        else:
            for row in aggregated["val"]:
                if label not in row["per_source"]:
                    continue
                sub = row["per_source"][label][benign_or_malicious]
                if sub is None:
                    continue
                steps.append(row["step_no"])
                values.append(sub[metric_key])

        if steps:
            lines[label] = (steps, values)

    return lines


def _combine_train_sub(
    per_source: dict[str, dict], benign_or_malicious: str
) -> dict | None:
    total_n = 0
    total_detected = 0
    total_reward = 0.0

    for source_stats in per_source.values():
        sub = source_stats[benign_or_malicious]
        if sub is None:
            continue
        total_n += sub["n"]
        total_detected += sub["n_detected"]
        total_reward += sub["total_reward"]

    if total_n == 0:
        return None

    return {
        "n": total_n,
        "n_detected": total_detected,
        "detection_rate": total_detected / total_n,
        "total_reward": total_reward,
        "mean_reward": total_reward / total_n,
    }


def write_plot_png(
    lines: dict[str, tuple[list[int], list[float]]],
    output_path: str,
    title: str,
    y_axis_label: str,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    for label in LINE_LABELS:
        if label not in lines:
            continue
        steps, values = lines[label]
        ax.plot(
            steps,
            values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
            color=LINE_COLORS[label],
        )

    ax.set_xlabel("step_no")
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def collect_choice_distribution(
    aggregated_rows: list[dict],
    benign_or_malicious: str,
) -> dict[str, tuple[list[int], list[float]]]:
    assert benign_or_malicious in {"benign", "malicious"}
    all_sources = sorted({src for row in aggregated_rows for src in row["per_source"]})

    steps: list[int] = []
    choice_fractions: dict[str, list[float]] = {c: [] for c in CHOICE_LABELS}

    for row in aggregated_rows:
        merged_counts: dict[str, int] = {c: 0 for c in CHOICE_LABELS}
        total = 0
        for src in all_sources:
            if src not in row["per_source"]:
                continue
            sub = row["per_source"][src][benign_or_malicious]
            if sub is None:
                continue
            counts = sub["choice_counts"]
            for choice, count in counts.items():
                merged_counts[choice] = merged_counts[choice] + count
                total += count
        if total == 0:
            continue
        steps.append(row["step_no"])
        for c in CHOICE_LABELS:
            choice_fractions[c].append(merged_counts[c] / total)

    return {c: (steps, choice_fractions[c]) for c in CHOICE_LABELS if steps}


def write_choice_distribution_plot(
    lines: dict[str, tuple[list[int], list[float]]],
    output_path: str,
    title: str,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    for choice in CHOICE_LABELS:
        if choice not in lines:
            continue
        steps, values = lines[choice]
        ax.plot(
            steps,
            values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=choice,
            color=CHOICE_COLORS[choice],
        )

    ax.set_xlabel("step_no")
    ax.set_ylabel("fraction")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_aggregated_to_json(
    aggregated: dict[str, list[dict]], output_path: str
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)


def build_plot_path(run_dir: str, suffix: str) -> str:
    run_name = os.path.basename(run_dir)
    figs_dir = os.path.join(PROJECT_ROOT, "rl", "figs")
    return os.path.join(figs_dir, f"{run_name}.{suffix}.png")


PLOT_CONFIGS = [
    ("mean_reward.benign", "benign", "mean_reward", "mean_reward (benign)"),
    ("mean_reward.malicious", "malicious", "mean_reward", "mean_reward (malicious)"),
    ("detection_rate.benign", "benign", "detection_rate", "detection_rate (benign)"),
    (
        "detection_rate.malicious",
        "malicious",
        "detection_rate",
        "detection_rate (malicious)",
    ),
]


def main(run_dir: str, output_json: str = "", max_step: int = -1) -> None:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    assert os.path.isdir(run_dir), run_dir
    rich.print("[magenta bold]═══ Plot Rewards ═══[/magenta bold]")
    rich.print(f"[blue]→ run_dir = {run_dir}[/blue]")

    aggregated = aggregate_run(run_dir)

    if max_step >= 0:
        for split_name in ["train", "val"]:
            aggregated[split_name] = [
                row for row in aggregated[split_name] if row["step_no"] <= max_step
            ]
        rich.print(f"[blue]→ max_step = {max_step}[/blue]")

    if output_json:
        output_json = os.path.abspath(os.path.expanduser(output_json))
        rich.print(f"[blue]→ output_json = {output_json}[/blue]")
        write_aggregated_to_json(aggregated, output_json)
        rich.print(f"[green]✓ wrote json: {output_json}[/green]")

    for suffix, benign_or_malicious, metric_key, title in PLOT_CONFIGS:
        lines = collect_line_data(aggregated, benign_or_malicious, metric_key)
        plot_path = build_plot_path(run_dir, suffix)
        rich.print(f"[blue]→ plot_path = {plot_path}[/blue]")
        write_plot_png(lines, plot_path, title, metric_key)
        rich.print(f"[green]✓ wrote plot: {plot_path}[/green]")

    for split_name in ["train", "val"]:
        if not aggregated[split_name]:
            continue
        for bm in ["benign", "malicious"]:
            choice_lines = collect_choice_distribution(aggregated[split_name], bm)
            suffix = f"choice_distribution.{split_name}.{bm}"
            choice_plot_path = build_plot_path(run_dir, suffix)
            rich.print(f"[blue]→ plot_path = {choice_plot_path}[/blue]")
            write_choice_distribution_plot(
                choice_lines,
                choice_plot_path,
                f"choice_distribution ({split_name}, {bm})",
            )
            rich.print(f"[green]✓ wrote plot: {choice_plot_path}[/green]")


if __name__ == "__main__":
    fire.Fire(main)
