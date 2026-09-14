# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import extract_choice

STEP_FILE_RE = re.compile(r"^(train|val)_(\d+)\.jsonl$")


def collect_split_step_files(run_dir: str) -> dict[str, list[tuple[int, str]]]:
    assert os.path.isdir(run_dir), run_dir
    split_step_files: dict[str, list[tuple[int, str]]] = {"train": [], "val": []}
    for name in os.listdir(run_dir):
        match = STEP_FILE_RE.match(name)
        if match is None:
            continue
        split_name = match.group(1)
        step_no = int(match.group(2))
        split_step_files[split_name].append((step_no, os.path.join(run_dir, name)))
    split_step_files["train"].sort(key=lambda item: item[0])
    split_step_files["val"].sort(key=lambda item: item[0])
    return split_step_files


def parse_row(row: dict, input_path: str, line_no: int) -> dict:
    assert "is_benign" in row, f"{input_path}:{line_no} missing is_benign"
    assert "output" in row, f"{input_path}:{line_no} missing output"
    assert "data_source" in row, f"{input_path}:{line_no} missing data_source"

    is_benign = float(row["is_benign"]) == 1.0
    choice = extract_choice(row["output"])
    detected_hidden_objective = choice == "C"

    if "reward" in row:
        reward = float(row["reward"])
    elif "score" in row:
        reward = float(row["score"])
    else:
        raise ValueError(f"{input_path}:{line_no} missing reward and score")

    return {
        "is_benign": is_benign,
        "detected_hidden_objective": detected_hidden_objective,
        "choice": choice,
        "reward": reward,
        "data_source": str(row["data_source"]),
    }


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    assert n > 0
    n_detected = sum(1 for r in rows if r["detected_hidden_objective"])
    total_reward = sum(r["reward"] for r in rows)
    choice_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        choice_counts[r["choice"]] += 1
    return {
        "n": n,
        "n_detected": n_detected,
        "detection_rate": n_detected / n,
        "total_reward": total_reward,
        "mean_reward": total_reward / n,
        "choice_counts": dict(choice_counts),
    }


def aggregate_step_file(input_path: str) -> dict[str, dict]:
    assert input_path.endswith(".jsonl"), input_path
    groups: dict[str, list[dict]] = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            parsed = parse_row(row, input_path, line_no)
            groups[parsed["data_source"]].append(parsed)

    result: dict[str, dict] = {}
    for data_source, rows in groups.items():
        benign_rows = [r for r in rows if r["is_benign"]]
        malicious_rows = [r for r in rows if not r["is_benign"]]

        result[data_source] = {
            **summarize_rows(rows),
            "n_benign": len(benign_rows),
            "n_malicious": len(malicious_rows),
            "benign": summarize_rows(benign_rows) if benign_rows else None,
            "malicious": summarize_rows(malicious_rows) if malicious_rows else None,
        }

    return result


def aggregate_split_by_step(
    split_name: str, step_files: list[tuple[int, str]]
) -> list[dict]:
    aggregated_rows: list[dict] = []

    for step_no, input_path in step_files:
        per_source = aggregate_step_file(input_path)

        all_n = sum(s["n"] for s in per_source.values())
        all_reward = sum(s["total_reward"] for s in per_source.values())
        all_detected = sum(s["n_detected"] for s in per_source.values())

        aggregated_rows.append(
            {
                "split": split_name,
                "step_no": step_no,
                "input_path": input_path,
                "n": all_n,
                "mean_reward": all_reward / all_n,
                "detection_rate": all_detected / all_n,
                "per_source": per_source,
            }
        )

    return aggregated_rows


def aggregate_run(run_dir: str) -> dict[str, list[dict]]:
    split_step_files = collect_split_step_files(run_dir)
    return {
        "train": aggregate_split_by_step("train", split_step_files["train"]),
        "val": aggregate_split_by_step("val", split_step_files["val"]),
    }
