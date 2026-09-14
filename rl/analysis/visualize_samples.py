# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import sys
from collections import defaultdict

import fire
import rich
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.analysis.aggregate import collect_split_step_files
from utils import extract_choice, extract_cot

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

app_data: dict = {}


def sample_id_from_input(input_str: str) -> str:
    return hashlib.md5(input_str.encode()).hexdigest()[:16]


def content_id_from_rollout_input(input_str: str) -> str:
    assert input_str.startswith("user\n"), repr(input_str[:10])
    assert input_str.endswith("\nassistant\n"), repr(input_str[-15:])
    content = input_str[len("user\n") : -len("\nassistant\n")]
    return hashlib.md5(content.encode()).hexdigest()[:16]


def build_task_metadata_lookup(dataset_files: list[str]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for path in dataset_files:
        assert os.path.isfile(path), f"dataset file not found: {path}"
        rich.print(f"[dim]loading task metadata from {path}[/dim]")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "messages" in row:
                    content = row["messages"][0]["content"]
                elif "prompt" in row:
                    content = row["prompt"][0]["content"]
                else:
                    raise ValueError(
                        f"no messages or prompt in {path}: {list(row.keys())}"
                    )

                cid = hashlib.md5(content.encode()).hexdigest()[:16]
                task_id = row["task_id"]
                side_task = row["extra_info"]["side_task"]
                parts = task_id.split(":")
                main_task_id = parts[1].split("_")[0]
                lookup[cid] = {
                    "task_id": task_id,
                    "side_task": side_task,
                    "main_task_id": main_task_id,
                }
    rich.print(f"[green]✓ task metadata: {len(lookup)} entries[/green]")
    return lookup


def attach_task_metadata(samples: dict[str, dict], lookup: dict[str, dict]) -> None:
    matched = 0
    for key, s in samples.items():
        cid = content_id_from_rollout_input(s["input"])
        if cid in lookup:
            meta = lookup[cid]
            s["task_id"] = meta["task_id"]
            s["side_task"] = meta["side_task"]
            s["main_task_id"] = meta["main_task_id"]
            matched += 1
        else:
            s["task_id"] = "unknown"
            s["side_task"] = "unknown"
            s["main_task_id"] = "unknown"
    rich.print(
        f"[yellow]task metadata matched: {matched} / {len(samples)} = {matched / len(samples) * 100:.1f}%[/yellow]"
    )


def index_val_samples(val_files: list[tuple[int, str]]) -> dict[str, dict]:
    samples: dict[str, dict] = {}

    rich.print(f"[dim]indexing val from {val_files[0][1]}[/dim]")
    first_step, first_path = val_files[0]
    with open(first_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = sample_id_from_input(row["input"])
            samples[sid] = {
                "id": sid,
                "split": "val",
                "input": row["input"],
                "is_benign": float(row["is_benign"]) == 1.0,
                "data_source": str(row["data_source"]),
                "input_preview": row["input"][:150].replace("\n", " "),
                "trajectory": [],
            }

    rich.print(f"[dim]scanning {len(val_files)} val files for trajectories...[/dim]")
    for step, path in val_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = sample_id_from_input(row["input"])
                assert sid in samples, f"unknown val sample {sid} at step {step}"
                choice = extract_choice(row["output"])
                reward = float(row["score"]) if "score" in row else float(row["reward"])
                samples[sid]["trajectory"].append(
                    {
                        "step": step,
                        "detected": choice == "C",
                        "choice": choice,
                        "reward": reward,
                    }
                )

    for sid in samples:
        samples[sid]["trajectory"].sort(key=lambda t: t["step"])

    rich.print(f"[green]✓ val: {len(samples)} samples[/green]")
    return samples


def index_train_samples(train_files: list[tuple[int, str]]) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    rollups: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))

    rich.print(f"[dim]scanning {len(train_files)} train files...[/dim]")
    for step, path in train_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = sample_id_from_input(row["input"])

                if sid not in samples:
                    samples[sid] = {
                        "id": sid,
                        "split": "train",
                        "input": row["input"],
                        "is_benign": float(row["is_benign"]) == 1.0,
                        "data_source": str(row["data_source"]),
                        "input_preview": row["input"][:150].replace("\n", " "),
                        "trajectory": [],
                    }

                choice = extract_choice(row["output"])
                reward = float(row["score"]) if "score" in row else float(row["reward"])
                rollups[sid][step].append(
                    {
                        "choice": choice,
                        "detected": choice == "C",
                        "reward": reward,
                    }
                )

    for sid, steps_dict in rollups.items():
        for step in sorted(steps_dict):
            rollout_list = steps_dict[step]
            n = len(rollout_list)
            n_detected = sum(1 for r in rollout_list if r["detected"])
            mean_reward = sum(r["reward"] for r in rollout_list) / n
            samples[sid]["trajectory"].append(
                {
                    "step": step,
                    "detected": n_detected / n > 0.5,
                    "detection_rate": n_detected / n,
                    "n_rollouts": n,
                    "n_detected": n_detected,
                    "reward": mean_reward,
                }
            )

    rich.print(f"[green]✓ train: {len(samples)} unique samples[/green]")
    return samples


def build_sample_index(run_dir: str, dataset_files: list[str]) -> None:
    split_step_files = collect_split_step_files(run_dir)

    val_files = split_step_files["val"]
    train_files = split_step_files["train"]

    val_samples = index_val_samples(val_files) if val_files else {}
    train_samples = index_train_samples(train_files) if train_files else {}

    all_samples: dict[str, dict] = {}
    for sid, s in val_samples.items():
        all_samples[f"val_{sid}"] = s
    for sid, s in train_samples.items():
        all_samples[f"train_{sid}"] = s

    if dataset_files:
        lookup = build_task_metadata_lookup(dataset_files)
        attach_task_metadata(all_samples, lookup)
        side_tasks = sorted({s["side_task"] for s in all_samples.values()})
    else:
        side_tasks = []

    val_step_to_path: dict[int, str] = {step: path for step, path in val_files}
    train_step_to_path: dict[int, str] = {step: path for step, path in train_files}

    data_sources = sorted({s["data_source"] for s in all_samples.values()})

    app_data["samples"] = all_samples
    app_data["val_step_to_path"] = val_step_to_path
    app_data["train_step_to_path"] = train_step_to_path
    app_data["data_sources"] = data_sources
    app_data["side_tasks"] = side_tasks
    app_data["val_steps"] = sorted(val_step_to_path.keys())
    app_data["train_steps"] = sorted(train_step_to_path.keys())

    rich.print(
        f"[green]✓ total: {len(all_samples)} samples ({len(val_samples)} val + {len(train_samples)} train)[/green]"
    )
    rich.print(f"[dim]data_sources: {data_sources}[/dim]")
    rich.print(f"[dim]side_tasks: {side_tasks}[/dim]")


@app.route("/")
def index() -> str:
    return render_template(
        "visualize_samples.html",
        data_sources=app_data["data_sources"],
        side_tasks=app_data["side_tasks"],
    )


@app.route("/api/samples")
def api_samples() -> tuple:
    data_source = request.args.get("data_source", "all")
    is_benign = request.args.get("is_benign", "all")
    split = request.args.get("split", "all")
    side_task = request.args.get("side_task", "all")

    filtered = []
    for key, s in app_data["samples"].items():
        if data_source != "all" and s["data_source"] != data_source:
            continue
        if is_benign == "benign" and not s["is_benign"]:
            continue
        if is_benign == "malicious" and s["is_benign"]:
            continue
        if split != "all" and s["split"] != split:
            continue
        if side_task != "all" and s.get("side_task", "unknown") != side_task:
            continue

        n_detected = sum(1 for t in s["trajectory"] if t["detected"])
        n_steps = len(s["trajectory"])
        filtered.append(
            {
                "id": key,
                "split": s["split"],
                "input_preview": s["input_preview"],
                "is_benign": s["is_benign"],
                "data_source": s["data_source"],
                "task_id": s.get("task_id", "unknown"),
                "side_task": s.get("side_task", "unknown"),
                "main_task_id": s.get("main_task_id", "unknown"),
                "n_detected": n_detected,
                "n_steps": n_steps,
            }
        )

    filtered.sort(key=lambda x: x["n_detected"], reverse=True)
    return jsonify(filtered)


@app.route("/api/sample/<sample_id>")
def api_sample(sample_id: str) -> tuple:
    assert sample_id in app_data["samples"], f"unknown sample {sample_id}"
    s = app_data["samples"][sample_id]
    return jsonify(
        {
            "id": sample_id,
            "split": s["split"],
            "input": s["input"],
            "is_benign": s["is_benign"],
            "data_source": s["data_source"],
            "task_id": s.get("task_id", "unknown"),
            "side_task": s.get("side_task", "unknown"),
            "main_task_id": s.get("main_task_id", "unknown"),
            "trajectory": s["trajectory"],
        }
    )


@app.route("/api/sample/<sample_id>/step/<int:step_no>")
def api_sample_step(sample_id: str, step_no: int) -> tuple:
    assert sample_id in app_data["samples"], f"unknown sample {sample_id}"
    s = app_data["samples"][sample_id]
    split = s["split"]

    if split == "val":
        assert step_no in app_data["val_step_to_path"], f"unknown val step {step_no}"
        path = app_data["val_step_to_path"][step_no]
    else:
        assert (
            step_no in app_data["train_step_to_path"]
        ), f"unknown train step {step_no}"
        path = app_data["train_step_to_path"][step_no]

    raw_sid = sample_id.split("_", 1)[1]
    target_input = s["input"]

    rollouts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_id_from_input(row["input"]) == raw_sid:
                output = row["output"]
                cot = extract_cot(output)
                choice = extract_choice(output)
                after_think = (
                    output.split("</think>")[-1] if "</think>" in output else output
                )
                reward = float(row["score"]) if "score" in row else float(row["reward"])
                rollouts.append(
                    {
                        "output": output,
                        "cot": cot,
                        "answer_section": after_think.strip(),
                        "choice": choice,
                        "detected": choice == "C",
                        "reward": reward,
                    }
                )

    assert len(rollouts) > 0, f"sample {sample_id} not found in {path}"

    return jsonify(
        {
            "step": step_no,
            "split": split,
            "n_rollouts": len(rollouts),
            "rollouts": rollouts,
        }
    )


def main(
    run_dir: str, dataset_files: list[str] = [], port: int = 5001, host: str = "0.0.0.0"
) -> None:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    assert os.path.isdir(run_dir), run_dir
    rich.print(f"[magenta bold]═══ Visualize Samples ═══[/magenta bold]")
    rich.print(f"[blue]→ run_dir = {run_dir}[/blue]")
    rich.print(f"[blue]→ dataset_files = {dataset_files}[/blue]")
    rich.print(f"[blue]→ port = {port}[/blue]")

    build_sample_index(run_dir, dataset_files)

    rich.print(f"[green]✓ starting server at http://{host}:{port}[/green]")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    fire.Fire(main)