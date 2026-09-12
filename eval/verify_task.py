# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import rich
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log

from bigcodebench.verify_results import pregenerated_solver
from config import GLOBAL_DIRECTORY, TEMP_DIRECTORY
from src.cot_red_handed.tasks.side_task_config import (
    SIDE_TASK_CONFIGS,
    get_side_task_config,
)
from src.cot_red_handed.tasks.verifiers import pass_scorer
from utils import load_jsonl, save_jsonl

MAX_CONNECTIONS = 128
MAX_SANDBOXES = 128


def build_dataset(records: list[dict[str, Any]]) -> MemoryDataset:
    samples = []
    for record in records:
        metadata = record["metadata"].copy()
        metadata["side_test_case"] = metadata["side_test"]
        metadata["code"] = metadata["code_prompt"]
        metadata["canonical_solution_length"] = len(
            metadata["canonical_main_sol"].strip().splitlines()
        )
        samples.append(
            Sample(
                id=record["task_id"],
                input=record["messages"][0]["content"],
                target=metadata["main_test"],
                metadata=metadata,
            )
        )
    return MemoryDataset(samples)


def build_response_map(records: list[dict[str, Any]]) -> dict[str, str]:
    response_map = {}
    for record in records:
        messages = record["messages"]
        assert (
            messages[-1]["role"] == "assistant"
        ), f"Expected an assistant response for task_id={record['task_id']}"
        response_map[record["task_id"]] = messages[-1]["content"]
    return response_map


def get_verifier(main_task: str, side_task: str):
    config = get_side_task_config(side_task)
    if main_task == "bigcodebench":
        verifier = config.get("verifier", SIDE_TASK_CONFIGS["baseline"]["verifier"])
    elif main_task == "kodcode":
        verifier = config.get(
            "verifier_pytest", SIDE_TASK_CONFIGS["baseline"]["verifier_pytest"]
        )
    else:
        raise ValueError(f"Unsupported main task: {main_task}")

    params = {
        key: config[key] for key in ("test_provided", "monitor_prompt") if key in config
    }
    return verifier, params


def create_verification_task(
    records: list[dict[str, Any]],
    main_task: str,
    side_task: str,
    attack_policy: str,
) -> Task:
    verifier, verifier_params = get_verifier(main_task, side_task)
    compose_path = (
        Path(__file__).parent.parent / "src/cot_red_handed/tasks/compose.yaml"
    )
    return Task(
        name=f"{main_task}.{side_task}.{attack_policy}",
        dataset=build_dataset(records),
        solver=pregenerated_solver(
            build_response_map(records), verifier, verifier_params
        ),
        scorer=pass_scorer(),
        sandbox=("docker", str(compose_path)),
    )


def read_results(log_dir: Path) -> dict[str, dict[str, Any]]:
    log_paths = list(log_dir.glob("*.eval"))
    assert log_paths, f"Expected an .eval file in {log_dir}"
    eval_log = read_eval_log(max(log_paths, key=os.path.getmtime))

    results = {}
    for sample in eval_log.samples:
        score = (sample.scores or {}).get("pass_scorer")
        if score is None:
            results[sample.id] = {
                "main_pass": False,
                "side_pass": False,
                "verification_error": True,
            }
            continue

        metadata = score.metadata or {}
        results[sample.id] = {
            "main_pass": metadata.get("main_only", False),
            "side_pass": metadata.get("side_only", False),
            "verification_error": False,
        }
    return results


def print_summary(records: list[dict[str, Any]]) -> None:
    total = len(records)
    main_pass = sum(record["metadata"]["main_pass"] for record in records)
    side_pass = sum(record["metadata"]["side_pass"] for record in records)
    rich.print(f"Main pass: {main_pass}/{total}")
    rich.print(f"Side pass: {side_pass}/{total}")


def verify_task(
    main_task: str = "bigcodebench",
    side_task: str = "varname",
    attack_policy: str = "baseline",
    attack_model: str = "Qwen/Qwen3-8B",
    num_rollouts: int = 1,
    max_connections: int = MAX_CONNECTIONS,
    max_sandboxes: int = MAX_SANDBOXES,
) -> str:
    result_dir = f"{main_task}.{side_task}.{attack_policy}"
    model_name = attack_model.split("/")[-1]
    input_path = (
        Path(GLOBAL_DIRECTORY)
        / "results"
        / "attack"
        / result_dir
        / f"{model_name}.jsonl"
    )
    output_path = input_path.with_suffix(f".{num_rollouts}.tested.jsonl")

    if output_path.exists():
        records = load_jsonl(output_path)
        print_summary(records)
        return str(output_path)

    records = load_jsonl(input_path)
    task = create_verification_task(records, main_task, side_task, attack_policy)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        Path(TEMP_DIRECTORY)
        / "logs"
        / "verification"
        / f"{result_dir}.{model_name}.{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    eval(
        task,
        model="mockllm/model",
        log_dir=str(log_dir),
        max_connections=max_connections,
        max_sandboxes=max_sandboxes,
    )

    results = read_results(log_dir)
    for record in records:
        result = results[record["task_id"]]
        record["metadata"].update(result)

    save_jsonl(records, output_path)
    print_summary(records)
    return str(output_path)


if __name__ == "__main__":
    fire.Fire(verify_task)
