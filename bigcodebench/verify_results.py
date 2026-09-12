# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, TaskState
from inspect_ai.solver import solver as solver_decorator

# Import TaskManager to properly generate tasks with metadata
from src.cot_red_handed.tasks.tasks import TaskManager

# Add cot_red_handed to Python path
# COT_RED_PATH = Path("/home/hanw14/codebase/cot_red/src")
# if COT_RED_PATH.exists() and str(COT_RED_PATH) not in sys.path:
# sys.path.insert(0, str(COT_RED_PATH))


# check /home/hanw14/codebase/cot_red/src/cot_red_handed/tasks/solvers.py for orignal implementaiotn
@solver_decorator
def pregenerated_solver(response_map: dict[str, str], verifier, verifier_params: dict):
    """
    Solver that returns pre-generated responses and calls the verifier.

    This allows us to use inspect_ai's verification framework with vLLM results.

    Args:
        response_map: Dictionary mapping sample IDs to their responses
        verifier: The verifier function to call
        verifier_params: Parameters to pass to the verifier
    """

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        # Get the response for this sample using sample ID
        sample_id = state.sample_id

        if sample_id in response_map:
            response = response_map[sample_id]
        else:
            response = f"No response available for sample {sample_id}"

        # Create a ModelOutput with the pre-generated response
        state.output = ModelOutput(
            model="vllm-pregenerated",
            completion=response,
            stop_reason="stop",
            message=ChatMessageAssistant(content=response),
        )

        # Call the verifier to get the score
        temp_score = await verifier(state, state.target, **verifier_params)
        state.metadata["score"] = temp_score

        return state

    return solve


def load_vllm_results(results_path: str) -> list[dict[str, Any]]:
    """Load all vLLM inference results from JSONL shard files.

    Args:
        results_path: Path to directory containing shard_*.jsonl files

    Returns:
        List of records containing responses and metadata
    """
    results_dir = Path(results_path)
    jsonl_paths = sorted(results_dir.glob("shard_*.jsonl"))

    if not jsonl_paths:
        raise FileNotFoundError(f"No shard files found in {results_dir}")

    records = []
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {len(jsonl_paths)} shards")
    return records


def create_verification_task(
    records: list[dict[str, Any]],
    side_task: str,
    rollout_index: int = 0,
    dataset: str = "bigcodebench",
    attack_policy: str = "baseline",
    id_set: str = "TRAIN",
    use_pairings: bool = True,
) -> tuple[Task, dict[str, str]]:
    """Create an inspect_ai Task using TaskManager with pregenerated responses.

    Args:
        records: List of vLLM inference records
        side_task: Name of side task for verification
        dataset: Dataset name (e.g., 'bigcodebench')
        attack_policy: Attack policy used during inference
        id_set: Dataset split (TRAIN/TEST/VAL)

    Returns:
        Tuple of (Task with proper configuration, response_map)
    """
    # Warn if attack_policy doesn't match the suffixes present in the results.
    suffixes = set()
    for record in records:
        sample_id = record["id"]
        if "_" in sample_id:
            suffixes.add(sample_id.split("_", 1)[1])
    if suffixes and attack_policy not in suffixes:
        print(
            f"Warning: attack_policy '{attack_policy}' not found in results suffixes: "
            f"{', '.join(sorted(suffixes))}"
        )

    # Build response map from records: sample_id -> response
    response_map = {}
    for record in records:
        sample_id = record["id"]
        responses = record["response"]

        # Extract base task ID (remove attack policy suffix if present)
        # e.g., "149_baseline" -> "149"
        base_id = sample_id.split("_", 1)[0] if "_" in sample_id else sample_id

        if rollout_index < len(responses):
            response = responses[rollout_index]
        else:
            raise IndexError(
                f"Rollout index {rollout_index} out of range for sample {sample_id} with {len(responses)} responses"
            )

        if sample_id:
            response_map[sample_id] = response

        # Keep a base-id entry as a fallback for older logs that used stripped ids.
        if base_id and base_id not in response_map:
            response_map[base_id] = response

    # Use TaskManager to generate task with proper metadata
    task_manager = TaskManager()
    task = task_manager.generate_task(
        main_task_name=dataset,
        side_task_name=side_task,
        attack_policies=[attack_policy],
        id_set=id_set,
        use_pairings=use_pairings,
    )

    # Get verifier from side task config
    from src.cot_red_handed.tasks.side_task_config import (
        SIDE_TASK_CONFIGS,
        get_side_task_config,
    )

    side_task_config = get_side_task_config(side_task) if side_task else {}
    verifier = side_task_config.get(
        "verifier", SIDE_TASK_CONFIGS["baseline"]["verifier"]
    )

    # Prepare verifier parameters
    verifier_params = {}
    if "test_provided" in side_task_config:
        verifier_params["test_provided"] = side_task_config["test_provided"]
    if "monitor_prompt" in side_task_config:
        verifier_params["monitor_prompt"] = side_task_config["monitor_prompt"]

    # Replace the task's solver with our pregenerated solver
    # Keep the original scorer (pass_scorer) which wraps the verifier
    task.solver = pregenerated_solver(response_map, verifier, verifier_params)

    return task, response_map


def verify_vllm_results(args):
    records = load_vllm_results(args.results_path)
    task, response_map = create_verification_task(
        records=records,
        side_task=args.side_tasks,
        rollout_index=args.rollout_index,
        dataset=args.dataset,
        attack_policy=args.attack_policy,
        id_set=args.bcb_id_set,
        use_pairings=args.bcb_use_pairings,
    )

    # Determine output directory
    if args.output_dir is None:
        results_parent = Path(args.results_path).parent
        args.output_dir = str(
            results_parent
            / f"verification_logs_{args.side_tasks}/rollout_{args.rollout_index}"
        )

    eval(
        task,
        model="mockllm/model",  # Dummy model since we use pregenerated responses
        log_dir=args.output_dir,
        max_connections=args.max_connections,
        max_sandboxes=args.max_sandboxes,
    )
