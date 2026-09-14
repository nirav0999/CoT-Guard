import copy
import os
import subprocess
from datetime import datetime
from typing import Any

import fire
import rich
from filelock import FileLock

from utils import (
    load_jsonl,
    save_jsonl,
)

from config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_ROLLOUTS,
    DEFAULT_TEMP,
    TEMP_DIRECTORY,
)


def get_completed_ids(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    data = load_jsonl(output_path)
    return {row["task_id"] for row in data}


def filter_pending_queries(
    input_paths: list[str],
    output_paths: list[str],
    num_rollouts: int,
) -> list[dict[str, Any]]:
    pending = []

    for file_idx, (input_path, output_path) in enumerate(
        zip(input_paths, output_paths, strict=False)
    ):
        completed_ids = get_completed_ids(output_path)

        data = load_jsonl(input_path)

        for row in data:
            task_id = row["task_id"]
            existing_rollouts = sum(
                1 for cid in completed_ids if ":".join(cid.split(":")[:-1]) == task_id
            )
            needed_rollouts = num_rollouts - existing_rollouts

            if needed_rollouts > 0:
                row_copy = copy.deepcopy(row)
                row_copy["metadata"] = row.get("metadata", {}).copy()
                row_copy["metadata"]["_source_file_idx"] = file_idx
                row_copy["metadata"]["_source_file_path"] = input_path
                row_copy["metadata"]["_output_path"] = output_path
                row_copy["metadata"]["_existing_rollouts"] = existing_rollouts
                row_copy["metadata"]["_needed_rollouts"] = needed_rollouts
                pending.append(row_copy)

    return pending


def estimate_query_cost(query: dict[str, Any]) -> int:
    return sum(len(str(m.get("content", ""))) for m in query["messages"])


def create_shards(
    queries: list[dict[str, Any]],
    num_shards: int,
    shard_dir: str,
    balance_by_length: bool = False,
) -> list[str]:
    os.makedirs(shard_dir, exist_ok=True)

    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]

    if balance_by_length:
        shard_costs = [0] * num_shards
        sorted_queries = sorted(queries, key=estimate_query_cost, reverse=True)
        for query in sorted_queries:
            min_idx = min(range(num_shards), key=lambda i: shard_costs[i])
            shards[min_idx].append(query)
            shard_costs[min_idx] += estimate_query_cost(query)
    else:
        for i, query in enumerate(queries):
            shards[i % num_shards].append(query)

    shard_paths = []
    for i, shard in enumerate(shards):
        if shard:
            shard_path = f"{shard_dir}/shard_{i}.jsonl"
            save_jsonl(shard, shard_path)
            shard_paths.append(shard_path)

            total_cost = sum(estimate_query_cost(q) for q in shard)
            avg_cost = total_cost / len(shard)
            rich.print(
                f"[dim]shard_{i}: {len(shard)} queries, total={total_cost}, avg={avg_cost:.1f}[/dim]"
            )

    return shard_paths


def merge_single_shard(shard_output_path: str) -> None:
    if not os.path.exists(shard_output_path):
        return

    shard_data = load_jsonl(shard_output_path)
    results_by_output: dict[str, list[dict]] = {}

    for row in shard_data:
        output_path = row["metadata"]["_output_path"]

        task_id = row["task_id"]
        parts = task_id.split(":")
        try:
            rollout_idx = int(parts[-1])
            base_id = ":".join(parts[:-1])
        except ValueError:
            base_id = task_id
            rollout_idx = 0

        existing_rollouts = row["metadata"]["_existing_rollouts"]
        needed_rollouts = row["metadata"]["_needed_rollouts"]

        if rollout_idx >= needed_rollouts:
            continue

        new_rollout_idx = existing_rollouts + rollout_idx
        new_task_id = f"{base_id}:{new_rollout_idx}"

        clean_row = {
            "task_id": new_task_id,
            "messages": row["messages"],
            "metadata": {
                k: v for k, v in row["metadata"].items() if not k.startswith("_")
            },
        }

        if output_path not in results_by_output:
            results_by_output[output_path] = []
        results_by_output[output_path].append(clean_row)

    for output_path, new_results in results_by_output.items():
        if not new_results:
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lock_path = f"{output_path}.lock"
        with FileLock(lock_path):
            if os.path.exists(output_path):
                existing_data = load_jsonl(output_path)
                all_results = existing_data + new_results
            else:
                all_results = new_results

            all_task_ids = [row["task_id"] for row in all_results]
            assert len(all_task_ids) == len(
                set(all_task_ids)
            ), f"Duplicate task_ids in {output_path}: {[tid for tid in all_task_ids if all_task_ids.count(tid) > 1]}"

            save_jsonl(all_results, output_path)


def run_single_shard(
    shard_idx: int,
    shard_path: str,
    shard_output_path: str,
    log_path: str,
    model: str,
    devices: str,
    tp: int,
    temp: float,
    max_tokens: int,
    bs: int,
    scaling_factor: float | None = None,
) -> tuple[int, int]:
    max_needed = max(
        row["metadata"]["_needed_rollouts"] for row in load_jsonl(shard_path)
    )

    if isinstance(devices, tuple):
        devices = ",".join(map(str, devices))

    cmd = [
        "python3",
        "-m",
        "gen.generate",
        "--input_path",
        shard_path,
        "--output_path",
        shard_output_path,
        "--model",
        model,
        "--backend",
        "vllm",
        "--temperature",
        str(temp),
        "--max_tokens",
        str(max_tokens),
        "--n",
        str(max_needed),
        "--tp",
        str(tp),
        "--bs",
        str(bs),
        "--devices",
        devices,
    ]

    if scaling_factor is not None:
        cmd.extend(["--scaling_factor", str(scaling_factor)])

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    return shard_idx, result.returncode


def run_parallel_generation(
    shard_paths: list[str],
    shard_dir: str,
    model: str,
    devices_list: list[str],
    tp: int,
    temp: float,
    max_tokens: int,
    bs: int,
    scaling_factor: float | None = None,
) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_shards = len(shard_paths)
    shard_output_paths = [
        f"{shard_dir}/shard_{i}.output.jsonl" for i in range(num_shards)
    ]
    log_paths = [f"{shard_dir}/shard_{i}.log" for i in range(num_shards)]

    rich.print(f"[yellow]Launching {num_shards} shards in parallel...[/yellow]")
    rich.print(f"[dim]Logs: {shard_dir}/shard_*.log[/dim]")
    rich.print()

    with ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = {
            executor.submit(
                run_single_shard,
                i,
                shard_paths[i],
                shard_output_paths[i],
                log_paths[i],
                model,
                devices_list[i],
                tp,
                temp,
                max_tokens,
                bs,
                scaling_factor,
            ): i
            for i in range(num_shards)
        }

        for future in as_completed(futures):
            shard_idx, exit_code = future.result()
            if exit_code != 0:
                rich.print(
                    f"[dark_red bold]✗ Shard {shard_idx} failed (exit {exit_code}) → see {log_paths[shard_idx]}[/dark_red bold]"
                )
            else:
                rich.print(f"[green]✓ Shard {shard_idx} completed, merging...[/green]")
                merge_single_shard(shard_output_paths[shard_idx])

    return shard_output_paths


def parallel_generate(
    input_paths: str | list[str],
    output_paths: str | list[str],
    model: str = "Qwen/Qwen3-14B",
    devices: str = "0,1,2,3,4,5,6,7,8",
    tp: int = 2,
    temp: float = DEFAULT_TEMP,
    num_rollouts: int = DEFAULT_ROLLOUTS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    bs: int = 128,
    scaling_factor: float | None = None,
    balance_by_length: bool = False,
) -> list[str]:

    if model == "Qwen/Qwen3-32B":  # use more tp for larger model
        rich.print(f"[blue]Changing tp=2 from tp={tp} for Qwen3-32B[/blue]")
        tp = max(2, tp)

    if isinstance(input_paths, str):
        input_paths = input_paths.split(",")
    if isinstance(output_paths, str):
        output_paths = output_paths.split(",")
    if isinstance(devices, tuple):
        devices = ",".join(map(str, devices))

    if len(input_paths) != len(output_paths):
        raise ValueError(
            f"len(input_paths)={len(input_paths)} != len(output_paths)={len(output_paths)}"
        )

    gpu_list = [g.strip() for g in str(devices).split(",")]
    rich.print(f"[dim]Using GPUs: {gpu_list}[/dim]")

    num_instances = len(gpu_list) // tp

    if len(gpu_list) % tp != 0:
        raise ValueError(
            f"len(devices) ({len(gpu_list)}) must be divisible by tp ({tp})"
        )

    devices_list = [
        ",".join(gpu_list[i * tp : (i + 1) * tp]) for i in range(num_instances)
    ]

    rich.print(f"[magenta bold]{'=' * 60}[/magenta bold]")
    rich.print("[cyan bold underline]gen/generate_parallel.py[/cyan bold underline]")
    rich.print(f"[dim]model: {model}[/dim]")
    rich.print(f"[dim]devices: {devices}, tp: {tp} → {num_instances} instances[/dim]")
    rich.print(f"[dim]input_paths: {len(input_paths)} files[/dim]")
    rich.print(f"[dim]num_rollouts: {num_rollouts}[/dim]")
    rich.print(f"[magenta bold]{'=' * 60}[/magenta bold]")
    rich.print()

    pending_queries = filter_pending_queries(input_paths, output_paths, num_rollouts)

    if not pending_queries:
        rich.print("[green]✓ All queries complete, nothing to do[/green]")
        return output_paths

    total_needed = sum(q["metadata"]["_needed_rollouts"] for q in pending_queries)
    rich.print(
        f"[yellow]Pending: {len(pending_queries)} tasks, {total_needed} total rollouts needed[/yellow]"
    )

    safe_model = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_dir = f"{TEMP_DIRECTORY}/logs/shards_parallel_generation/{safe_model}_{temp:.2f}_{timestamp}"

    actual_instances = min(num_instances, len(pending_queries))
    shard_paths = create_shards(
        pending_queries, actual_instances, shard_dir, balance_by_length
    )

    rich.print(f"[dim]Created {len(shard_paths)} shards in {shard_dir}[/dim]")

    run_parallel_generation(
        shard_paths=shard_paths,
        shard_dir=shard_dir,
        model=model,
        devices_list=devices_list[: len(shard_paths)],
        tp=tp,
        temp=temp,
        max_tokens=max_tokens,
        bs=bs,
        scaling_factor=scaling_factor,
    )

    rich.print(f"[dim]Logs preserved at: {shard_dir}[/dim]")

    rich.print()
    rich.print("[green]✓ Done! Outputs:[/green]")
    for p in output_paths:
        rich.print(f"[green]  {p}[/green]")

    return output_paths


if __name__ == "__main__":
    fire.Fire(parallel_generate)
