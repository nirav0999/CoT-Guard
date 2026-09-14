# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import sys

import fire
import rich

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.analysis.aggregate import collect_split_step_files
from utils import load_jsonl, save_jsonl


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def discover_rollout_files(
    run_dir: str, include_val: bool
) -> list[tuple[int, str, str]]:
    assert os.path.isdir(run_dir), run_dir
    base = os.path.basename(os.path.normpath(run_dir))
    if base in {"rollout_logs", "val_logs"}:
        scan_dir = os.path.dirname(os.path.normpath(run_dir))
        rich.print(
            f"[dim]run_dir is a split subdir ({base}); scanning parent {scan_dir}[/dim]"
        )
    else:
        scan_dir = run_dir
    split_step_files = collect_split_step_files(scan_dir)
    items: list[tuple[int, str, str]] = []
    for step, path in split_step_files["train"]:
        items.append((step, "train", path))
    if include_val:
        for step, path in split_step_files["val"]:
            items.append((step, "val", path))
    items.sort(key=lambda x: (x[1], x[0]))
    assert items, f"no rollout files discovered under {scan_dir}"
    rich.print(
        f"[green]✓ discovered {len(items)} rollout files under {scan_dir} "
        f"(train={len(split_step_files['train'])}, val={len(split_step_files['val']) if include_val else 0})[/green]"
    )
    return items


def rollout_user_prompt(row: dict) -> str:
    assert "input" in row, f"rollout row missing 'input': {list(row.keys())}"
    s = row["input"]
    assert s.startswith(
        "user\n"
    ), f"expected input to start with 'user\\n', got: {s[:20]!r}"
    assert s.endswith(
        "\nassistant\n"
    ), f"expected input to end with '\\nassistant\\n', got: {s[-20:]!r}"
    return s[len("user\n") : -len("\nassistant\n")]


def build_dataset_lookup(dataset_path: str) -> dict[str, dict]:
    assert dataset_path.endswith(".jsonl"), dataset_path
    assert os.path.isfile(dataset_path), dataset_path
    rows = load_jsonl(dataset_path)
    lookup: dict[str, dict] = {}
    n_collisions = 0
    for row in rows:
        assert "messages" in row and "task_id" in row and "extra_info" in row, list(
            row.keys()
        )
        user_prompt = row["messages"][0]["content"]
        h = content_hash(user_prompt)
        task_id = row["task_id"]
        parts = task_id.split(":")
        assert len(parts) == 5, f"expected 5 colon parts, got {len(parts)}: {task_id}"
        side_task = parts[0]
        main_task_id = parts[1].split("_")[0]
        extra = row["extra_info"]
        assert "side_task" in extra and "benign" in extra, list(extra.keys())
        assert side_task == extra["side_task"], f"{side_task} != {extra['side_task']}"
        if h in lookup:
            n_collisions += 1
            rich.print(
                f"[orange3]⚠ prompt-hash collision {h}: keeping {lookup[h]['task_id']}, skipping {task_id}[/orange3]"
            )
            continue
        lookup[h] = {
            "task_id": task_id,
            "main_task_id": main_task_id,
            "side_task": side_task,
            "benign": bool(extra["benign"]),
            "user_prompt": user_prompt,
        }
    rich.print(
        f"[green]✓ dataset lookup: {len(lookup)} unique prompt hashes "
        f"({n_collisions} collisions skipped) from {dataset_path}[/green]"
    )
    return lookup


def merge_rollouts(
    rollout_files: list[tuple[int, str, str]],
    dataset_lookup: dict[str, dict],
) -> list[dict]:
    merged: list[dict] = []
    total_rows = 0
    for step_no, split, path in rollout_files:
        assert path.endswith(".jsonl"), path
        assert os.path.isfile(path), path
        rows = load_jsonl(path)
        total_rows += len(rows)
        for row in rows:
            user_prompt = rollout_user_prompt(row)
            h = content_hash(user_prompt)
            assert (
                h in dataset_lookup
            ), f"no dataset match for rollout prompt (hash={h}) at {path}"
            meta = dataset_lookup[h]
            assert "output" in row, f"rollout row missing 'output': {list(row.keys())}"
            record = {
                "task_id": f"{meta['task_id']}:step{step_no}",
                "step_no": step_no,
                "split": split,
                "rollout_path": path,
                "main_task_id": meta["main_task_id"],
                "side_task": meta["side_task"],
                "benign": meta["benign"],
                "user_prompt": user_prompt,
                "rollout_output": row["output"],
            }
            if "data_source" in row:
                record["rollout_data_source"] = row["data_source"]
            if "is_benign" in row:
                record["rollout_is_benign"] = float(row["is_benign"]) == 1.0
            if "reward" in row:
                record["rollout_reward"] = float(row["reward"])
            elif "score" in row:
                record["rollout_reward"] = float(row["score"])
            if "monitor_guess" in row:
                record["rollout_monitor_guess"] = row["monitor_guess"]
            merged.append(record)
        rich.print(
            f"[dim]  {split} step {step_no}: {len(rows)} rollouts from {path}[/dim]"
        )
    rich.print(
        f"[green]✓ merged: {len(merged)} / {total_rows} rollouts "
        f"across {len(rollout_files)} files[/green]"
    )
    return merged


def eval_dir_for(
    eval_root: str,
    main_task: str,
    side_task: str,
    benign: bool,
    attack_policy: str,
    evaluated_model: str,
    rollout_num: int,
) -> str:
    policy_seg = "no_side_task" if benign else attack_policy
    return f"{eval_root}/{main_task}.{side_task}.{policy_seg}/{evaluated_model}--{rollout_num}"


def build_eval_cache(eval_dir: str) -> dict[tuple[str, str], dict[str, dict]]:
    assert os.path.isdir(eval_dir), eval_dir
    file_names = sorted(fn for fn in os.listdir(eval_dir) if fn.endswith(".jsonl"))
    assert file_names, f"no .jsonl files in {eval_dir}"
    cache: dict[tuple[str, str], dict[str, dict]] = {}
    n_rows = 0
    for fn in file_names:
        path = f"{eval_dir}/{fn}"
        rows = load_jsonl(path)
        n_rows += len(rows)
        for row in rows:
            assert "messages" in row and "metadata" in row, list(row.keys())
            md = row["metadata"]
            assert "main_task_id" in md and "side_task_name" in md, list(md.keys())
            user_prompt = row["messages"][0]["content"]
            h = content_hash(user_prompt)
            key = (h, str(md["main_task_id"]))
            assert (
                row["messages"][-1]["role"] == "assistant"
            ), f"expected last message role=assistant in {path}, got {row['messages'][-1]['role']}"
            cache.setdefault(key, {})[fn] = {
                "eval_path": path,
                "eval_task_id": row["task_id"],
                "eval_response": row["messages"][-1]["content"],
                "eval_metadata": md,
            }
    rich.print(
        f"[dim]  indexed {n_rows} eval rows → {len(cache)} distinct (prompt, main_task_id) keys "
        f"across {len(file_names)} files in {eval_dir}[/dim]"
    )
    return cache


def attach_eval_matches(
    merged_rollouts: list[dict],
    eval_root: str,
    main_task: str,
    attack_policy: str,
    evaluated_model: str,
    rollout_num: int,
) -> list[dict]:
    caches: dict[tuple[str, bool], dict[tuple[str, str], dict[str, dict]]] = {}
    n_matched = 0
    unmatched_preview: list[str] = []
    for rec in merged_rollouts:
        cache_key = (rec["side_task"], rec["benign"])
        if cache_key not in caches:
            edir = eval_dir_for(
                eval_root=eval_root,
                main_task=main_task,
                side_task=rec["side_task"],
                benign=rec["benign"],
                attack_policy=attack_policy,
                evaluated_model=evaluated_model,
                rollout_num=rollout_num,
            )
            rich.print(f"[blue]→ loading eval cache: {edir}[/blue]")
            caches[cache_key] = build_eval_cache(edir)
        cache = caches[cache_key]
        lookup_key = (content_hash(rec["user_prompt"]), rec["main_task_id"])
        if lookup_key in cache:
            rec["eval_matches"] = cache[lookup_key]
            n_matched += 1
        else:
            rec["eval_matches"] = None
            if len(unmatched_preview) < 5:
                unmatched_preview.append(
                    f"side={rec['side_task']} main_task_id={rec['main_task_id']} benign={rec['benign']} step={rec['step_no']}"
                )
    total = len(merged_rollouts)
    rich.print(
        f"[yellow]eval matches: {n_matched} / {total} = {n_matched / total * 100:.1f}%[/yellow]"
    )
    if n_matched < total:
        rich.print(f"[orange3]⚠ {total - n_matched} unmatched rollouts[/orange3]")
        for line in unmatched_preview:
            rich.print(f"[orange3]  · {line}[/orange3]")
    return merged_rollouts


def main(
    run_dir: str,
    dataset_path: str,
    eval_root: str,
    output_path: str,
    main_task: str,
    attack_policy: str = "baseline",
    evaluated_model: str = "Qwen3-14B",
    rollout_num: int = 1,
    include_val: bool = True,
) -> None:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    eval_root = os.path.abspath(os.path.expanduser(eval_root))
    output_path = os.path.abspath(os.path.expanduser(output_path))
    assert output_path.endswith(".jsonl"), output_path
    assert os.path.isdir(eval_root), eval_root
    assert os.path.isdir(run_dir), run_dir

    rich.print("[magenta bold]═══ Compare Rollouts With Eval ═══[/magenta bold]")
    rich.print(f"[blue]→ run_dir = {run_dir}[/blue]")
    rich.print(f"[blue]→ dataset_path = {dataset_path}[/blue]")
    rich.print(f"[blue]→ eval_root = {eval_root}[/blue]")
    rich.print(f"[blue]→ output_path = {output_path}[/blue]")
    rich.print(f"[blue]→ main_task = {main_task}[/blue]")
    rich.print(f"[blue]→ attack_policy = {attack_policy}[/blue]")
    rich.print(f"[blue]→ evaluated_model = {evaluated_model}[/blue]")
    rich.print(f"[blue]→ rollout_num = {rollout_num}[/blue]")
    rich.print(f"[blue]→ include_val = {include_val}[/blue]")

    rich.print("[cyan bold underline]📊 discovering rollout files[/]")
    rollout_files = discover_rollout_files(run_dir, include_val=include_val)

    rich.print("[cyan bold underline]📊 building dataset lookup[/]")
    dataset_lookup = build_dataset_lookup(dataset_path)

    rich.print("[cyan bold underline]📊 merging rollouts[/]")
    merged_rollouts = merge_rollouts(rollout_files, dataset_lookup)

    rich.print("[cyan bold underline]📊 attaching eval matches[/]")
    merged_rollouts = attach_eval_matches(
        merged_rollouts,
        eval_root=eval_root,
        main_task=main_task,
        attack_policy=attack_policy,
        evaluated_model=evaluated_model,
        rollout_num=rollout_num,
    )

    save_jsonl(merged_rollouts, output_path)
    rich.print(
        f"[green]✓ wrote {len(merged_rollouts)} records to {output_path}[/green]"
    )


if __name__ == "__main__":
    fire.Fire(main)
