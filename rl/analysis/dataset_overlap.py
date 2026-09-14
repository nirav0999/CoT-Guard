# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import fire
import rich

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SampleKey = tuple[str, str, str]


def parse_train_task_id(task_id: str, side_task: str, attack_policy: str) -> SampleKey:
    parts = task_id.split(":")
    assert len(parts) == 5, f"train task_id must have 5 colon parts: {task_id}"
    assert (
        parts[0] == side_task
    ), f"train task_id side prefix {parts[0]!r} != side_task {side_task!r}: {task_id}"
    main_attack = parts[1]
    suffix = "_" + attack_policy
    assert main_attack.endswith(suffix), (
        f"train task_id main_attack {main_attack!r} does not end with "
        f"{suffix!r}: {task_id}"
    )
    main_task_id = main_attack[: -len(suffix)]
    return (side_task, main_task_id, attack_policy)


def load_train_keys_by_side(
    train_path: str, attack_policy: str
) -> dict[str, set[SampleKey]]:
    assert train_path.endswith(".jsonl"), train_path
    assert os.path.isfile(train_path), train_path

    by_side: dict[str, set[SampleKey]] = {}
    n_rows = 0
    n_malicious = 0
    n_benign = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            n_rows += 1
            rec = json.loads(line)
            assert "extra_info" in rec, f"{train_path}:{line_no} missing extra_info"
            assert "task_id" in rec, f"{train_path}:{line_no} missing task_id"
            extra = rec["extra_info"]
            assert (
                "benign" in extra and "side_task" in extra
            ), f"{train_path}:{line_no} extra_info missing benign/side_task"
            if extra["benign"]:
                n_benign += 1
                continue
            n_malicious += 1
            side_task = extra["side_task"]
            key = parse_train_task_id(rec["task_id"], side_task, attack_policy)
            by_side.setdefault(side_task, set()).add(key)

    rich.print(
        f"[dim]  train rows: {n_rows} "
        f"(benign={n_benign}, malicious={n_malicious})[/dim]"
    )
    return by_side


def eval_dir_for_side(
    results_root: str,
    main_task: str,
    side_task: str,
    attack_policy: str,
    evaluated_model: str,
    rollout_num: int,
) -> str:
    return os.path.join(
        results_root,
        f"{main_task}.{side_task}.{attack_policy}",
        f"{evaluated_model}--{rollout_num}",
    )


def load_eval_keys_for_side(eval_dir: str) -> set[SampleKey]:
    assert os.path.isdir(eval_dir), eval_dir
    keys: set[SampleKey] = set()
    file_names = sorted(fn for fn in os.listdir(eval_dir) if fn.endswith(".jsonl"))
    assert file_names, f"no .jsonl files in {eval_dir}"

    n_rows = 0
    for fn in file_names:
        path = os.path.join(eval_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                n_rows += 1
                rec = json.loads(line)
                assert "metadata" in rec, f"{path}:{line_no} missing metadata"
                md = rec["metadata"]
                assert (
                    "side_task_name" in md
                    and "main_task_id" in md
                    and "attack_policy" in md
                ), f"{path}:{line_no} metadata missing required fields"
                keys.add(
                    (
                        str(md["side_task_name"]),
                        str(md["main_task_id"]),
                        str(md["attack_policy"]),
                    )
                )

    rich.print(
        f"[dim]  eval rows pooled across {len(file_names)} files: {n_rows} "
        f"→ distinct keys: {len(keys)}[/dim]"
    )
    return keys


def print_overlap_row(
    side_task: str,
    train_keys: set[SampleKey],
    eval_keys: set[SampleKey],
) -> tuple[int, int, int]:
    overlap = train_keys & eval_keys
    n_overlap = len(overlap)
    n_eval = len(eval_keys)
    n_train = len(train_keys)

    if n_eval == 0:
        eval_frac = "n/a (eval=0)"
    else:
        eval_frac = f"{n_overlap} / {n_eval} = {n_overlap / n_eval * 100:.1f}%"
    if n_train == 0:
        train_frac = "n/a (train=0)"
    else:
        train_frac = f"{n_overlap} / {n_train} = {n_overlap / n_train * 100:.1f}%"

    rich.print(
        f"[yellow]  {side_task:<14} "
        f"overlap/eval = {eval_frac}  |  "
        f"overlap/train = {train_frac}[/]"
    )
    return n_overlap, n_eval, n_train


def main(
    train_path: str,
    main_task: str,
    side_tasks: list[str],
    evaluated_model: str = "Qwen3-14B",
    rollout_num: int = 1,
    attack_policy: str = "baseline",
    results_root: str = "/srv/local/hanw14/icml2026/coding/clean/results/monitor",
) -> None:
    train_path = os.path.abspath(os.path.expanduser(train_path))
    results_root = os.path.abspath(os.path.expanduser(results_root))
    assert os.path.isfile(train_path), train_path
    assert os.path.isdir(results_root), results_root
    assert isinstance(side_tasks, list) and side_tasks, side_tasks

    rich.print("[magenta bold]═══ Dataset Overlap ═══[/magenta bold]")
    rich.print(f"[blue]→ train_path = {train_path}[/blue]")
    rich.print(f"[blue]→ results_root = {results_root}[/blue]")
    rich.print(f"[blue]→ main_task = {main_task}[/blue]")
    rich.print(f"[blue]→ side_tasks = {side_tasks}[/blue]")
    rich.print(f"[blue]→ evaluated_model = {evaluated_model}[/blue]")
    rich.print(f"[blue]→ rollout_num = {rollout_num}[/blue]")
    rich.print(f"[blue]→ attack_policy = {attack_policy}[/blue]")

    rich.print("[cyan bold underline]📊 loading train keys[/]")
    train_keys_by_side = load_train_keys_by_side(train_path, attack_policy)
    for st in sorted(train_keys_by_side):
        rich.print(
            f"[dim italic]  train[{st}] distinct keys: "
            f"{len(train_keys_by_side[st])}[/]"
        )

    rich.print("[cyan bold underline]📊 per side_task overlap[/]")
    totals_overlap = 0
    totals_eval = 0
    totals_train = 0
    for side_task in side_tasks:
        eval_dir = eval_dir_for_side(
            results_root=results_root,
            main_task=main_task,
            side_task=side_task,
            attack_policy=attack_policy,
            evaluated_model=evaluated_model,
            rollout_num=rollout_num,
        )
        rich.print(f"[blue]→ eval_dir = {eval_dir}[/blue]")
        eval_keys = load_eval_keys_for_side(eval_dir)
        train_keys = train_keys_by_side.setdefault(side_task, set())
        n_overlap, n_eval, n_train = print_overlap_row(
            side_task=side_task, train_keys=train_keys, eval_keys=eval_keys
        )
        totals_overlap += n_overlap
        totals_eval += n_eval
        totals_train += n_train

    rich.print("[magenta bold]═══ totals ═══[/magenta bold]")
    if totals_eval > 0:
        rich.print(
            f"[yellow]  overlap / eval_total = {totals_overlap} / "
            f"{totals_eval} = {totals_overlap / totals_eval * 100:.1f}%[/]"
        )
    if totals_train > 0:
        rich.print(
            f"[yellow]  overlap / train_total = {totals_overlap} / "
            f"{totals_train} = {totals_overlap / totals_train * 100:.1f}%[/]"
        )
    if totals_overlap == 0:
        rich.print("[green]✓ no overlap found[/green]")
    else:
        rich.print(f"[orange3]⚠ {totals_overlap} overlapping sample keys[/]")


if __name__ == "__main__":
    fire.Fire(main)
