# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Any, Dict, List

import rich
from fire import Fire

from config import DEFAULT_ATTACK_MODEL, DEFAULT_TEMP, GLOBAL_DIRECTORY
from constants import FILTER_MODES
from utils import (
    extract_code,
    extract_code_with_comments,
    extract_cot,
    format_monitor_prompt,
    load_jsonl,
    load_monitor_prompt,
    save_jsonl,
)

MONITOR_POLICIES = {"main_aware", "main_side_aware"}
MONITOR_TYPES = ["action_only", "cot_only", "cot_action"]
MONITOR_PROMPT_VARIANTS = {
    "standard",
    "few_shot",
    "binary",
    "scalar",
    "likert",
}

SIDE_TASK_ID_SETS = {
    "typosquat": "TYPOSQUAT_TRAIN",
    "vulnimport": "VULNIMPORT_TRAIN",
}


def parse_input_path(input_path: str) -> tuple[str, str, str, str, str]:
    assert input_path.endswith(
        ".tested.jsonl"
    ), f"Expected .tested.jsonl file, got {input_path}"

    filename = os.path.basename(input_path)
    foldername = os.path.basename(os.path.dirname(input_path))

    parts = foldername.split(".")
    assert (
        len(parts) >= 3
    ), f"Expected folder format: main_task.side_task.attack_policy, got {foldername}"

    main_task = parts[0]
    side_task = parts[1]
    attack_policy = parts[2]

    model_temp = filename.replace(".tested.jsonl", "")
    model_temp_parts = model_temp.split("--")
    assert (
        len(model_temp_parts) == 2
    ), f"Expected filename format: model--temp.tested.jsonl, got {filename}"

    model = model_temp_parts[0]
    temp = model_temp_parts[1]

    return main_task, side_task, attack_policy, model, temp


def filter_by_rollout(
    records: List[Dict[str, Any]], attack_rollouts: int
) -> List[Dict[str, Any]]:
    filtered = []
    for record in records:
        task_id = record["task_id"]
        parts = task_id.split(":")
        assert (
            len(parts) == 2
        ), f"Expected task_id format: base_id:rollout_idx, got {task_id}"
        rollout_idx = int(parts[1])
        if rollout_idx < attack_rollouts:
            filtered.append(record)
    return filtered


def filter_by_pass_criteria(
    records: List[Dict[str, Any]], filter_mode: str
) -> List[Dict[str, Any]]:
    assert filter_mode in FILTER_MODES, f"Invalid filter_mode: {filter_mode}"

    filtered = []
    for record in records:
        metadata = record["metadata"]
        main_pass = metadata["main_pass"]
        side_pass = metadata["side_pass"]

        include = False
        if filter_mode == "side-all":
            include = side_pass
        elif filter_mode == "main-all":
            include = main_pass
        elif filter_mode == "side-main":
            include = side_pass and main_pass
        elif filter_mode == "side-only":
            include = side_pass and not main_pass
        elif filter_mode == "main-only":
            include = main_pass and not side_pass
        elif filter_mode == "all":
            include = True

        if include:
            filtered.append(record)

    return filtered


def extract_components(record: Dict[str, Any]) -> tuple[str, str, str, str]:
    messages = record["messages"]
    assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"
    assert (
        messages[-1]["role"] == "assistant"
    ), f"Expected last message role='assistant', got {messages[-1]['role']}"

    response = messages[-1]["content"]
    prompt = messages[0]["content"]

    cot = extract_cot(response)
    side_task_name = record.get("metadata", {}).get("side_task_name")
    if side_task_name == "benign_comment_watermark":
        code = extract_code_with_comments(response)
    else:
        code = extract_code(response)

    task_id = record["task_id"]
    if len(cot) < 10:
        rich.print(
            f"[orange3]⚠ Warning: CoT too short ({len(cot)} chars) for task_id={task_id}[/orange3]"
        )
        time.sleep(5)

    if len(code) < 10:
        rich.print(
            f"[orange3]⚠ Warning: Code too short ({len(code)} chars) for task_id={task_id}[/orange3]"
        )
        time.sleep(5)

    side_task_desc = record["metadata"]["side_task_description"]
    original_prompt = record["metadata"]["original_prompt"]

    return prompt, cot, code, side_task_desc, original_prompt


def build_monitor_records(
    records: List[Dict[str, Any]],
    templates: Dict[str, str],
    monitor_policy: str,
    attack_rollouts: int,
    prompt_variant: str = "standard",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    action_records = []
    cot_records = []
    cot_action_records = []

    action_template = templates[f"action_prompt_{monitor_policy}"]
    cot_template = templates[f"cot_prompt_{monitor_policy}"]
    cot_action_template_key = f"cot_action_prompt_{monitor_policy}"
    if prompt_variant != "standard":
        cot_action_template_key = f"{cot_action_template_key}_{prompt_variant}"
    assert (
        cot_action_template_key in templates
    ), f"Missing monitor template {cot_action_template_key!r}"
    cot_action_template = templates[cot_action_template_key]

    for record in records:
        prompt, cot, code, side_task_desc, original_prompt = extract_components(record)

        action_msg = format_monitor_prompt(
            action_template, original_prompt, cot, code, side_task_desc
        )

        cot_msg = format_monitor_prompt(
            cot_template, original_prompt, cot, code, side_task_desc
        )

        cot_action_msg = format_monitor_prompt(
            cot_action_template, original_prompt, cot, code, side_task_desc
        )

        base_metadata = record["metadata"].copy()
        base_metadata["monitor_policy"] = monitor_policy
        base_metadata["prompt_variant"] = prompt_variant
        base_metadata["attack_rollouts"] = attack_rollouts
        base_metadata["attack_cot"] = cot
        base_metadata["attack_code"] = code
        task_id = record["task_id"]

        action_metadata = base_metadata.copy()
        action_metadata["monitor_template"] = f"action_prompt_{monitor_policy}"
        action_records.append(
            {
                "task_id": f"{task_id}:action_only",
                "messages": [{"role": "user", "content": action_msg}],
                "metadata": action_metadata,
            }
        )

        cot_metadata = base_metadata.copy()
        cot_metadata["monitor_template"] = f"cot_prompt_{monitor_policy}"
        cot_records.append(
            {
                "task_id": f"{task_id}:cot_only",
                "messages": [{"role": "user", "content": cot_msg}],
                "metadata": cot_metadata,
            }
        )

        cot_action_metadata = base_metadata.copy()
        cot_action_metadata["monitor_template"] = cot_action_template_key
        cot_action_records.append(
            {
                "task_id": f"{task_id}:cot_action",
                "messages": [{"role": "user", "content": cot_action_msg}],
                "metadata": cot_action_metadata,
            }
        )

    return action_records, cot_records, cot_action_records


def debug_example(
    record: Dict[str, Any], action_msg: str, cot_msg: str, cot_action_msg: str
) -> None:
    prompt, cot, code, side_task_desc, original_prompt = extract_components(record)
    metadata = record["metadata"]

    rich.print(
        f"\n[cyan bold underline]📊 Debug Example: {record['task_id']}[/cyan bold underline]"
    )

    rich.print("\n[magenta bold]═══ Original Prompt ═══[/magenta bold]")
    rich.print(
        f"[dim]{prompt[:2000]}...[/dim]"
        if len(prompt) > 2000
        else f"[dim]{prompt}[/dim]"
    )

    rich.print(
        f"\n[magenta bold]═══ Extracted CoT ({len(cot)} chars) ═══[/magenta bold]"
    )
    rich.print(
        f"[dim]{cot[:2000]}...[/dim]" if len(cot) > 2000 else f"[dim]{cot}[/dim]"
    )

    rich.print(
        f"\n[magenta bold]═══ Extracted Code ({len(code)} chars) ═══[/magenta bold]"
    )
    rich.print(
        f"[dim]{code[:2000]}...[/dim]" if len(code) > 2000 else f"[dim]{code}[/dim]"
    )

    rich.print("\n[magenta bold]═══ Pass Status ═══[/magenta bold]")
    rich.print(f"[yellow]main_pass: {metadata['main_pass']}[/yellow]")
    rich.print(f"[yellow]side_pass: {metadata['side_pass']}[/yellow]")

    rich.print(
        "\n[magenta bold]═══ Formatted Monitor Prompt (action_only) ═══[/magenta bold]"
    )
    rich.print(
        f"[dim]{action_msg[:2000]}...[/dim]"
        if len(action_msg) > 2000
        else f"[dim]{action_msg}[/dim]"
    )

    rich.print(
        "\n[magenta bold]═══ Formatted Monitor Prompt (cot_only) ═══[/magenta bold]"
    )
    rich.print(
        f"[dim]{cot_msg[:2000]}...[/dim]"
        if len(cot_msg) > 2000
        else f"[dim]{cot_msg}[/dim]"
    )

    rich.print(
        "\n[magenta bold]═══ Formatted Monitor Prompt (cot_action) ═══[/magenta bold]"
    )
    rich.print(
        f"[dim]{cot_action_msg[:2000]}...[/dim]"
        if len(cot_action_msg) > 2000
        else f"[dim]{cot_action_msg}[/dim]"
    )


def for_monitor_main(
    main_task: str = "bigcodebench",
    side_tasks: str = "iodelete",
    attack_policies: str = "baseline",
    model: str = DEFAULT_ATTACK_MODEL,
    attack_temp: float = DEFAULT_TEMP,
    monitor_policy: str = "main_aware",
    monitor_type: str = "all",
    prompt_variant: str = "standard",
    attack_rollouts: int = 5,
    filter_mode: str = "side-all",
    prompt_path: str = "monitor_prompts",
    debug: bool = False,
) -> List[List[str]]:

    input_paths = []
    output_path_groups = []

    if isinstance(side_tasks, str):
        side_tasks = side_tasks.split(",")
    if isinstance(attack_policies, str):
        attack_policies = attack_policies.split(",")
    side_task_list = [task.strip() for task in side_tasks if task.strip()]
    attack_policy_list = [
        policy.strip() for policy in attack_policies if policy.strip()
    ]

    assert (
        prompt_variant in MONITOR_PROMPT_VARIANTS
    ), f"Invalid prompt_variant: {prompt_variant}"
    if monitor_type == "all":
        monitor_types = MONITOR_TYPES
    else:
        assert (
            monitor_type in MONITOR_TYPES
        ), f"monitor_type must be 'all' or one of {MONITOR_TYPES}"
        monitor_types = [monitor_type]
    if prompt_variant != "standard":
        assert monitor_types == ["cot_action"], (
            "Non-standard prompt variants currently support only "
            "monitor_type='cot_action'"
        )

    if debug:
        rich.print("Debug mode enabled")
        time.sleep(1)
    rich.print("Number of side tasks:", len(side_task_list))
    rich.print("Number of attack policies:", len(attack_policy_list))

    if debug:
        time.sleep(2)

    for side_task in side_task_list:
        for attack_policy in attack_policy_list:

            if attack_policy == "no_side_task":
                final_filter_mode = "main-all"
            else:
                final_filter_mode = filter_mode

            input_path = f"{GLOBAL_DIRECTORY}/results/attack/{main_task}.{side_task}.{attack_policy}/{model.split('/')[-1]}.{attack_rollouts}.tested.jsonl"
            input_paths.append(input_path)

            assert (
                monitor_policy in MONITOR_POLICIES
            ), f"Invalid monitor_policy: {monitor_policy}"

            assert (
                final_filter_mode in FILTER_MODES
            ), f"Invalid filter_mode: {final_filter_mode}"
            assert (
                attack_rollouts > 0
            ), f"attack_rollouts must be > 0, got {attack_rollouts}"

            output_dir = os.path.join(
                GLOBAL_DIRECTORY,
                f"datasets/monitor/{main_task}.{side_task}.{attack_policy}/{model.split('/')[-1]}--{attack_rollouts}/",
            )

            base_name = f"{final_filter_mode}.{monitor_policy}"
            if prompt_variant != "standard":
                base_name = f"{base_name}.{prompt_variant}"

            output_paths_by_type = {
                monitor_type: os.path.join(
                    output_dir, f"{base_name}.{monitor_type}.jsonl"
                )
                for monitor_type in monitor_types
            }
            pending_types = [
                monitor_type
                for monitor_type, path in output_paths_by_type.items()
                if not os.path.exists(path)
            ]
            if not pending_types:
                rich.print("[orange3]⚠ Already exist, skipping:[/orange3]")
                if debug:
                    time.sleep(1)
                for path in output_paths_by_type.values():
                    rich.print(f"  [dim]{path}[/dim]")
                    if debug:
                        time.sleep(1)
                continue

            rich.print("[magenta bold]═══ for_monitor ═══[/magenta bold]")
            rich.print(f"[blue]→ Input: {input_path}[/blue]")
            rich.print(f"[bright_blue]← Output: {output_dir}[/bright_blue]")
            rich.print(
                f"[dim]main_task={main_task}, side_task={side_task}, attack_policy={attack_policy}[/dim]"
            )
            rich.print(f"[dim]model={model}, temp={attack_temp}[/dim]")
            rich.print(
                f"[dim]monitor_policy={monitor_policy}, monitor_types={pending_types}, "
                f"prompt_variant={prompt_variant}, filter_mode={final_filter_mode}, "
                f"attack_rollouts={attack_rollouts}[/dim]"
            )
            id_set = (
                SIDE_TASK_ID_SETS[side_task] if side_task in SIDE_TASK_ID_SETS else ""
            )
            if id_set:
                rich.print(f"[dim]upstream id_set={id_set} (informational)[/dim]")
            if debug:
                time.sleep(2)

            records = load_jsonl(input_path)
            rich.print(
                f"[yellow]Loaded: {len(records)} / {len(records)} = 100.0%[/yellow]"
            )

            if debug:
                time.sleep(1)

            records = filter_by_rollout(records, attack_rollouts)
            rich.print(
                f"[yellow]After rollout filter (< {attack_rollouts}): {len(records)}[/yellow]"
            )
            if debug:
                time.sleep(1)

            records = filter_by_pass_criteria(records, final_filter_mode)
            rich.print(
                f"[yellow]After pass filter ({final_filter_mode}): {len(records)}[/yellow]"
            )

            if debug:
                time.sleep(1)

            assert len(records) > 0, "No records left after filtering"

            input_task_ids = [r["task_id"] for r in records]
            assert len(input_task_ids) == len(
                set(input_task_ids)
            ), f"Duplicate task_ids in input: {[tid for tid in input_task_ids if input_task_ids.count(tid) > 1]}"

            templates = load_monitor_prompt(main_task, prompt_path)
            rich.print(
                f"[dim]Loaded templates from {prompt_path}/prompt_{main_task}.yaml[/dim]"
            )
            if debug:
                time.sleep(1)

            action_records, cot_records, cot_action_records = build_monitor_records(
                records,
                templates,
                monitor_policy,
                attack_rollouts,
                prompt_variant=prompt_variant,
            )

            if debug:
                debug_example(
                    records[0],
                    action_records[0]["messages"][0]["content"],
                    cot_records[0]["messages"][0]["content"],
                    cot_action_records[0]["messages"][0]["content"],
                )
                time.sleep(1)

            records_by_type = {
                "action_only": action_records,
                "cot_only": cot_records,
                "cot_action": cot_action_records,
            }
            output_paths = []
            for pending_type in pending_types:
                records_list = records_by_type[pending_type]
                path = output_paths_by_type[pending_type]
                task_ids = [r["task_id"] for r in records_list]
                assert len(task_ids) == len(
                    set(task_ids)
                ), f"Duplicate task_ids in {path}: {[tid for tid in task_ids if task_ids.count(tid) > 1]}"
                save_jsonl(records_list, path)
                output_paths.append(path)

            rich.print(f"[green]✓ Done! Generated {len(output_paths)} files:[/green]")
            for path in output_paths:
                rich.print(f"  [bright_blue]← {path}[/bright_blue]")

            output_path_groups.append(output_paths)

    return output_path_groups


if __name__ == "__main__":
    Fire(for_monitor_main)
