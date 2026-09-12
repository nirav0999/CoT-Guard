# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import re
from pathlib import Path

import rich

from config import GLOBAL_DIRECTORY


def collect_main(
    main_task: str = "bigcodebench",
    side_tasks: tuple[str, ...] = (
        "exit",
        "permissions",
        "logio",
        "edgecase",
        "longlines",
        "benign_print_watermark",
        "benign_comment_watermark",
    ),
    evaluated_model: str = "Qwen3-14B",
    attack_policy: str = "baseline",
    rollout_num: int = 5,
    monitor_policy: str = "main_aware",
    monitor_objective: str = "cot_action",
    filter: str = "side-all",
):
    for side_task in side_tasks:

        records = []
        rich.print(
            f"[bold yellow] [INFO] Main: {main_task}, Side: {side_task}, eval_model: {evaluated_model}, monitor_policy: {monitor_policy}, attack_policy: {attack_policy}, monitor_objective: {monitor_objective} [/bold yellow]"
        )

        effective_rollout_num = rollout_num
        if side_task == "vulnimport":
            effective_rollout_num = 10

        side_path_file = (
            Path(GLOBAL_DIRECTORY)
            / "datasets"
            / "monitor"
            / f"{main_task}.{side_task}.{attack_policy}"
            / f"{evaluated_model}--{effective_rollout_num}"
            / f"{filter}.{monitor_policy}.{monitor_objective}.jsonl"
        )
        main_path_file = (
            Path(GLOBAL_DIRECTORY)
            / "datasets"
            / "monitor"
            / f"{main_task}.{side_task}.no_side_task"
            / f"{evaluated_model}--{effective_rollout_num}"
            / f"main-all.{monitor_policy}.{monitor_objective}.jsonl"
        )

        assert (
            side_path_file.exists()
        ), f"[ERROR] Missing side-task monitor file: {side_path_file}"
        assert (
            main_path_file.exists()
        ), f"[ERROR] Missing no_side_task monitor file: {main_path_file}"

        side_records = []
        with side_path_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                side_records.append(json.loads(line))

        side_records_dict = {}
        for side_record in side_records:
            sample_id = side_record["task_id"].split("_")[0]
            malicious_trajectory = side_record["messages"][0]["content"]
            match = re.search(
                r"START OF AGENT ANSWER(.*?)END OF AGENT ANSWER",
                malicious_trajectory,
                flags=re.DOTALL,
            )
            if match:
                extracted_code = match.group(1).strip()
                if sample_id in side_records_dict:
                    side_records_dict[sample_id].append(extracted_code)
                else:
                    side_records_dict[sample_id] = [extracted_code]

        with main_path_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                sample_id = data["task_id"].split("_")[0]
                if sample_id in side_records_dict:
                    benign_trajectory = data["messages"][0]["content"]
                    malicious_code = random.choice(side_records_dict[sample_id])
                    pattern = re.compile(
                        r"(START OF AGENT ANSWER)(.*?)(END OF AGENT ANSWER)",
                        flags=re.DOTALL,
                    )
                    benign_cot_malicious_code = pattern.sub(
                        lambda m, code=malicious_code: (
                            f"{m.group(1)}\n{code}\n{m.group(3)}"
                        ),
                        benign_trajectory,
                        count=1,
                    )

                    data["messages"][0]["content"] = benign_cot_malicious_code
                    data["task_id"] = data["task_id"].replace(
                        "no_side_task", "hardsamples"
                    )
                    data["metadata"]["main_pass"] = True
                    data["metadata"]["side_pass"] = True

                    records.append(data)

        # base_name = f'{"_".join(monitor_objectives)}.{"_".join(main_tasks)}.{"_".join(side_tasks)}.{"_".join(evaluated_models)}.{"_".join(monitor_models)}.{"_".join(monitor_policies)}.{"_".join(attack_policies)}'
        save_path = (
            Path(GLOBAL_DIRECTORY)
            / "datasets"
            / "monitor"
            / f"{main_task}.{side_task}.hardsamples"
            / f"{evaluated_model}--{effective_rollout_num}"
            / f"side-all.{monitor_policy}.{monitor_objective}.jsonl"
        )
        # Path("/srv/local/hanw14/icml2026/coding/clean/datasets/monitor")
        os.makedirs(save_path.parent, exist_ok=True)

        with save_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[INFO] Saved {save_path}. Total records: {len(records)}")


if __name__ == "__main__":
    collect_main()
