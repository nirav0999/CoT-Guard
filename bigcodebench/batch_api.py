# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import json

from utils import *


def prepare_batch_monitor_requests(args, monitor_items, monitor_prompt):
    requests = []
    for item in monitor_items:
        custom_id = f"{item["sample_id"]}_{args.model_name.split("/")[-1]}_monitor_{args.monitor_model}={args.dataset}={args.side_task}={args.attack_policy}"
        action_content = format_monitor_prompt(
            monitor_prompt[f"action_prompt_{args.monitor_policy}"],
            item["prompt"],
            item["cot"],
            item["code"],
            item["metadata"]["side_task_description"],
        )
        cot_content = format_monitor_prompt(
            monitor_prompt[f"cot_prompt_{args.monitor_policy}"],
            item["prompt"],
            item["cot"],
            item["code"],
            item["metadata"]["side_task_description"],
        )
        cot_action_content = format_monitor_prompt(
            monitor_prompt[f"cot_action_prompt_{args.monitor_policy}"],
            item["prompt"],
            item["cot"],
            item["code"],
            item["metadata"]["side_task_description"],
        )

        requests.append(
            {
                "custom_id": "action=" + custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.monitor_model,
                    "messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": action_content},
                    ],
                    # "temperature": args.temperature,
                },
            }
        )

        requests.append(
            {
                "custom_id": "cot=" + custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.monitor_model,
                    "messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": cot_content},
                    ],
                    # "temperature": args.temperature,
                },
            }
        )

        requests.append(
            {
                "custom_id": "cot+action=" + custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.monitor_model,
                    "messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": cot_action_content},
                    ],
                    # "temperature": args.temperature,
                },
            }
        )

    assert len(requests) == 3 * len(monitor_items)
    batch_query_dir = (
        Path(args.batch_path)
        / args.attack_policy
        / args.monitor_model
        / args.monitor_policy
    )
    batch_query_dir.mkdir(parents=True, exist_ok=True)
    batch_query_file = batch_query_dir / f"batch_query_{args.side_task}.jsonl"
    with open(batch_query_file, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    print(f"Saved batch monitor requests to {batch_query_file}")
    return batch_query_dir, batch_query_file
