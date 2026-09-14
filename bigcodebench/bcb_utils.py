# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from utils import *


def extract_prompts_from_tasks(task_list: List) -> Tuple[List[str], List, List[Dict]]:
    """Extract prompts, targets, and metadata from inspect_ai Task objects.

    Args:
        task_list: List of inspect_ai Task objects

    Returns:
        Tuple of (prompts, targets, metadata)
    """
    prompts = []
    targets = []
    metadata = []

    for task in task_list:
        # Iterate through all samples in the task's dataset
        for sample in task.dataset:
            prompts.append(sample.input)
            targets.append(sample.target)

            # Extract all metadata from the sample
            meta = sample.metadata.copy() if sample.metadata else {}

            # Add task identification
            meta["task_id"] = sample.id
            meta["task_name"] = task.name

            # We do not need side_test_case for all the cases?

            # Verify critical fields are present for verification
            # required_fields = ["side_test_case", "code", "side_task_name"]
            # missing_fields = [field for field in required_fields if field not in meta]
            # if missing_fields:
            #     print(f"Task_id: {sample.id} | Task_name: {task.name}")
            #     print(f"WARNING: Sample {sample.id} missing metadata fields: {missing_fields}")
            #     print(f"  Available fields: {list(meta.keys())}")

            # Sanitize metadata to ensure it's JSON serializable
            meta = sanitize_for_json(meta)

            metadata.append(meta)

    # Print summary of metadata
    # if metadata:
    #     print(f"\nMetadata summary for first sample:")
    #     first_meta = metadata[0]
    #     print(f"  Fields present: {list(first_meta.keys())}")
    #     print(f"  Side task: {first_meta.get('side_task_name', 'MISSING')}")
    #     print(f"  Has side_test_case: {'side_test_case' in first_meta}")

    return prompts, targets, metadata
