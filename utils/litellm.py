# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Callable, Dict, List

import rich
from dotenv import load_dotenv
from litellm import completion_with_retries
from termcolor import cprint
from tqdm import tqdm

load_dotenv()


def log_costs(completions):
    costs = [r._hidden_params["response_cost"] for r in completions]
    if len(costs) == 0 or None in costs:
        return
    cprint(f"{len(costs)} requests costs ${sum(costs):.3f}", "yellow")


def mini_batch_completion(messages, parallel: int = 32, **kwargs):
    from utils import split_batch

    batches = split_batch(messages, n=parallel)
    outputs = []
    for minibatch in tqdm(batches):
        with ThreadPoolExecutor(max_workers=len(minibatch)) as executor:
            futures = []
            for sample in minibatch:
                future = executor.submit(
                    completion_with_retries,
                    messages=sample["messages"],
                    num_retries=32,
                    retry_strategy="exponential_backoff_retry",
                    **kwargs,
                )
                futures.append(future)

            for future in futures:
                outputs.append(future.result())

    return outputs


def run_batched_inference(
    batched_rows: List,  # each row includes at least "messages"
    row_transform: Callable[[Dict], Dict] = lambda x: x,
    max_new_tokens: int = None,
    temperature: float = None,
    model: str = "openai/gpt-5-mini",
    parallel: int = 12,
    **kwargs,
):
    assert batched_rows and "messages" in batched_rows[0]
    batched_rows = [row_transform(row) for row in batched_rows]
    print("Running batched completion with model:", model)
    parameters = {
        "model": model,
        "parallel": parallel,
        "messages": batched_rows,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        **kwargs,
    }
    if "thinking" in kwargs:
        assert parameters["max_tokens"] is None
        assert parameters["temperature"] is None

        if "gpt-5" in model:
            rich.print("Using GPT-5 with thinking enabled.")
            parameters["reasoning_effort"] = "medium"
            parameters.pop("thinking")
    else:
        if parameters["temperature"] is None:
            parameters["temperature"] = 0.0

    outputs = mini_batch_completion(**parameters)
    log_costs(outputs)
    outputs = [item.choices[0].message for item in outputs]

    output_rows = []
    for row, ext in zip(batched_rows, outputs, strict=False):
        row = deepcopy(row)
        reasoning_content = (
            "<think>\n" + ext.reasoning_content + "\n</think>\n"
            if hasattr(ext, "reasoning_content") and ext.reasoning_content
            else ""
        )
        row["messages"].append(
            {"role": "assistant", "content": reasoning_content + ext.content}
        )
        output_rows.append(row)
    return output_rows
