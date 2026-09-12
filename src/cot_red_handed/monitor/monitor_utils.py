import os
import re
from typing import Any

import yaml
from inspect_ai.log import EvalSample


def load_monitor_prompt(prompt_id: int, prompt_path: str) -> dict[Any, Any]:
    prompt_files = [f for f in os.listdir(prompt_path) if f.endswith(".yaml")]

    monitor_prompt = None
    filename = None

    for filename in prompt_files:
        if filename.endswith(".yaml") and str(prompt_id) in filename:
            with open(os.path.join(prompt_path, filename), "r") as f:
                monitor_prompt = yaml.safe_load(f)
            break

    if monitor_prompt is None:
        raise ValueError(f"Error: prompt id {prompt_id} not loaded from prompts folder.")

    if monitor_prompt.get("id", -1) != prompt_id:
        raise ValueError(
            f"Prompt file {filename} has id {monitor_prompt.get('id', -1)} but expected id {prompt_id}. Check that the 'id' field matches the requested prompt_id."
        )

    keys = list(monitor_prompt.keys())
    if not ("prompt" in keys or ("action_prompt" in keys and "cot_prompt" in keys)):
        raise ValueError(
            f"Error: could not find 'prompt' or ('action_prompt' AND 'cot_prompt') field in specified monitor prompt {prompt_id}"
        )
    return monitor_prompt


def pre_monitor_check(sample: EvalSample, scorer_name: str, eval_log_path: str) -> dict[str, Any]:
    """Catch samples with errors that would break the monitor, output appropriate error messages for csv"""

    if len(sample.output.choices) == 0 or sample.output.choices[0].stop_reason == "max_tokens":
        return {
            "eval_log": eval_log_path,
            "sample_id": sample.id,
            "epoch": sample.epoch,
            "error": "No output choices or max tokens reached.",
        }
    if sample.scores.get(scorer_name) is None:  # type: ignore
        return {
            "eval_log": eval_log_path,
            "sample_id": sample.id,
            "epoch": sample.epoch,
            "error": f"No scores in original eval log for scorer '{scorer_name}'",
        }

    return {}


def template_check(template: str):
    curly_bracket_pattern = re.compile(r"\{(.*?)\}")
    all_matches = curly_bracket_pattern.findall(template)
    if len(all_matches) == 0:
        raise ValueError(
            f"Error: template {template} should contain curly brackets for e.g coding problem, answer, reasoning process, etc."
        )
    for match in all_matches:
        if match not in ["input", "output", "cot", "action"]:
            raise ValueError(
                f"Error: template should only contain curly brackets {{input}}, {{output}}, {{cot}}, {{action}} but found {match}"
            )


def find_cot(model_response: str) -> str:
    """
    Find the COT wrapped in <think> tags.
    TODO: need to accommodate for models that are not DeepSeek-R1 or Sonnet 3.7 Extended Thinking, ie those that don't use <think> tags
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = pattern.findall(model_response)
    if len(matches) == 0:
        pattern = re.compile(r"(.*)</think>", re.DOTALL)
        matches = pattern.findall(model_response)

    return matches[0] if len(matches) >= 1 else ""
