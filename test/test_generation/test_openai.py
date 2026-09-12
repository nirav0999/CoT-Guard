import json
import os
import sys
import time
from datetime import datetime

from gen.generate import GenerationConfig, run_api_generation

TEST_DIR = os.path.dirname(__file__)


def get_test_paths(timestamp: str, model: str, test_no: int) -> tuple[str, str]:
    safe_model = model.replace("/", "_")
    results_dir = f"{TEST_DIR}/results_openai/{timestamp}/{safe_model}"
    os.makedirs(results_dir, exist_ok=True)
    input_path = f"{results_dir}/test_{test_no}_input.jsonl"
    output_path = f"{results_dir}/test_{test_no}_output.jsonl"
    return input_path, output_path


def test_run_openai_generation_single_rollout(timestamp: str, model: str, test_no: int):
    input_path, output_path = get_test_paths(timestamp, model, test_no)
    with open(input_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "task_id": "task1",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "task_id": "task2",
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                }
            )
            + "\n"
        )

    config = GenerationConfig(model=model, backend="openai", n=1, max_tokens=256, bs=2)
    run_api_generation(input_path, output_path, config)

    with open(output_path, "r") as f:
        results = [json.loads(line) for line in f]

    assert len(results) == 2
    assert results[0]["task_id"] == "task1"
    assert results[0]["messages"][-1]["role"] == "assistant"
    assert len(results[0]["messages"][-1]["content"]) > 0
    assert results[1]["task_id"] == "task2"
    assert results[1]["messages"][-1]["role"] == "assistant"
    assert len(results[1]["messages"][-1]["content"]) > 0


def test_run_openai_generation_multiple_rollouts(
    timestamp: str, model: str, test_no: int
):
    input_path, output_path = get_test_paths(timestamp, model, test_no)

    with open(input_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "task_id": "task1",
                    "messages": [
                        {"role": "user", "content": "Write a haiku about coding"}
                    ],
                }
            )
            + "\n"
        )

    config = GenerationConfig(model=model, backend="openai", n=3, max_tokens=256, bs=2)
    run_api_generation(input_path, output_path, config)

    with open(output_path, "r") as f:
        results = [json.loads(line) for line in f]

    assert len(results) == 3
    assert results[0]["task_id"] == "task1:0"
    assert results[1]["task_id"] == "task1:1"
    assert results[2]["task_id"] == "task1:2"
    for r in results:
        assert r["messages"][-1]["role"] == "assistant"
        assert len(r["messages"][-1]["content"]) > 0


def test_run_openai_generation_preserves_metadata(
    timestamp: str, model: str, test_no: int
):
    input_path, output_path = get_test_paths(timestamp, model, test_no)

    with open(input_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "task_id": "task1",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "metadata": {"source": "test", "difficulty": "easy"},
                }
            )
            + "\n"
        )

    config = GenerationConfig(model=model, backend="openai", n=1, max_tokens=64, bs=2)
    run_api_generation(input_path, output_path, config)

    with open(output_path, "r") as f:
        results = [json.loads(line) for line in f]

    assert results[0]["metadata"]["source"] == "test"
    assert results[0]["metadata"]["difficulty"] == "easy"


def run_timed(name: str, fn, timestamp: str, model: str, test_no: int) -> float:
    start = time.time()
    fn(timestamp, model, test_no)
    elapsed = time.time() - start
    print(f"[PASS] {name} ({elapsed:.2f}s)")
    return elapsed


if __name__ == "__main__":
    MODELS = [
        "gpt-5-mini-2025-08-07"
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in MODELS:
        print(f"Running OpenAI Batch API tests for {model} (jobs may take a while)...")
        print(
            f"Results: {TEST_DIR}/results_openai/{timestamp}/{model.replace('/', '_')}/"
        )
        print()

        total = 0.0
        total += run_timed(
            "test_single_rollout",
            test_run_openai_generation_single_rollout,
            timestamp,
            model,
            1,
        )
        total += run_timed(
            "test_multiple_rollouts",
            test_run_openai_generation_multiple_rollouts,
            timestamp,
            model,
            2,
        )
        total += run_timed(
            "test_preserves_metadata",
            test_run_openai_generation_preserves_metadata,
            timestamp,
            model,
            3,
        )

        print()
        print(f"All tests passed for {model}! Total: {total:.2f}s")
        print("=" * 60)
        print()
