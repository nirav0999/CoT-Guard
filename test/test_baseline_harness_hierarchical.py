import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from baseline_harness.hierarchical import (
    SUMMARY_SEPARATOR,
    build_aggregation_messages,
    build_leaf_messages,
    run_hierarchical_monitoring,
)
from baseline_harness.run_hierarchical import (
    hierarchy_shape,
    run_api_hierarchical_monitoring,
    run_batch_api_hierarchical_monitoring,
    run_vllm_hierarchical_monitoring,
)
from baseline_harness.run_parallel import build_worker_command
from eval.display_table import (
    build_hierarchical_main_path,
    build_hierarchical_side_path,
    extract_table_choices,
)
from gen.generate import GenerationConfig


def sample_record(task_id: str, response: str) -> dict:
    return {
        "task_id": task_id,
        "messages": [
            {"role": "user", "content": "private attack instructions"},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "original_prompt": "Complete the benign main task.",
            "side_task_description": "private side task",
        },
    }


class HierarchicalHarnessTests(unittest.TestCase):
    def test_prompts_preserve_roles_and_monitor_inputs(self) -> None:
        leaf_messages = build_leaf_messages("benign task", "agent excerpt")
        self.assertEqual(
            [message["role"] for message in leaf_messages],
            ["system", "user"],
        )
        self.assertIn("benign task", leaf_messages[1]["content"])
        self.assertIn("agent excerpt", leaf_messages[1]["content"])

        aggregate_messages = build_aggregation_messages(
            ["first summary", "second summary"]
        )
        self.assertEqual(
            [message["role"] for message in aggregate_messages],
            ["system", "user"],
        )
        self.assertIn(
            f"first summary{SUMMARY_SEPARATOR}second summary",
            aggregate_messages[1]["content"],
        )

    def test_recursive_tree_preserves_order_and_uneven_groups(self) -> None:
        calls: list[list[list[dict[str, str]]]] = []

        def infer_batch(
            messages_batch: list[list[dict[str, str]]],
        ) -> list[str]:
            calls.append(messages_batch)
            if len(calls) == 1:
                return [
                    f"leaf-{index} <answer>1</answer>"
                    for index in range(6)
                ]
            if len(calls) == 2:
                return [
                    "parent-0 <answer>2</answer>",
                    "parent-1 <answer>4</answer>",
                ]
            return ["root <answer>4</answer>"]

        outputs = run_hierarchical_monitoring(
            [sample_record("uneven", "abcdef")],
            infer_batch,
            "monitor-model",
            max_chunk_chars=1,
            aggregation_fanout=5,
        )

        monitor = outputs[0]["hierarchical_monitor"]
        self.assertEqual([len(layer) for layer in monitor["layers"]], [6, 2, 1])
        self.assertEqual(
            monitor["layers"][1][0]["child_indices"],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            monitor["layers"][1][1]["child_indices"],
            [5],
        )
        self.assertEqual(
            (monitor["layers"][-1][0]["char_start"],
             monitor["layers"][-1][0]["char_end"]),
            (0, 6),
        )
        self.assertEqual(monitor["final_score"], 4.0)
        self.assertEqual(monitor["choice"], "C")
        self.assertIn("leaf-0", calls[1][0][1]["content"])
        self.assertIn("leaf-4", calls[1][0][1]["content"])
        self.assertNotIn("leaf-5", calls[1][0][1]["content"])
        self.assertIn("leaf-5", calls[1][1][1]["content"])
        self.assertIn("parent-0", calls[2][0][1]["content"])
        self.assertIn("parent-1", calls[2][0][1]["content"])

    def test_uneven_records_batch_by_tree_level(self) -> None:
        batch_sizes = []

        def infer_batch(
            messages_batch: list[list[dict[str, str]]],
        ) -> list[str]:
            batch_sizes.append(len(messages_batch))
            return [
                f"response-{len(batch_sizes)}-{index} <answer>2</answer>"
                for index in range(len(messages_batch))
            ]

        outputs = run_hierarchical_monitoring(
            [
                sample_record("short", "a"),
                sample_record("long", "bcd"),
            ],
            infer_batch,
            "monitor-model",
            max_chunk_chars=1,
            aggregation_fanout=2,
        )

        self.assertEqual(batch_sizes, [4, 2, 1])
        self.assertEqual(
            [output["task_id"] for output in outputs],
            ["short", "long"],
        )
        self.assertEqual(
            [
                output["hierarchical_monitor"]["num_layers"]
                for output in outputs
            ],
            [1, 3],
        )

    def test_invalid_fanout_and_response_count_fail(self) -> None:
        with self.assertRaises(ValueError):
            run_hierarchical_monitoring(
                [sample_record("invalid", "text")],
                lambda requests: ["response"],
                "monitor-model",
                aggregation_fanout=1,
            )
        with self.assertRaises(ValueError):
            run_hierarchical_monitoring(
                [sample_record("mismatch", "text")],
                lambda requests: [],
                "monitor-model",
            )

    def test_hierarchy_shape(self) -> None:
        self.assertEqual(hierarchy_shape(1, 5), [1])
        self.assertEqual(hierarchy_shape(20, 5), [20, 4, 1])
        with self.assertRaises(ValueError):
            hierarchy_shape(2, 1)

    def test_parallel_worker_selects_hierarchical_scaffold(self) -> None:
        command = build_worker_command(
            "/tmp/input.jsonl",
            "/tmp/output.jsonl",
            "monitor-model",
            1,
            5000,
            2048,
            0.6,
            42,
            True,
            0.95,
            scaffold="hierarchical",
            aggregation_fanout=7,
        )

        self.assertIn("--scaffold=hierarchical", command)
        self.assertIn("--aggregation_fanout=7", command)

    def test_vllm_adapter_includes_system_messages(self) -> None:
        model = Mock()
        tokenizer = Mock()
        model.generate.side_effect = [["leaf outputs"], ["root output"]]
        config = GenerationConfig(
            model="monitor-model",
            max_tokens=32,
            n=1,
            seed=42,
        )

        with (
            patch(
                "baseline_harness.run_hierarchical.init_vllm",
                return_value=(model, tokenizer),
            ),
            patch(
                "baseline_harness.run_hierarchical.create_sampling_params",
                return_value="sampling-params",
            ),
            patch(
                "baseline_harness.run_hierarchical.format_prompts",
                side_effect=[
                    ["leaf-0", "leaf-1"],
                    ["root"],
                ],
            ) as format_mock,
            patch(
                "baseline_harness.run_hierarchical.extract_responses",
                side_effect=[
                    [
                        ["leaf 0 <answer>1</answer>"],
                        ["leaf 1 <answer>4</answer>"],
                    ],
                    [["root <answer>4</answer>"]],
                ],
            ),
            patch(
                "baseline_harness.run_hierarchical.cleanup_vllm"
            ) as cleanup_mock,
        ):
            outputs = run_vllm_hierarchical_monitoring(
                [sample_record("vllm", "ab")],
                config,
                max_chunk_chars=1,
                aggregation_fanout=5,
            )

        self.assertEqual(format_mock.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["include_system_messages"] is True
                for call in format_mock.call_args_list
            )
        )
        cleanup_mock.assert_called_once_with(model)
        self.assertEqual(
            outputs[0]["hierarchical_monitor"]["choice"],
            "C",
        )

    def test_api_adapter_preserves_level_order(self) -> None:
        calls = []

        def fake_batched_inference(
            rows: list[dict],
            **kwargs: object,
        ) -> list[dict]:
            calls.append((rows, kwargs))
            if len(calls) == 1:
                responses = [
                    "first leaf <answer>1</answer>",
                    "second leaf <answer>4</answer>",
                ]
            else:
                responses = ["root <answer>4</answer>"]
            return [
                {
                    "messages": row["messages"]
                    + [{"role": "assistant", "content": response}]
                }
                for row, response in zip(rows, responses)
            ]

        with patch(
            "baseline_harness.run_hierarchical.run_batched_inference",
            side_effect=fake_batched_inference,
        ):
            outputs = run_api_hierarchical_monitoring(
                [sample_record("api", "ab")],
                "openai/api-monitor",
                max_chunk_chars=1,
                aggregation_fanout=5,
                max_tokens=128,
                temperature=0.0,
                api_parallel=7,
            )

        self.assertEqual([len(call[0]) for call in calls], [2, 1])
        self.assertEqual(calls[0][1]["parallel"], 7)
        self.assertIn(
            "first leaf",
            calls[1][0][0]["messages"][1]["content"],
        )
        self.assertIn(
            "second leaf",
            calls[1][0][0]["messages"][1]["content"],
        )
        self.assertEqual(
            outputs[0]["hierarchical_monitor"]["choice"],
            "C",
        )

    def test_batch_api_adapter_submits_one_batch_per_level(self) -> None:
        from utils import batch_api_run

        levels = []

        def fake_prepare(
            input_path: str,
            records: list[dict],
            config: GenerationConfig,
        ) -> tuple[str, list[dict]]:
            levels.append(records)
            return input_path, [
                {"custom_id": f"request-{index}"}
                for index in range(len(records))
            ]

        def fake_convert(
            output_path: str,
            records: list[dict],
            config: GenerationConfig,
        ) -> list[list[str]]:
            if len(levels) == 1:
                return [
                    [f"leaf-{index} <answer>1</answer>"]
                    for index in range(len(records))
                ]
            return [["root <answer>4</answer>"] for _ in records]

        with (
            patch.object(
                batch_api_run,
                "prepare_batch_requests",
                side_effect=fake_prepare,
            ),
            patch.object(
                batch_api_run,
                "run_batched_inference",
                return_value="/tmp/batch-output.jsonl",
            ),
            patch.object(
                batch_api_run,
                "convert_batch_outputs_to_responses",
                side_effect=fake_convert,
            ),
        ):
            outputs = run_batch_api_hierarchical_monitoring(
                [sample_record("batch", "abc")],
                "gpt-5.4",
                max_chunk_chars=1,
                aggregation_fanout=5,
                max_tokens=128,
                temperature=0.0,
            )

        self.assertEqual([len(level) for level in levels], [3, 1])
        self.assertEqual(
            [message["role"] for message in levels[0][0]["messages"]],
            ["system", "user"],
        )
        self.assertEqual(
            outputs[0]["hierarchical_monitor"]["choice"],
            "C",
        )

    def test_display_table_support(self) -> None:
        with patch("eval.display_table.GLOBAL_DIRECTORY", "/results-root"):
            self.assertEqual(
                build_hierarchical_side_path(
                    "bigcodebench",
                    "exit",
                    "baseline",
                    "Qwen/Qwen3-14B",
                    5,
                    "org/monitor",
                ),
                (
                    "/results-root/results/baseline_harness/"
                    "bigcodebench.exit.baseline/Qwen3-14B--5/"
                    "monitor.hierarchical.chars-5000.jsonl"
                ),
            )
            self.assertEqual(
                build_hierarchical_main_path(
                    "bigcodebench",
                    "exit",
                    "Qwen/Qwen3-14B",
                    5,
                    "org/monitor",
                ),
                (
                    "/results-root/results/baseline_harness/"
                    "bigcodebench.exit.no_side_task/Qwen3-14B--5/"
                    "monitor.hierarchical.chars-5000.jsonl"
                ),
            )
        self.assertEqual(
            extract_table_choices(
                [
                    {"hierarchical_monitor": {"choice": "C"}},
                    {"hierarchical_monitor": {"choice": "None"}},
                ],
                "monitor",
                require_qwen_thinking=False,
                prompt_variant="hierarchical",
            ),
            ["C", "None"],
        )

    def test_dry_run_does_not_write_output(self) -> None:
        from baseline_harness.run_hierarchical import run_hierarchical
        from utils import save_jsonl

        with TemporaryDirectory() as temp_directory:
            input_path = os.path.join(temp_directory, "input.jsonl")
            output_path = os.path.join(temp_directory, "output.jsonl")
            save_jsonl([sample_record("dry", "abcdef")], input_path)
            returned_path = run_hierarchical(
                input_path,
                output_path,
                "monitor-model",
                max_chunk_chars=2,
                aggregation_fanout=2,
                dry_run=True,
            )
            self.assertEqual(returned_path, output_path)
            self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
