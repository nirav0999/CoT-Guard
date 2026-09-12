import argparse
import os

from dotenv import load_dotenv
from inspect_ai import eval_set
from inspect_ai.model import get_model

from cot_red_handed.monitor.inspect_monitor import run_monitor

load_dotenv()


def run_monitor_on_dir(
    eval_log_dir: str,
    prompt_id: int,
    prompt_path: str,
    limit: int | None = None,
    log_dir: str | None = None,
    has_side_task: bool = True,
    max_tasks: int = 10,
    max_connections: int = 100,
    max_sandboxes: int = 100,
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required but not set")

    monitor_model = get_model("openai/gpt-4.1-mini", api_key=api_key)
    if log_dir is None:
        log_dir = os.path.join(eval_log_dir, "monitor")

    eval_files = [os.path.join(eval_log_dir, ef) for ef in os.listdir(eval_log_dir) if ef.endswith(".eval")]
    if len(eval_files) == 0:
        raise ValueError(f"No '.eval' files found in {eval_log_dir}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    basenames = [os.path.splitext(os.path.basename(ef))[0] for ef in eval_files]
    errorfiles = [os.path.join(log_dir, bn + f"_monitor_prompt_{prompt_id}_errors.csv") for bn in basenames]

    tasks = []
    for ef, er in zip(eval_files, errorfiles):
        try:
            task = run_monitor(
                eval_log_path=ef,
                epochs=1,
                prompt_id=prompt_id,
                prompt_path=prompt_path,
                errorfile=er,
                has_side_task=has_side_task,
            )
            tasks.append(task)
        except ValueError as e:
            print(f"WARNING: file {ef} failed with error : \n{e}")
            continue

    eval_set(
        tasks=tasks,
        model=monitor_model,
        limit=limit,
        log_dir=log_dir,
        max_tasks=max_tasks,
        max_connections=max_connections,
        max_sandboxes=max_sandboxes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_log_path", type=str)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--prompt-path", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument(
        "--no-side-task",
        action="store_false",
        dest="has_side_task",
        default=True,
        help="Use this flag to get baseline scores without a side task.",
    )
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-connections", type=int, default=100)
    parser.add_argument("--max-sandboxes", type=int, default=100)
    args = parser.parse_args()

    if not os.path.isdir(args.eval_log_path):
        raise ValueError(f"Invalid eval_log_path: {args.eval_log_path}. Must be a directory containing '.eval' files.")

    run_monitor_on_dir(
        eval_log_dir=args.eval_log_path,
        prompt_id=args.prompt_id,
        prompt_path=args.prompt_path,
        limit=args.limit,
        log_dir=args.log_dir,
        has_side_task=args.has_side_task,
        max_tasks=args.max_tasks,
        max_connections=args.max_connections,
        max_sandboxes=args.max_sandboxes,
    )
