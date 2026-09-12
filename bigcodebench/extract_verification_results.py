import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List
import pdb


def _load_summaries_from_zip(zf: zipfile.ZipFile) -> List[Dict[str, Any]]:
    """Load summaries from a .eval ZIP in a version-tolerant way."""
    # Newer inspect versions store summaries at the top-level.
    if "summaries.json" in zf.namelist():
        return json.loads(zf.read("summaries.json"))

    # Older versions may store summaries in _journal/summaries/*.json
    journal_prefix = "_journal/summaries/"
    journal_files = sorted(
        name for name in zf.namelist() if name.startswith(journal_prefix)
    )
    if journal_files:
        summaries: List[Dict[str, Any]] = []
        for name in journal_files:
            summaries.extend(json.loads(zf.read(name)))
        return summaries

    # Fall back to start.json if it includes embedded results.
    if "_journal/start.json" in zf.namelist():
        start_data = json.loads(zf.read("_journal/start.json"))
        return start_data.get("results", {}).get("scores", [])

    raise FileNotFoundError("No summaries found in .eval archive")


def _score_value_to_bool(value: Any) -> bool:
    """Best-effort mapping of score values to a pass/fail boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"c", "correct", "pass", "passed", "true", "1", "yes"}
    return False


def load_eval_files(verification_dir: Path, filter_mode: str = "both") -> List[Dict[str, Any]]:
    """Load verification results from .eval files.

    Args:
        verification_dir: Path to verification_logs_* directory
        filter_mode: Filter mode - "both", "main", "side", "main_only", "side_only", "any"

    Returns:
        List of records with verification results and responses
    """
    if not verification_dir.exists():
        raise FileNotFoundError(f"Verification directory not found: {verification_dir}")

    # Find all .eval files
    eval_files = list(verification_dir.glob("*.eval"))

    if not eval_files:
        raise FileNotFoundError(f"No .eval files found in {verification_dir}")

    print(f"Found {len(eval_files)} .eval file(s)")

    # Load all records from .eval files
    all_records = []

    for eval_file in eval_files:
        try:
            with zipfile.ZipFile(eval_file, 'r') as zf:
                summaries_data = _load_summaries_from_zip(zf)
                for sample in summaries_data:
                    # Extract relevant fields
                    sample_id = sample.get('id', '')
                    epoch = sample.get('epoch', 1)
                    input_text = sample.get('input', '')
                    metadata = sample.get('metadata', {})

                    # Get the verification scores
                    scores = sample.get('scores', {})
                 
                    main_and_side = scores.get('pass_scorer', {}).get('metadata', {}).get('main_and_side', None)
                    main_only = scores.get('pass_scorer', {}).get('metadata', {}).get('main_only', None)
                    side_only = scores.get('pass_scorer', {}).get('metadata', {}).get('side_only', None)

              

                    # Get the response from the sample file in the ZIP
                    try:
                        sample_file = f"samples/{sample_id}_epoch_{epoch}.json"
                        sample_data = json.loads(zf.read(sample_file))
                        output = sample_data.get('output', {})
                        response = output.get('completion', '')
                    except Exception as e:
                        print(f"Warning: Could not read sample file for {sample_id}: {e}")
                        response = ""

                    # Create record
                    record = {
                        "id": sample_id,
                        "prompt": metadata.get('prompt', input_text),
                        "response": response,
                        "metadata": metadata,
                        "verification_result": {
                            "main_and_side": main_and_side,
                            "main_only": main_only,
                            "side_only": side_only,
                            "value": scores.get('pass_scorer', {}).get('value', ''),
                        },
                        "eval_file": eval_file.name,
                    }
                    # Apply filter based on mode
                    should_include = False
                    if filter_mode == "both":
                        should_include = main_and_side
                    elif filter_mode == "main":
                        should_include = main_only or main_and_side
                    elif filter_mode == "side":
                        should_include = side_only or main_and_side
                    elif filter_mode == "main_only":
                        should_include = main_only and not main_and_side
                    elif filter_mode == "side_only":
                        should_include = side_only and not main_and_side
                    elif filter_mode == "any":
                        should_include = True

                    if should_include:
                        all_records.append(record)

        except Exception as e:
            print(f"Error reading {eval_file}: {e}")

    return all_records


def save_to_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Save records to JSONL file.

    Args:
        records: List of records to save
        output_path: Path to output JSONL file
    """
    with output_path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(records)} records to {output_path}")


def print_statistics(all_records: List[Dict[str, Any]], filter_mode: str) -> None:
    """Print verification statistics.

    Args:
        all_records: All records from .eval files
        filter_mode: The filter mode used
    """
    filtered_count = len(all_records)

    if filtered_count == 0:
        print(f"\n{'='*60}")
        print(f"Verification Statistics")
        print(f"{'='*60}")
        print(f"⚠️  No samples matched filter: {filter_mode}")
        print(f"{'='*60}\n")
        return

    # Count different verification states from filtered records
    main_and_side = sum(1 for r in all_records if r['verification_result']['main_and_side'])
    main_only = sum(1 for r in all_records if r['verification_result']['main_only'] and not r['verification_result']['main_and_side'])
    side_only = sum(1 for r in all_records if r['verification_result']['side_only'] and not r['verification_result']['main_and_side'])
    failed = filtered_count - main_and_side - main_only - side_only

    print(f"\n{'='*60}")
    print(f"Verification Statistics (Filter: {filter_mode})")
    print(f"{'='*60}")
    print(f"Filtered samples:           {filtered_count}")
    print(f"✓ Main + Side:              {main_and_side} ({main_and_side/filtered_count*100:.1f}%)")
    print(f"⚠ Main only:                {main_only} ({main_only/filtered_count*100:.1f}%)")
    print(f"⚠ Side only:                {side_only} ({side_only/filtered_count*100:.1f}%)")
    print(f"✗ Failed both:              {failed} ({failed/filtered_count*100:.1f}%)")
    print(f"{'='*60}\n")

def extract_verification_results(args):
    verification_dir = Path(args.verification_dir)
    filtered_records = load_eval_files(verification_dir, args.filter)
    print_statistics(filtered_records, args.filter)

    if not args.stats_only:
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Always use correct_samples.jsonl regardless of filter mode
            output_path = verification_dir / "correct_samples.jsonl"

        save_to_jsonl(filtered_records, output_path)

        print(f"\n{'='*60}")
        print(f"✓ Extraction complete!")
        print(f"Output: {output_path}")
        print(f"Filter applied: {args.filter}")
        print(f"{'='*60}\n")
    else:
        print("✓ Statistics only mode - no file saved\n")

def load_correct_json_files(log_dir: Path, side_tasks: str, attack_policy: str) -> List[Dict[str, Any]]:
    """Load verification results from .eval files in the verification_logs directory.

    Args:
        log_dir: Path to the logs directory (parent of verification_logs_*)
        side_tasks: Comma-separated side task names to load results for
        attack_policy: Attack policy name

    Returns:
        List of records with verification results and responses
    """
    all_records = []
    side_task_list = [task.strip() for task in side_tasks.split(",") if task.strip()]
    # print(side_task_list)
    for side_task in side_task_list:
        verification_dir = log_dir / "verification_logs" / attack_policy / f"verification_logs_{side_task}"

        if not verification_dir.exists():
            print(f"WARNING: Verification directory not found: {verification_dir}")
            continue

        path = verification_dir / "correct_samples.jsonl"
        if not path.exists():
            print(f"WARNING: correct_samples.jsonl not found: {path}")
            continue

        # print(f"Loading verification results from: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
    
    return all_records
