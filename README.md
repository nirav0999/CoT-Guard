# CoT-Guard

This repo includes the training, evaluation, and data curation code for CoT-Guard. Please also check out:

- [📝 Paper](https://arxiv.org/abs/2605.12746) with technical and evaluation details
- [🤗 HuggingFace](https://huggingface.co/CoT-Guard) 4B model with SFT and RL model checkpoints

CoT-Guard will be presented at [AdvML-Frontiers × CoTMA: From Model Security to Compositional Threats in Multi-Agent AI Systems @ COLM 2026](https://advml-frontier.github.io/) as an oral presentation. It is currently under review to a full conference venue.

## Environment variables

From the project root, set the data and temporary directories and add the project to `PYTHONPATH`:

```bash
export COT_GLOBAL_DIRECTORY="/path/to/data_and_results"
export COT_TEMP_DIRECTORY="/path/to/temp_for_vllm_logs"
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-gemini-api-key"
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Inference Pipeline

1. Attack: create tasks and generate attack responses.
2. Verify: test responses for main-task correctness and side-task success.
3. Monitor: create monitor tasks from the responses and run monitors.
4. Score: evaluate monitor predictions.

## Run Attack

a) Create prompts containing the main task (`bigcodebench`), side task (`varname`), and attack policy (`baseline`):

```bash
python3 task_creation/for_attack.py --main_task=bigcodebench --side_tasks=varname --attack_policies=baseline
```

b) Run the tasks with `Qwen/Qwen3-8B` on GPU 0 to generate one attack response per prompt:

```bash
python3 gen/attack.py single --main_task=bigcodebench --side_task=varname --attack_policy=baseline --model=Qwen/Qwen3-14B --devices=0 --tp=1 --num_rollouts=5
```

## Verify

With Docker running:

```bash
python3 eval/verify_task.py --main_task=bigcodebench --side_task=varname --attack_policy=baseline --attack_model=Qwen/Qwen3-14B --num_rollouts=5
```

Output: `$COT_GLOBAL_DIRECTORY/results/attack/bigcodebench.varname.baseline/Qwen3-14B.5.tested.jsonl`

## Monitor

a) Create monitor prompts from the attacks that were successful:

```bash
python3 task_creation/for_monitor.py --main_task=bigcodebench --side_tasks=varname --attack_policies=baseline --model=Qwen/Qwen3-14B --attack_rollouts=5 --filter_mode=side-all --monitor_policy=main_aware --monitor_type=cot_action
```

b) Run the monitor with `Qwen/Qwen3-8B`:

```bash
python3 gen/monitor.py single --main_task=bigcodebench --side_task=varname --attack_policy=baseline --attack_model=Qwen/Qwen3-14B --attack_rollouts=5 --filter_mode=side-all --monitor_policy=main_aware --monitor_model=Qwen/Qwen3-8B --monitor_type=cot_action --devices=0 --tp=1 --num_rollouts=1
```

## Results

Display one setting:

```bash
python3 eval/display.py --main_task=bigcodebench --evaluated_model=Qwen/Qwen3-14B --rollout_num=5 --attack_policy=baseline --monitor_policy=main_aware --monitor_objective=cot_action --filter_mode=side-all --side_tasks=varname --monitor_models=Qwen/Qwen3-8B
```

Display the result tables:

```bash
python3 eval/display_table.py --main_task=bigcodebench --evaluated_model=Qwen/Qwen3-14B --rollout_num=5 --attack_policy=baseline --monitor_policy=main_aware --monitor_objective=cot_action --filter_mode=side-all --side_tasks=varname --monitor_models=Qwen/Qwen3-8B
```


### Params

Attack policy values include `--attack_policy=baseline`, `no_side_task`, `stealth`, and `monitor_notice`.

`--tp=1` shards the model across one GPU. `--num_rollouts=1` generates one response per prompt.

Side task values include `--side_task=varname`, `defaultval`, `logio`, `iodelete`, `iodownload`, `permissions`, `ioexfil`, `edgecase`, `complexity`, `longlines`, `exit`, `vulnimport`, `typosquat`, `benign_comment_watermark`, `benign_print_watermark`.

You can also use comma-separated values, like `--side_tasks=varname,exit`.

For parallel generation across 2 GPUs --

```bash
python3 gen/attack.py parallel --main_task=bigcodebench --side_tasks=varname,exit --attack_policies=baseline --model=Qwen/Qwen3-14B --devices=0,1 --tp=1 --num_rollouts=5
```

tp per N devices --> tensor parallel for vllm


## RL Training Setup

TBD

## RL Dataset

Our prompt dataset with verified ground truth is available at [RL-hard](https://huggingface.co/datasets/CoT-Guard/rl-hard).

## Stored Eval and Results

All our results and datasets available for download from [here](https://drive.google.com/drive/folders/1eOg5--gmDHXLwIt17RyGC3FHxN-17y9q?usp=drive_link).

Place them in the `$COT_GLOBAL_DIRECTORY/datasets/` directory, and run the `python3 eval/display_table.py` to get all of the results.
