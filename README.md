# CoT-Guard: Small Models for Strong Monitoring

[arXiv Paper](https://arxiv.org/abs/2605.12746) • [Models](https://huggingface.co/CoT-Guard) • [Data & Results](https://drive.google.com/drive/folders/1eOg5--gmDHXLwIt17RyGC3FHxN-17y9q?usp=drive_link)

[About](#about) • [Documents](#documents) • [Citation](#citation) • [Acknowledgement](#acknowledgement)

**Accepted at NeurIPS 2026.**

**[AdvML-Frontiers × CoTMA: From Model Security to Compositional Threats in Multi-Agent AI Systems @ COLM 2026 (Oral)](https://advml-frontier.github.io/)**

<a id="about"></a>

## About

CoT-Guard is a 4B-parameter monitor trained to detect hidden objectives in code-generation tasks using chain-of-thought and generated code. This repository contains the training, evaluation, and data curation code, with SFT and RL model checkpoints available on [Hugging Face](https://huggingface.co/CoT-Guard).

Read [CoT-Guard: Small Models for Strong Monitoring](https://arxiv.org/abs/2605.12746) for the method and evaluation details.

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

### Run Attack

a) Create prompts containing the main task (`bigcodebench`), side task (`varname`), and attack policy (`baseline`):

```bash
python3 task_creation/for_attack.py --main_task=bigcodebench --side_tasks=varname --attack_policies=baseline
```

b) Run the tasks with `Qwen/Qwen3-8B` on GPU 0 to generate one attack response per prompt:

```bash
python3 gen/attack.py single --main_task=bigcodebench --side_task=varname --attack_policy=baseline --model=Qwen/Qwen3-14B --devices=0 --tp=1 --num_rollouts=5
```

### Verify

With Docker running:

```bash
python3 eval/verify_task.py --main_task=bigcodebench --side_task=varname --attack_policy=baseline --attack_model=Qwen/Qwen3-14B --num_rollouts=5
```

Output: `$COT_GLOBAL_DIRECTORY/results/attack/bigcodebench.varname.baseline/Qwen3-14B.5.tested.jsonl`

### Monitor

a) Create monitor prompts from the attacks that were successful:

```bash
python3 task_creation/for_monitor.py --main_task=bigcodebench --side_tasks=varname --attack_policies=baseline --model=Qwen/Qwen3-14B --attack_rollouts=5 --filter_mode=side-all --monitor_policy=main_aware --monitor_type=cot_action
```

b) Run the monitor with `Qwen/Qwen3-8B`:

```bash
python3 gen/monitor.py single --main_task=bigcodebench --side_task=varname --attack_policy=baseline --attack_model=Qwen/Qwen3-14B --attack_rollouts=5 --filter_mode=side-all --monitor_policy=main_aware --monitor_model=Qwen/Qwen3-8B --monitor_type=cot_action --devices=0 --tp=1 --num_rollouts=1
```

### Results

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

<a id="documents"></a>

## Documents

- [Paper](https://arxiv.org/abs/2605.12746): method, threat model, and evaluation.
- [Inference pipeline](#inference-pipeline): generate attacks, verify tasks, run monitors, and score results.
- [Training](#training): prepare data and train with SFT and RL / GRPO.
- [Stored evaluation and results](#stored-eval-and-results): download datasets and reproduce result tables.
- [Model checkpoints](https://huggingface.co/CoT-Guard): SFT and RL monitors.

## Training

The [paper](https://arxiv.org/pdf/2605.12746) uses I/O Delete, I/O Download,
and I/O Exfiltrate Secret for SFT, then Default Value, Complexity, and Variable
Name for RL. For each group, prepare verified `baseline` and
`no_side_task` monitor outputs from Qwen3-14B attacks and a Qwen3-32B monitor,
with all three scopes (`--monitor_type=all`).

### SFT

Install [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory), then collect
the I/O task examples:

```bash
python3 train/sft/collect.py --main-tasks kodcode \
  --side-tasks iodelete,ioexfil,iodownload --attack-rollouts 5
```

The collector registers three datasets in `dataset_info.json`. Edit
`train/sft/config/qwen3_full_sft_three.yaml` before training: replace the
example `deepspeed`, `dataset_dir`, and `output_dir` paths with absolute paths.
`$COT_GLOBAL_DIRECTORY` in YAML is literal text and will not expand. Check the
dataset and model names if you use different sources. Then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train train/sft/config/qwen3_full_sft_three.yaml
```

### RL / GRPO

Collect training prompts from the three subtle tasks:

```bash
python3 train/rl/collect.py --main-tasks bigcodebench \
  --side-tasks defaultval,complexity,varname --set-type train
```

Install [veRL](https://github.com/verl-project/verl) in a separate training
environment and activate it. Before running, edit `DATA_ROOT`, `TRAIN_FILE`,
`MODEL_PATH`, `CHECKPOINT_DIR`, `GPU_IDS`, and `EXPERIMENT` at the top
of `train/rl/run_grpo.sh`. Use your collected JSONL paths and set `MODEL_PATH`
to the same checkpoint as `output_dir` in the edited SFT YAML. The launcher
disables validation. Then run:

```bash
bash train/rl/run_grpo.sh
```

## Stored Eval and Results

All our results and datasets available for download from [here](https://drive.google.com/drive/folders/1eOg5--gmDHXLwIt17RyGC3FHxN-17y9q?usp=drive_link).

After downloading and extracting the files, place datasets under
`$COT_GLOBAL_DIRECTORY/datasets/` and evaluation outputs under
`$COT_GLOBAL_DIRECTORY/results/`. The display script reads outputs from
`results/monitor/`; preserve the downloaded subdirectory structure.

From the repository root, with the project environment activated, run the
following commands. Replace the example paths below; `COT_DOWNLOAD_DIRECTORY`
should point to the extracted folder containing `datasets/` and `results/`.

```bash
export COT_GLOBAL_DIRECTORY="/path/to/data_and_results"
export COT_DOWNLOAD_DIRECTORY="/path/to/extracted_download"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$COT_GLOBAL_DIRECTORY/datasets" "$COT_GLOBAL_DIRECTORY/results"
cp -a "$COT_DOWNLOAD_DIRECTORY/datasets/." "$COT_GLOBAL_DIRECTORY/datasets/"
cp -a "$COT_DOWNLOAD_DIRECTORY/results/." "$COT_GLOBAL_DIRECTORY/results/"

# Display tables for the default evaluation settings.
python3 eval/display_table.py

# Include all configured side tasks for those settings.
python3 eval/display_table.py --side_tasks=None
```

Use `--monitor_models`, `--attack_policy`, and `--monitor_objective` to select
other downloaded evaluation settings; the command does not automatically
discover every experiment. See the [results commands](#results) for an example.

<a id="citation"></a>

## Citation

If you use CoT-Guard in your research, please cite our [paper](https://openreview.net/forum?id=mURtPVmTjF):

```bibtex
@inproceedings{diwan2026cotguard,
  title = {CoT-Guard: Small Models for Strong Monitoring},
  author = {Nirav Diwan and Han Wang and Berkcan Kapusuzoglu and Ramin Moradi and Supriyo Chakraborty and Giri Iyengar and Sambit Sahu and Huan Zhang and Gang Wang},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year = {2026},
  url = {https://openreview.net/forum?id=mURtPVmTjF}
}
```

<a id="acknowledgement"></a>

## Acknowledgement

We thank the developers and maintainers of [BigCodeBench](https://github.com/bigcode-project/bigcodebench), [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), [vLLM](https://github.com/vllm-project/vllm), [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory), and [veRL](https://github.com/verl-project/verl) for the open-source tools used in this project.
