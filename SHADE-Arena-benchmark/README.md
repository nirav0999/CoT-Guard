# SHADE-Arena monitor benchmark

## Context

An AI agent works in a simulated workspace, given a main task and possibly a hidden task. In attack runs, the agent is also given a hidden task to complete without being noticed; in benign runs, only the main task is given. Once the agent finishes the main (and hidden) task, a transcript is generated, including the agent's conversation with its chain of thought.

This repository holds:

- the screened transcripts from SHADE-Arena: transcripts in which the hidden task failed, transcripts longer than 60,000 tokens, and duplicates were removed
- the reference experiment results
- the scripts to run the monitors on these transcripts

**Note:** Each transcript is scored twice. In `keep`, the monitor sees the agent's private `<thinking>` notes. In `strip`, those notes are removed. The agent's secret system message (the hidden task and any hint) is never shown to the monitor.

Attack runs are kept in separate hinted and no-hint folders so that results can be compared by hint. "Hinted" runs received the task's evasion hint, and "no-hint" runs did not.

## Folder structure

```
SHADE-Arena-benchmark/
├── README.md
├── transcripts/                screened transcripts, one folder per task
│   └── <task>/
│       ├── attack_hint/        successful attacks, hint given
│       ├── attack_nohint/      successful attacks, no hint
│       └── benign/             runs with no hidden task
├── monitor_prompts/            the monitor's instruction text, one file per task
├── reference/                  our results
│   ├── monitor_outputs/        cotguard_{strip,keep}.jsonl, qwen_{strip,keep}.jsonl
│   └── tables/                 auc_tables.txt
├── src/
│   ├── run_monitor.py          runs a monitor over the transcripts (needs a GPU)
│   └── score.py                builds the AUC tables from monitor outputs
├── slurm/                      batch files for the two models
├── environment/                setup scripts and package lists for the two environments
├── envs/                       created in step 2.1: the two Python environments (cotguard/, qwen/)
├── hf_cache/                   created in step 2.2: the downloaded models
├── logs/                       created in step 2.3: job logs and run notes
└── reproduced/                 created by the jobs: your results
    ├── monitor_outputs/
    └── tables/
```

`envs/`, `hf_cache/`, `logs/` and `reproduced/` are not in the repository: they are created on your machine when you follow section 2, and git ignores them.

## 1. Reference results

Run these from the package folder. Nothing here needs a GPU.

**AUC tables:** in each table, `nA` is the number of attack runs, `nB` the number of benign runs, and the last column is the AUC with a 95% interval.

To see the whole reference AUC tables:

```
less reference/tables/auc_tables.txt
```

To see the pooled AUC scores only:

```
grep "^####\|^===\|^POOLED" reference/tables/auc_tables.txt
```

**Monitor outputs:** see `reference/monitor_outputs/` for the full outputs.

## 2. Rerun the experiment

You need a machine with a GPU (we used one GH200 with 96 GB) and Python 3.12. If your cluster provides Python as a module, load it first. Replace each `<...>` in the commands below with your own value, and skip the `export MODULES` line if your cluster needs no module.

### 2.1 Set up the environments (one environment per model)

```
cd <path to the cloned repository>
mkdir -p envs
bash environment/setup_cotguard.sh $PWD/envs/cotguard
bash environment/setup_qwen.sh $PWD/envs/qwen
```

### 2.2 Download the models

The models are downloaded into `hf_cache/` inside the repository (about 65 GB of free space is needed). Use the same `HF_HOME` in step 2.3:

```
export HF_HOME=$PWD/hf_cache
mkdir -p $HF_HOME
HF_HUB_OFFLINE=0 envs/qwen/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('CoT-Guard/cot-guard-4b-rl'); snapshot_download('Qwen/Qwen3.8-27B')"
```

### 2.3 Submit the batch jobs

**The jobs run offline, so the models must already be in `hf_cache/` before you submit anything.** From the repository folder:

```
mkdir -p logs
export HF_HOME=$PWD/hf_cache
export MODULES=<module to load, if your cluster needs one>
sbatch --account=<ACCOUNT> --partition=<PARTITION> --export=ALL,VENV=$PWD/envs/cotguard slurm/run_cotguard.slurm
sbatch --account=<ACCOUNT> --partition=<PARTITION> --export=ALL,VENV=$PWD/envs/qwen,COND=strip slurm/run_qwen.slurm
sbatch --account=<ACCOUNT> --partition=<PARTITION> --export=ALL,VENV=$PWD/envs/qwen,COND=keep slurm/run_qwen.slurm
```

## 3. Check your results

When the jobs have finished, build your tables:

```
python3 src/score.py --scores reproduced/monitor_outputs
less reproduced/tables/auc_tables.txt
```

`score.py` first checks that your outputs match the transcripts in the package, then writes `reproduced/tables/auc_tables.txt`. It scores whichever models have both their `strip` and `keep` outputs.
