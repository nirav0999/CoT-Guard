#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --job-name=bench-qwen
#SBATCH --output=logs/%x_%j.log
# Submit from the package root, after: mkdir -p logs
#   sbatch --account=ACCOUNT --partition=PARTITION --export=ALL,COND=strip slurm/run_qwen.slurm
# Required environment: VENV, HF_HOME, COND (strip or keep).
# Optional: MODULES (modules to load), PILOT (number of transcripts, quick test), SEED.
: "${VENV:?set VENV to the virtual environment folder}"
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
: "${COND:?set COND to strip or keep}"
if [ -n "$MODULES" ]; then module load $MODULES; fi
source "$VENV/bin/activate"
export HF_HOME
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
set -e
cd "$SLURM_SUBMIT_DIR"
which python3
python3 src/run_monitor.py --model qwen --condition $COND ${PILOT:+--pilot $PILOT} ${SEED:+--seed $SEED}
