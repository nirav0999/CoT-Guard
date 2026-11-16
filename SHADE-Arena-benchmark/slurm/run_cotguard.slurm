#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --job-name=bench-cotguard
#SBATCH --output=logs/%x_%j.log
# Submit from the package root, after: mkdir -p logs
#   sbatch --account=ACCOUNT --partition=PARTITION slurm/run_cotguard.slurm
# Required environment: VENV (virtual environment folder), HF_HOME (Hugging Face cache folder).
# Optional: MODULES (modules to load), PILOT (number of transcripts, quick test), SEED.
: "${VENV:?set VENV to the virtual environment folder}"
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
if [ -n "$MODULES" ]; then module load $MODULES; fi
source "$VENV/bin/activate"
export HF_HOME
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
set -e
cd "$SLURM_SUBMIT_DIR"
which python3
for C in strip keep; do
  python3 src/run_monitor.py --model cotguard --condition $C ${PILOT:+--pilot $PILOT} ${SEED:+--seed $SEED}
done
