#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Edit these values before running. Use absolute paths for local files.
DATA_ROOT="/path/to/data_and_results"
TRAIN_FILE="${DATA_ROOT}/datasets/rl/action_only_cot_only_cot_action.bigcodebench.defaultval_complexity_varname.train.jsonl"
MODEL_PATH="${DATA_ROOT}/checkpoints/sft/qwen3-4b"
EXPERIMENT="cot-guard-grpo"
CHECKPOINT_DIR="${DATA_ROOT}/checkpoints/rl/${EXPERIMENT}"
ROLLOUT_LOG_DIR="${CHECKPOINT_DIR}/rollout_logs"
GPU_IDS="0,1,2,3,4,5,6,7"
CHECKPOINT_SAVE_CONTENTS='["hf_model"]'
LOGGER_BACKENDS='["console","wandb"]'

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}
test -f "${TRAIN_FILE}"
test -f "${PROJECT_DIR}/train/rl/reward_fn.py"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# veRL builds a validation loader even when evaluation is disabled.
# Reuse the training file only to satisfy that initialization requirement.
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
  +actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=20480 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.train_batch_size=256 \
  data.max_prompt_length=16384 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  custom_reward_function.path="${PROJECT_DIR}/train/rl/reward_fn.py" \
  custom_reward_function.name=compute_score \
  trainer.total_epochs=4 \
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
  trainer.project_name=cot-guard \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.save_freq=10 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.rollout_data_dir="${ROLLOUT_LOG_DIR}" \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}"
