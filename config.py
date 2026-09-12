# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

GLOBAL_DIRECTORY = os.environ.get(
    "COT_GLOBAL_DIRECTORY", "/srv/local/hanw14/neurips26/"
)

TEMP_DIRECTORY = Path(os.environ.get("COT_TEMP_DIRECTORY", Path.cwd()))

DEFAULT_MONITOR_MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "openai/gpt-5-mini-2025-08-07",
    "Qwen/qwen4b-action-only-defaultval_complexity_varname-rl_step_80",
]

DEFAULT_ATTACK_MODEL = "Qwen/Qwen3-14B"
DEFAULT_TEMP = 0.6
DEFAULT_MAX_TOKENS = (
    32768  # For attack models, for monitor models we default to DEFAULT_MAX_TOKENS*2
)
DEFAULT_ROLLOUTS = 5
DEFAULT_ATTACK_POLICY = "baseline"

FILTER_MODES = ["side-all"]
