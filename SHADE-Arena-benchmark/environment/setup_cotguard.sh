#!/bin/bash
# Build the CoT-Guard environment (vLLM 0.13.0, torch 2.9.0 with CUDA 12.9 wheels).
# Usage: bash environment/setup_cotguard.sh /path/to/new/environment
# Python 3.12 is expected. On DeltaAI: module load python/miniforge3_pytorch
set -e
VENV="${1:?usage: bash setup_cotguard.sh /path/to/new/environment}"
python3 --version
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# torch first, from the CUDA wheel index (the plain PyPI ARM build has no CUDA)
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple
python -m pip install vllm==0.13.0
python -m pip list 2>/dev/null | grep -iE "^(torch|vllm|transformers) "
python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"
python -c "import vllm; print('vllm', vllm.__version__)"
echo "Done. Whether torch sees a GPU can only be checked on a compute node: python -c 'import torch; print(torch.cuda.is_available())'"
