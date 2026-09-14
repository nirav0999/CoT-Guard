#!/bin/bash
# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "=== Installing uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

echo "=== Creating venv ==="
uv venv env-verl --python 3.11
source env-verl/bin/activate

echo "=== Step 1: vllm (most constrained) ==="
uv pip install vllm==0.8.3

echo "=== Step 2: force torch + torchvision to cu124 (ABI=FALSE) ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
uv pip install nvidia-cusparselt-cu12

echo "=== Step 3: flash-attn (pre-built wheel, torch2.6, ABI=FALSE) ==="
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=== Step 4: verl dependencies (AFTER torch is finalized) ==="
uv pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    einops \
    hydra-core \
    numpy \
    pandas \
    peft \
    pyarrow \
    pybind11 \
    pylatexenc \
    "ray[data,train,tune,serve]" \
    tensorboard \
    tensordict \
    torchdata \
    transformers \
    wandb

echo "=== Step 5: re-force torch + torchvision (in case deps overwrote) ==="
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps

echo "=== Step 6: clone repo + install verl (editable) ==="
if [ ! -d "minimal-veRL" ]; then
    git clone --recurse-submodules git@github.com:nirav0999/minimal-veRL.git
fi
cd minimal-veRL/verl
git submodule update --init --recursive
uv pip install --no-deps -e .
cd ../..

# echo "=== Step 7: set LD_LIBRARY_PATH ==="
# NVIDIA_LIB=$(python -c "import nvidia.cusparselt; import os; print(os.path.dirname(nvidia.cusparselt.__file__) + '/lib')")
# export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== Step 8: verification ==="
python << 'PYEOF'
import sys

def check_torch():
    import torch
    abi = torch._C._GLIBCXX_USE_CXX11_ABI
    assert not abi, f"Expected cxx11abi=FALSE, got {abi}"
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"  torch=={torch.__version__}")
    print(f"  cuda=={torch.version.cuda}")
    print(f"  gpu={torch.cuda.get_device_name(0)}")
    print(f"  cxx11_abi={abi}")

def check_vllm():
    import vllm
    print(f"  vllm=={vllm.__version__}")

def check_flash_attn():
    import flash_attn
    print(f"  flash_attn=={flash_attn.__version__}")
    import torch
    from flash_attn import flash_attn_func
    b, s, h, d = 2, 128, 8, 64
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v, causal=True)
    assert out.shape == (b, s, h, d), f"unexpected shape: {out.shape}"
    print("  flash_attn_func forward: PASSED")

def check_verl():
    import verl
    print("  verl: ok")

def check_deps():
    for dep in ["accelerate", "datasets", "hydra", "pandas", "peft",
                "ray", "tensorboard", "tensordict", "torchvision",
                "transformers", "wandb"]:
        try:
            __import__(dep)
            print(f"  {dep}: ok")
        except ImportError as e:
            print(f"  {dep}: FAILED ({e})")
            sys.exit(1)

print("[torch]")
check_torch()
print("[vllm]")
check_vllm()
print("[flash_attn]")
check_flash_attn()
print("[verl]")
check_verl()
print("[deps]")
check_deps()
print()
print("All checks passed.")
PYEOF

echo "=== Done ==="
echo ""
echo "Before each session, run:"
echo "  source env-verl/bin/activate"
echo "  export LD_LIBRARY_PATH=\$(python -c \"import nvidia.cusparselt; import os; print(os.path.dirname(nvidia.cusparselt.__file__) + '/lib')\")\:\$LD_LIBRARY_PATH"
