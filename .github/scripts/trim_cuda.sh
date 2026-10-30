#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export UV_CONSTRAINT=/workspace/cuda-ci-results/constraints.txt
export PYTORCH_VERSION
PYTORCH_VERSION=$(/opt/venv/bin/python -c 'import torch; print(torch.__version__)')
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ' ' -)
[[ "$TORCH_CUDA_ARCH_LIST" =~ ^[0-9]+\.[0-9]+(\ [0-9]+\.[0-9]+)*$ ]]
uv pip install --python /opt/venv/bin/python --no-build-isolation --no-deps -v ./vision

export VLLM_TORCH_VERSION_OVERRIDE="$PYTORCH_VERSION"
export VLLM_TORCHVISION_VERSION_OVERRIDE
VLLM_TORCHVISION_VERSION_OVERRIDE=$(/opt/venv/bin/python -c 'import torchvision; print(torchvision.__version__)')
printf 'torchvision==%s\n' "$VLLM_TORCHVISION_VERSION_OVERRIDE" >> "$UV_CONSTRAINT"
printf 'torch\ntorchvision\ntorchaudio\n' > /workspace/cuda-ci-results/frameworks.txt

cd vllm
uv pip install --python /opt/venv/bin/python \
    -r requirements/build/cuda.txt -r requirements/cuda.txt -r requirements/test/cuda.in \
    --excludes /workspace/cuda-ci-results/frameworks.txt
export VLLM_USE_PRECOMPILED=0
export CMAKE_ARGS='-DVLLM_X86_BASELINE=x86-64-v3 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++'
uv pip install --python /opt/venv/bin/python --no-build-isolation --no-deps -v -e .

/opt/venv/bin/python - <<'PY'
from pathlib import Path
import torch
import vllm
import vllm._C

expected = Path("/workspace/cuda-ci-results/pytorch.sha").read_text().strip()
if torch.version.git_version != expected:
    raise RuntimeError(f"PyTorch was replaced: expected {expected}, got {torch.version.git_version}")
if not Path(vllm.__file__).resolve().is_relative_to(Path("/workspace/vllm")):
    raise RuntimeError(f"vLLM is not imported from this checkout: {vllm.__file__}")
x = torch.ones(16, device="cuda")
torch.testing.assert_close(x + x, torch.full_like(x, 2))
torch.cuda.synchronize()
print(f"Testing vLLM with PyTorch {torch.__version__} at {torch.version.git_version}")
PY
