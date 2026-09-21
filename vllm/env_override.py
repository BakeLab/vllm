# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
import importlib.util
import os


def _get_torch_cuda_version():
    """Peripheral function to _maybe_set_cuda_compatibility_path().
    PyTorch version must not be determined by importing directly
    because it will trigger the CUDA initialization, losing the
    chance to set the LD_LIBRARY_PATH beforehand.
    """
    try:
        spec = importlib.util.find_spec("torch")
        if not spec:
            return None
        if spec.origin:
            torch_root = os.path.dirname(spec.origin)
        elif spec.submodule_search_locations:
            torch_root = spec.submodule_search_locations[0]
        else:
            return None
        version_path = os.path.join(torch_root, "version.py")
        if not os.path.exists(version_path):
            return None
        # Load the version module without importing torch
        ver_spec = importlib.util.spec_from_file_location("torch.version", version_path)
        if not ver_spec or not ver_spec.loader:
            return None
        module = importlib.util.module_from_spec(ver_spec)
        # Avoid registering in sys.modules to not confuse future imports
        ver_spec.loader.exec_module(module)
        return getattr(module, "cuda", None)
    except Exception:
        return None


def _maybe_set_cuda_compatibility_path():
    """Set LD_LIBRARY_PATH for CUDA forward compatibility if enabled.

    Must run before 'import torch' since torch loads CUDA shared libraries
    at import time and the dynamic linker only consults LD_LIBRARY_PATH when
    a library is first loaded.

    CUDA forward compatibility is only supported on select professional and
    datacenter NVIDIA GPUs. Consumer GPUs (GeForce, RTX) do not support it
    and will get Error 803 if compat libs are loaded.
    """
    enable = os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower() in (
        "1",
        "true",
    )
    if not enable:
        return

    cuda_compat_path = os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH", "")
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_compat = os.path.join(conda_prefix, "cuda-compat")
        if conda_prefix and os.path.isdir(conda_compat):
            cuda_compat_path = conda_compat
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):
        torch_cuda_version = _get_torch_cuda_version()
        if torch_cuda_version:
            default_path = f"/usr/local/cuda-{torch_cuda_version}/compat"
            if os.path.isdir(default_path):
                cuda_compat_path = default_path
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):
        return

    norm_path = os.path.normpath(cuda_compat_path)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = existing.split(os.pathsep) if existing else []

    if ld_paths and ld_paths[0] and os.path.normpath(ld_paths[0]) == norm_path:
        return  # Already at the front

    new_paths = [norm_path] + [
        p for p in ld_paths if not p or os.path.normpath(p) != norm_path
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_paths)


_maybe_set_cuda_compatibility_path()

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# see https://github.com/vllm-project/vllm/issues/10480 and
# https://github.com/vllm-project/vllm/issues/10619.
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# Enable Triton autotuning result caching to disk by default.
# Without this, Triton re-runs autotuning on every process restart,
# adding significant latency to the first inference request.
# This writes autotuning results to TRITON_CACHE_DIR.
# It can still be overridden by setting TRITON_CACHE_AUTOTUNING=0
# in the environment.
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

# When unset, TileLang routes JIT temp dirs through a world-shared
# /tmp/tvm-debug-mode-tempdirs/ whose ownership is pinned to whichever
# user compiled first, breaking every other user on a shared host.
# Opt into per-process tempdirs unless the user explicitly chose the
# debug layout (see https://github.com/vllm-project/vllm/issues/41410).
os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

# ============================================================
# Inductor FALLBACK_ALLOW_LIST fast-path for vllm::*/vllm_aiter::* ops
# ============================================================
# When Inductor encounters a custom op without a registered lowering or
# decomposition (e.g. vllm::all_reduce, vllm_aiter::fused_add_rms_norm) it
# correctly creates an implicit fallback that calls into the eager Python
# impl. However, unless `base_name` (e.g. "vllm::all_reduce") is in
# torch._inductor.lowering.FALLBACK_ALLOW_LIST, GraphLowering.call_function
# (torch/_inductor/graph.py:~1283) takes the slow path that emits
#   log.info("Creating implicit fallback for:\n%s",
#            error.operator_str(target, args, kwargs))
# `operator_str` eagerly recurses through __str__ on every input TensorBox;
# for deep MoE/TP graphs (e.g. Kimi-K2.6 at TP=8) the IR provenance tree
# behind a TP all-reduce input or a residual-fed RMSNorm input is hundreds
# of layers deep, and stringifying it consumes many minutes of CPU per call,
# effectively hanging compilation.
#
# Patching FALLBACK_ALLOW_LIST membership to also match any "vllm::*" or
# "vllm_aiter::*" base_name routes our custom ops through the fast path
# `make_fallback(target, warn=False, override_decomp=True)` instead. This
# preserves all downstream behaviour (allreduce_rms_fusion still pattern-
# matches them, partitioning still works, fallback semantics identical) but
# skips the expensive log formatting on the FIRST encounter of each op.
#
# We wrap the OrderedSet in a thin proxy that:
#   - Returns True from __contains__ for any vllm::*/vllm_aiter::* op
#   - Otherwise delegates to the underlying set (preserving membership of
#     the standard entries like "torchvision::roi_align", "aten::index_add")
#   - Forwards add()/__iter__()/__len__()/etc. so other Inductor code paths
#     that mutate or iterate the set keep working.

_VLLM_FALLBACK_NAMESPACE_PREFIXES = ("vllm::", "vllm_aiter::")


class _VllmFallbackAllowList:
    """Membership proxy that auto-allows vllm::*/vllm_aiter::* base_names."""

    _vllm_patched = True

    def __init__(self, inner):
        self._inner = inner

    def __contains__(self, item):
        if isinstance(item, str) and item.startswith(_VLLM_FALLBACK_NAMESPACE_PREFIXES):
            return True
        return item in self._inner

    def add(self, item):
        self._inner.add(item)

    def discard(self, item):
        self._inner.discard(item)

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def __repr__(self):
        return f"_VllmFallbackAllowList({self._inner!r})"

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch_inductor_fallback_allow_list() -> None:
    """Wrap torch._inductor.lowering.FALLBACK_ALLOW_LIST so any custom op in
    the ``vllm::`` or ``vllm_aiter::`` namespaces is treated as a member.

    Idempotent: a sentinel attribute on the proxy prevents re-wrapping.
    """
    try:
        from torch._inductor import lowering as _lowering
    except ImportError:
        return

    base = getattr(_lowering, "FALLBACK_ALLOW_LIST", None)
    if base is None or getattr(base, "_vllm_patched", False):
        return

    _lowering.FALLBACK_ALLOW_LIST = _VllmFallbackAllowList(base)

    # torch/_inductor/graph.py imports the symbol at module load time:
    #   from torch._inductor.lowering import FALLBACK_ALLOW_LIST
    # so we also need to overwrite the local binding in the graph module if
    # it has already been imported.
    try:
        from torch._inductor import graph as _graph

        if hasattr(_graph, "FALLBACK_ALLOW_LIST"):
            _graph.FALLBACK_ALLOW_LIST = _lowering.FALLBACK_ALLOW_LIST
    except ImportError:
        pass


_patch_inductor_fallback_allow_list()

# ============================================================
# Triton Autotuner determinism
# ============================================================
# Replace the Autotuner.run so it always pick the first running configuration.
# Useful to eliminate autotune variability leading to non determinism.
if os.environ.get("VLLM_TRITON_FORCE_FIRST_CONFIG", "0").strip().lower() in (
    "1",
    "true",
):
    from vllm.triton_utils.force_first_config import (
        install as _install_force_first_config,
    )

    _install_force_first_config()
