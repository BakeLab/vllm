# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
import importlib.util
import os

from packaging.version import Version


def _get_torch_root():
    """Locate the installed torch package without importing it."""
    spec = importlib.util.find_spec("torch")
    if not spec:
        return None
    if spec.origin:
        return os.path.dirname(spec.origin)
    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None


def _get_torch_version_attr(attr):
    """Read an attribute of torch.version without importing torch.

    PyTorch version must not be determined by importing directly
    because it will trigger the CUDA initialization, losing the
    chance to set the LD_LIBRARY_PATH beforehand.
    """
    try:
        torch_root = _get_torch_root()
        if not torch_root:
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
        return getattr(module, attr, None)
    except Exception:
        return None


def _get_torch_cuda_version():
    """Peripheral function to _maybe_set_cuda_compatibility_path()."""
    return _get_torch_version_attr("cuda")


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


def _maybe_promote_torch_symbols_for_rocm():
    """Put libtorch_cpu.so in the global symbol scope on ROCm, for GPU profiling.

    Must run before 'import torch'. libkineto advertises itself to
    rocprofiler-sdk by exporting rocprofiler_configure from libtorch_cpu.so,
    and rocprofiler-sdk looks for that symbol exactly once, from a lazy
    initializer that runs *during* 'import torch'. CPython loads extension
    modules RTLD_LOCAL, so unless something has already placed the symbol in
    the global scope the lookup misses and no client is ever registered.

    The failure is silent and total: no queue interception, so roctracer
    yields no GPU records, while torch.profiler still writes a complete
    CPU-only trace and reports success.

    Best effort. If the library cannot be loaded early we leave the process
    exactly as it would have been, losing only GPU tracing.
    """
    if _get_torch_version_attr("hip") is None:
        return
    # TheRock ships ROCm libraries as Python packages. Preloading
    # libtorch_cpu.so globally before importing torch makes its LLVM
    # command-line options visible to another LLVM copy loaded by the SDK;
    # LLVM then aborts on duplicate option registration. Preserve normal
    # inference startup for this layout and leave profiler symbol promotion
    # disabled there.
    if importlib.util.find_spec("rocm_sdk") is not None:
        return
    try:
        import ctypes
        import glob

        torch_root = _get_torch_root()
        if not torch_root:
            return
        lib = os.path.join(torch_root, "lib", "libtorch_cpu.so")
        if not os.path.exists(lib):
            return
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Some ROCm wheel layouts leave a stale RUNPATH in libtorch_cpu.so,
            # so a dependency that resolves once torch has set things up does
            # not resolve for a standalone load. Satisfy it by search, then
            # retry; give up quietly if it is not where we expect.
            site_root = os.path.dirname(torch_root)
            deps = glob.glob(
                os.path.join(
                    site_root, "_rocm_sdk_*", "lib", "*", "lib", "librocm-openblas.so.*"
                )
            )
            if not deps:
                return
            ctypes.CDLL(deps[0], mode=ctypes.RTLD_LOCAL)
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


_maybe_set_cuda_compatibility_path()
_maybe_promote_torch_symbols_for_rocm()

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


def _apply_inductor_pattern_matcher_patch() -> None:
    """Allow custom ops and functionalization wrappers with unsupported dtypes."""
    from torch._inductor import pattern_matcher
    from torch._inductor.lowering import fallback_node_due_to_unsupported_type

    def fallback_for_builtin(node, allow_cpu_inputs=True):
        return (
            isinstance(node.target, torch._ops.OpOverload)
            and torch._library.utils.is_builtin(node.target)
            and fallback_node_due_to_unsupported_type(node, allow_cpu_inputs)
        )

    pattern_matcher.fallback_node_due_to_unsupported_type = fallback_for_builtin


def _patch_inductor_pattern_matcher() -> None:
    """Apply the backport when Inductor imports its pattern matcher."""
    import sys

    target_name = "torch._inductor.pattern_matcher"
    if target_name in sys.modules:
        _apply_inductor_pattern_matcher_patch()
        return

    import importlib.abc

    class _PatternMatcherPatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != target_name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def _exec_then_patch(module):
                original_exec(module)
                _apply_inductor_pattern_matcher_patch()

            spec.loader.exec_module = _exec_then_patch  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _PatternMatcherPatchFinder())


# Remove once the minimum supported torch includes
# https://github.com/pytorch/pytorch/pull/196013.
if Version(torch.__version__) < Version("2.16.0.dev"):
    _patch_inductor_pattern_matcher()

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
