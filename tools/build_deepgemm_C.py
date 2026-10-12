# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's TORCH_LIBRARY extension and copy vendored artifacts.

DeepGEMM now registers ops via TORCH_LIBRARY into ``deep_gemm._C_extension``
(abi3) and exposes the legacy API through ``deep_gemm/_C.py``. This driver
delegates to DeepGEMM's ``setup.py build_ext --inplace`` and copies the shim
plus extension into the cmake output directory.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR>
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
_pkg = src / "deep_gemm"
out.mkdir(parents=True, exist_ok=True)

if not (_pkg / "_C.py").is_file():
    sys.exit(
        f"DeepGEMM source at {src} is missing deep_gemm/_C.py; "
        "expected TORCH_LIBRARY migration layout"
    )

# Avoid DeepGEMM's clean-git assertion when vendoring a local dirty tree.
env = os.environ.copy()
env.pop("DG_SKIP_CUDA_BUILD", None)

# The nested setuptools build does not inherit CMake's compiler launcher.
# Route its C/C++ compilations through ccache when it is available; this is
# particularly useful when the same DeepGEMM binding is rebuilt for a new
# vLLM wheel with unchanged compiler inputs.
ccache = shutil.which("ccache")
if ccache and env.get("CCACHE_DISABLE") != "1":
    ccache_enabled = False
    for compiler_var in ("CC", "CXX"):
        compiler = env.get(compiler_var)
        if not compiler:
            continue
        command = shlex.split(compiler)
        if command and Path(command[0]).name not in {"ccache", "sccache"}:
            env[compiler_var] = shlex.join([ccache, *command])
            ccache_enabled = True
    if ccache_enabled:
        print("[build_deepgemm_C] using ccache for host compilation", flush=True)

print(f"[build_deepgemm_C] building in {src} with {sys.executable}", flush=True)
# CMake decides when to rebuild; setuptools would only check python_api.cpp.
subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
    cwd=src,
    env=env,
)

shim = _pkg / "_C.py"
shutil.copy2(shim, out / shim.name)

so = _pkg / "_C_extension.abi3.so"
if not so.is_file():
    sys.exit(f"DeepGEMM build did not produce {so}")
for stale in out.glob("_C_extension*.so"):
    stale.unlink()
shutil.copy2(so, out / so.name)
print(f"[build_deepgemm_C] installed {so.name} -> {out}", flush=True)
