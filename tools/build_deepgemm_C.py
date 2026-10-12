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


def patch_for_trimmed_torch() -> None:
    """Adapt the pinned DeepGEMM source to the trimmed PyTorch headers."""
    setup = src / "setup.py"
    setup_text = setup.read_text()
    if "-std=c++20" in setup_text:
        setup.write_text(setup_text.replace("-std=c++20", "-std=c++26"))
    elif "-std=c++26" not in setup_text:
        raise RuntimeError(f"Unsupported DeepGEMM C++ standard in {setup}")

    for path in (src / "csrc").rglob("*"):
        if path.suffix not in {".h", ".hpp", ".cpp", ".cc"}:
            continue
        text = path.read_text()
        updated = text.replace("c10::optional", "std::optional")
        updated = updated.replace("#include <c10/util/Optional.h>\n", "")
        if (
            updated != text
            and "std::optional" in updated
            and "#include <optional>" not in updated
        ):
            updated = updated.replace(
                "#pragma once\n", "#pragma once\n\n#include <optional>\n", 1
            )
        if updated != text:
            path.write_text(updated)

    runtime = src / "csrc/runtime/runtime.hpp"
    if runtime.is_file():
        text = runtime.read_text()
        include = "#include <ATen/cuda/CUDABlasLtHandle.h>"
        if include not in text:
            text = text.replace(
                "#include <ATen/cuda/CUDAContext.h>\n",
                "#include <ATen/cuda/CUDAContext.h>\n" + include + "\n",
                1,
            )
            runtime.write_text(text)


patch_for_trimmed_torch()

# Avoid DeepGEMM's clean-git assertion when vendoring a local dirty tree.
env = os.environ.copy()
env.pop("DG_SKIP_CUDA_BUILD", None)

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
