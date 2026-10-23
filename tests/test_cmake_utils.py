# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import shutil
import subprocess
import sys
from pathlib import Path


def test_deepgemm_refreshes_pin_and_preserves_explicit_source(tmp_path, monkeypatch):
    """Reconfigure a real local Git dependency without CUDA or network access."""
    monkeypatch.delenv("DEEPGEMM_SRC_DIR", raising=False)
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(upstream), *args], text=True
        ).strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("config", "commit.gpgsign", "false")
    package = upstream / "deep_gemm"
    package.mkdir()
    (package / "_C.py").write_text("# old revision\n")
    for name in ("cutlass", "deep_jit"):
        dependency = upstream / "third-party" / name
        dependency.mkdir(parents=True)
        (dependency / "placeholder").touch()
    git("add", ".")
    git("commit", "-m", "old")
    old_pin = git("rev-parse", "HEAD")
    (package / "locality_domain.py").write_text("# new revision\n")
    git("add", ".")
    git("commit", "-m", "new")
    new_pin = git("rev-parse", "HEAD")

    source = tmp_path / "project"
    source.mkdir()
    module = source / "deepgemm.cmake"
    original = (
        Path(__file__).parents[1] / "cmake/external_projects/deepgemm.cmake"
    ).read_text()
    # Only redirect the external dependency; execute the real configure logic.
    local_module = re.sub(
        r'set\(_DEEPGEMM_UPSTREAM_REPO "[^"]+"\)',
        f'set(_DEEPGEMM_UPSTREAM_REPO "{upstream.as_posix()}")',
        original,
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\nproject(test NONE)\n"
        "set(CMAKE_CUDA_COMPILER_VERSION 13.4)\n"
        "function(cuda_archs_loose_intersection out)\n"
        '  set(${out} "" PARENT_SCOPE)\nendfunction()\n'
        'include("${CMAKE_CURRENT_SOURCE_DIR}/deepgemm.cmake")\n'
    )
    build = tmp_path / "build"

    def configure(pin, *args):
        module.write_text(
            re.sub(
                r'set\(_DEEPGEMM_UPSTREAM_TAG "[^"]+"\)',
                f'set(_DEEPGEMM_UPSTREAM_TAG "{pin}")',
                local_module,
            )
        )
        subprocess.run(
            [
                _get_cmake_bin(),
                "-G",
                "Ninja",
                "-S",
                str(source),
                "-B",
                str(build),
                *args,
            ],
            check=True,
        )

    configure(old_pin)
    cached = build / "_deps/deepgemm-src/deep_gemm"
    assert not (cached / "locality_domain.py").exists()
    configure(new_pin)
    assert (cached / "locality_domain.py").read_text() == "# new revision\n"
    unchanged = (cached / "_C.py").stat().st_mtime_ns
    upstream.rename(tmp_path / "offline")
    configure(new_pin)
    assert (cached / "_C.py").stat().st_mtime_ns == unchanged
    # An explicit source may be an archive with no .git, and must not be reset.
    override = tmp_path / "override"
    override.mkdir()
    marker = override / "user.txt"
    marker.write_text("preserve me")
    configure(old_pin, f"-DDEEPGEMM_SRC_DIR={override.as_posix()}")
    assert marker.read_text() == "preserve me"
    assert not (override / ".git").exists()
    assert (cached / "locality_domain.py").exists()
    monkeypatch.setenv("DEEPGEMM_SRC_DIR", str(override))
    configure(old_pin, "-DDEEPGEMM_SRC_DIR=nonexistent")
    assert marker.read_text() == "preserve me"


def _get_cmake_bin() -> str:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    venv_cmake = Path(sys.executable).parent / "cmake"
    if venv_cmake.is_file():
        return str(venv_cmake)
    return "cmake"


def test_exact_family_arch_precedes_generic_family_fallback(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_cuda_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(
  actual "10.0f;10.7f" "10.7")
if(NOT "${{actual}}" STREQUAL "10.7f")
  message(FATAL_ERROR "Expected 10.7f, got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_extract_archs_prefers_sass_target_over_corrupted_virtual_arch(
    tmp_path: Path,
):
    """Torch's autodetection can emit a bogus arch=compute_* half (e.g.
    capability 12.1 corrupted to arch=compute_20,code=sm_121); the SASS
    target must win, while PTX-only entries keep the virtual arch."""
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_extract_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
extract_unique_cuda_archs_ascending(actual
  "-gencode arch=compute_20,code=sm_121;\
-gencode arch=compute_80,code=sm_80;\
-gencode arch=compute_80,code=compute_80")
if(NOT "${{actual}}" STREQUAL "8.0;12.1")
  message(FATAL_ERROR "Expected '8.0;12.1', got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_clear_cuda_gencode_flags(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_clear_flags.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
set(CMAKE_CUDA_FLAGS "-Wall -gencode arch=compute_80,code=sm_80")
clear_cuda_gencode_flags(CUDA_ARCH_FLAGS)
if(NOT "${{CMAKE_CUDA_FLAGS}}" STREQUAL "-Wall ")
  message(FATAL_ERROR "Expected '-Wall ', got '${{CMAKE_CUDA_FLAGS}}'")
endif()
if(NOT "${{CUDA_ARCH_FLAGS}}" STREQUAL "-gencode arch=compute_80,code=sm_80")
  message(FATAL_ERROR "Expected '-gencode arch=compute_80,code=sm_80', "
    "got '${{CUDA_ARCH_FLAGS}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)
