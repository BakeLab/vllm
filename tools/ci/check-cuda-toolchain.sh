#!/usr/bin/env bash
set -euo pipefail

[[ "$("$CC" -dumpversion)" == 16 ]]
[[ "$("$CXX" -dumpversion)" == 16 ]]
"$CXX" --version
"$CUDACXX" --version

smoke_dir=$(mktemp -d)
trap 'rm -rf "$smoke_dir"' EXIT
cat > "$smoke_dir/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(vllm_toolchain_smoke LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=x86-64-v3" HAS_X86_BASELINE)
if(NOT HAS_X86_BASELINE)
  message(FATAL_ERROR "CXX26 x86-64-v3 compiler probe failed")
endif()
add_library(smoke OBJECT smoke.cpp smoke.cu)
EOF
printf 'int host_probe() { return 0; }\n' > "$smoke_dir/smoke.cpp"
printf '__global__ void cuda_probe() {}\n' > "$smoke_dir/smoke.cu"
cmake -S "$smoke_dir" -B "$smoke_dir/build" -G Ninja \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$smoke_dir/build"
