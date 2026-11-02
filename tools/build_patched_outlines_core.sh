#!/usr/bin/env bash
set -euo pipefail

python_executable=${1:?usage: build_patched_outlines_core.sh PYTHON [OUTPUT_DIR]}
output_dir=${2:-dist}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
temp_dir=$(mktemp -d)
host_os=$(uname -s)
host_arch=$(uname -m)
trap 'rm -rf "$temp_dir"' EXIT

case "$host_os/$host_arch" in
    Linux/x86_64)
        wheel_platform=manylinux_2_34_x86_64
        ;;
    Darwin/arm64)
        ;;
    *)
        echo "Unsupported Outlines wheel build host: $host_os/$host_arch" >&2
        echo "Supported hosts are Linux x86_64 and macOS arm64." >&2
        exit 2
        ;;
esac

uv run \
    --no-project \
    --with pip \
    --with maturin \
    --python "$python_executable" \
    python -m pip download \
        --no-deps \
        --no-binary outlines-core \
        --no-build-isolation \
        --dest "$temp_dir" \
        outlines-core==0.2.14

tar -xzf "$temp_dir/outlines_core-0.2.14.tar.gz" -C "$temp_dir"
source_dir="$temp_dir/outlines_core-0.2.14"
patch -p1 -d "$source_dir" \
    < "$repo_root/patches/outlines-core/0.2.14-python315.patch"

build_output_dir="$temp_dir/wheels"
mkdir -p "$build_output_dir" "$output_dir"
uv build \
    --wheel \
    --python "$python_executable" \
    --out-dir "$build_output_dir" \
    "$source_dir"

built_wheel=$(find "$build_output_dir" -maxdepth 1 -name '*.whl' -print -quit)
if [[ -z "$built_wheel" ]]; then
    echo "No Outlines wheel was built." >&2
    exit 1
fi

if [[ "$host_os/$host_arch" == Linux/x86_64 ]]; then
    repaired_wheel_dir="$temp_dir/repaired-wheels"
    mkdir -p "$repaired_wheel_dir"
    uv run \
        --no-project \
        --with auditwheel \
        --python "$python_executable" \
        auditwheel repair \
        --plat "$wheel_platform" \
        --wheel-dir "$repaired_wheel_dir" \
        "$built_wheel"
    wheel=$(find "$repaired_wheel_dir" -maxdepth 1 -name '*.whl' -print -quit)
else
    wheel=$built_wheel
fi

if [[ -z "$wheel" ]]; then
    echo "No patched Outlines wheel was produced." >&2
    exit 1
fi

cp "$wheel" "$output_dir/"
wheel="$output_dir/${wheel##*/}"

uv pip install \
    --python "$python_executable" \
    --no-deps \
    --reinstall \
    "$wheel"
