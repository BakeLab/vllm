#!/usr/bin/env bash
set -euo pipefail

python_executable=${1:?usage: build_patched_outlines_core.sh PYTHON [OUTPUT_DIR]}
output_dir=${2:-dist}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

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

mkdir -p "$output_dir"
uv build \
    --wheel \
    --python "$python_executable" \
    --out-dir "$output_dir" \
    "$source_dir"
uv pip install \
    --python "$python_executable" \
    --no-deps \
    --reinstall \
    "$output_dir"/outlines_core-0.2.14+bakelab.1-*.whl
