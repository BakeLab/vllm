#!/usr/bin/env bash
# This script builds FlashInfer wheels with optional AOT kernel precompilation.

set -ex

# FlashInfer configuration
FLASHINFER_GIT_REPO="${FLASHINFER_GIT_REPO:-https://github.com/BakeLab/flashinfer.git}"
FLASHINFER_GIT_REF="${FLASHINFER_GIT_REF:-ba21cd7da3d2ae02ffde3037d775c4cf2b7bb272}"
BUILD_WHEEL="${BUILD_WHEEL:-true}"
BUILD_AOT="${BUILD_AOT:-true}"

if [[ -z "${CUDA_VERSION}" ]]; then
    echo "❌ CUDA_VERSION must be specified" >&2
    exit 1
fi

echo "🏗️  Building FlashInfer ${FLASHINFER_GIT_REF} for CUDA ${CUDA_VERSION}"

# Clone FlashInfer and its submodules. A full commit ref pins the fork source;
# branch and tag refs remain available for local builds.
if [[ "${FLASHINFER_GIT_REF}" =~ ^[[:xdigit:]]{40}$ ]]; then
    git init flashinfer
    git -C flashinfer remote add origin "${FLASHINFER_GIT_REPO}"
    git -C flashinfer fetch --depth 1 origin "${FLASHINFER_GIT_REF}"
    git -C flashinfer checkout --detach FETCH_HEAD
    git -C flashinfer submodule update --init --recursive --depth 1
else
    git clone --depth 1 --recursive --shallow-submodules \
        --branch "${FLASHINFER_GIT_REF}" \
        "${FLASHINFER_GIT_REPO}" flashinfer
fi

# Set CUDA arch list based on CUDA version
# Exclude CUDA arches for older versions (11.x and 12.0-12.7)
if [[ "${CUDA_VERSION}" == 11.* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
elif [[ "${CUDA_VERSION}" == 12.[8-9]* ]]; then
    # CUDA 12.8–12.9
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0"
else
    # CUDA 13.0+
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0f 11.0 12.0f"
fi

if [[ "${BUILD_AOT}" == "true" ]]; then
    echo "🏗️ Building FlashInfer AOT for arches: ${FI_TORCH_CUDA_ARCH_LIST}"
fi

pushd flashinfer
    # Make sure the wheel is built for the correct CUDA version
    export UV_TORCH_BACKEND=cu$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')

    if [[ "${BUILD_AOT}" == "true" ]]; then
        export TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
        export FLASHINFER_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
        python3 -m flashinfer.aot
    fi
    
    if [[ "${BUILD_WHEEL}" == "true" ]]; then
        # Build wheel for distribution
        uv build --no-build-isolation --wheel --out-dir ../flashinfer-dist .
        echo "✅ FlashInfer wheel built successfully in flashinfer-dist/"
    else
        # Install directly (for Dockerfile)
        uv pip install --system --no-build-isolation --force-reinstall .
        echo "✅ FlashInfer installed successfully"
    fi
popd

# Cleanup
rm -rf flashinfer
