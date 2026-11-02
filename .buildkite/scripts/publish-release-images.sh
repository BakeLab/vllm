#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Publish a release Docker image family from ECR to DockerHub.
# Pulls per-arch images, tags with latest and versioned tags, pushes them,
# then creates and pushes multi-arch manifests.

set -euo pipefail

TARGET="${1:-all}"
case "${TARGET}" in
  cuda-13-0 | cuda-13-0-ubuntu-24-04 | rocm | xpu | cpu | all) ;;
  *)
    echo "Usage: $0 {cuda-13-0|cuda-13-0-ubuntu-24-04|rocm|xpu|cpu|all}"
    exit 2
    ;;
esac

target_enabled() {
  [ "${TARGET}" = "all" ] || [ "${TARGET}" = "$1" ]
}

RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
  echo "ERROR: release-version metadata not set"
  exit 1
fi

COMMIT="$BUILDKITE_COMMIT"

echo "========================================"
echo "Publishing ${TARGET} release images v${RELEASE_VERSION}"
echo "  Commit: ${COMMIT}"
echo "========================================"

# Login to ECR to pull staging images
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# ---- CUDA (default: 13.0) ----

if target_enabled cuda-13-0; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" vllm/vllm-openai:latest-x86_64
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64"
  docker push vllm/vllm-openai:latest-x86_64
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" vllm/vllm-openai:latest
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" "vllm/vllm-openai:v${RELEASE_VERSION}"
  docker push vllm/vllm-openai:latest
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}"

  ZSTD_DIGEST=$(docker buildx imagetools inspect \
    "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64" \
    --format '{{json .Manifest.Digest}}' | tr -d '"')
  .buildkite/scripts/publish-zstd-image.sh \
    "public.ecr.aws/q9t5s3a7/vllm-release-repo@${ZSTD_DIGEST}" \
    "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-zstd"
fi

# ---- Ubuntu 24.04 (CUDA 13.0) ----

if target_enabled cuda-13-0-ubuntu-24-04; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404" vllm/vllm-openai:latest-x86_64-ubuntu2404
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-ubuntu2404"
  docker push vllm/vllm-openai:latest-x86_64-ubuntu2404
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-x86_64-ubuntu2404"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404" vllm/vllm-openai:latest-ubuntu2404
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-ubuntu2404" "vllm/vllm-openai:v${RELEASE_VERSION}-ubuntu2404"
  docker push vllm/vllm-openai:latest-ubuntu2404
  docker push "vllm/vllm-openai:v${RELEASE_VERSION}-ubuntu2404"
fi

# ---- ROCm ----
# Each ROCm stack is identified by its base Dockerfile (see rocm/stack.sh). The
# default stack also owns the plain :latest / :v<ver> tags.

publish_rocm_stack() {
  local base_dockerfile="$1" default="$2" suffix
  (
    # shellcheck source=.buildkite/scripts/rocm/stack.sh
    source .buildkite/scripts/rocm/stack.sh "$base_dockerfile"
    local suffixes=("-${ROCM_STACK_VARIANT}")
    [[ "$default" == "1" ]] && suffixes=("" "-${ROCM_STACK_VARIANT}")
    docker pull "$ROCM_STACK_ECR_IMAGE"
    docker pull "$ROCM_STACK_ECR_BASE"
    for suffix in "${suffixes[@]}"; do
      for tag in "latest${suffix}" "v${RELEASE_VERSION}${suffix}"; do
        docker tag "$ROCM_STACK_ECR_IMAGE" "vllm/vllm-openai-rocm:${tag}"
        docker tag "$ROCM_STACK_ECR_BASE" "vllm/vllm-openai-rocm:${tag}-base"
        docker push "vllm/vllm-openai-rocm:${tag}"
        docker push "vllm/vllm-openai-rocm:${tag}-base"
      done
    done
  )
}

if target_enabled rocm; then
  publish_rocm_stack docker/Dockerfile.rocm_base 1
fi

# ---- XPU ----

if target_enabled xpu; then
  docker pull "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu"

  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu" vllm/vllm-openai-xpu:latest-x86_64
  docker tag "public.ecr.aws/q9t5s3a7/vllm-release-repo:${COMMIT}-x86_64-xpu" "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64"
  docker push vllm/vllm-openai-xpu:latest-x86_64
  docker push "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64"

  docker manifest rm vllm/vllm-openai-xpu:latest || true
  docker manifest rm "vllm/vllm-openai-xpu:v${RELEASE_VERSION}" || true
  docker manifest create vllm/vllm-openai-xpu:latest vllm/vllm-openai-xpu:latest-x86_64 --amend
  docker manifest create "vllm/vllm-openai-xpu:v${RELEASE_VERSION}" "vllm/vllm-openai-xpu:v${RELEASE_VERSION}-x86_64" --amend
  docker manifest push vllm/vllm-openai-xpu:latest
  docker manifest push "vllm/vllm-openai-xpu:v${RELEASE_VERSION}"
fi

# ---- CPU ----
# CPU images are optional and built only when the operator unblocks the job.
if target_enabled cpu; then
  CPU_TAG="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${COMMIT}-x86_64"
  if docker manifest inspect "${CPU_TAG}" >/dev/null 2>&1; then
    docker pull "${CPU_TAG}"
    docker tag "${CPU_TAG}" vllm/vllm-openai-cpu:latest
    docker tag "${CPU_TAG}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}"
    docker push vllm/vllm-openai-cpu:latest
    docker push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}"
    docker tag "${CPU_TAG}" vllm/vllm-openai-cpu:latest-x86_64
    docker tag "${CPU_TAG}" "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64"
    docker push vllm/vllm-openai-cpu:latest-x86_64
    docker push "vllm/vllm-openai-cpu:v${RELEASE_VERSION}-x86_64"
  else
    echo "WARNING: CPU image not found in ECR; skipping CPU publish."
  fi
fi

echo ""
echo "Successfully published ${TARGET} release images for v${RELEASE_VERSION}"
