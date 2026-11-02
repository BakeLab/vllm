#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Publish the x86_64 CPU nightly image from ECR to Docker Hub.

set -ex

BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the CPU image in ECR}"
DRY_RUN="${DRY_RUN:-0}"
ORIG_TAG="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${BUILDKITE_COMMIT}-x86_64"

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$ORIG_TAG"
docker tag "$ORIG_TAG" vllm/vllm-openai-cpu:nightly-x86_64
docker tag "$ORIG_TAG" "vllm/vllm-openai-cpu:nightly-${BUILDKITE_COMMIT}"
docker tag "$ORIG_TAG" vllm/vllm-openai-cpu:nightly

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Created CPU nightly tags locally; skipping Docker Hub push."
  exit 0
fi

docker push vllm/vllm-openai-cpu:nightly-x86_64
docker push "vllm/vllm-openai-cpu:nightly-${BUILDKITE_COMMIT}"
docker push vllm/vllm-openai-cpu:nightly
