# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file test accuracy of the vLLM server via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

import lm_eval
import pytest

MODEL_NAMES = [
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
]
FP8_KV_MODEL_NAMES = [
    "Qwen/Qwen3-1.7B",
]
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUES = {
    "Qwen/Qwen3-1.7B": 0.68,
    "google/gemma-3-1b-it": 0.25,
}


def run_test(model_name, more_args=None):
    """Run the end to end accuracy test."""

    model_args = f"pretrained={model_name},max_model_len=4096"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
    )

    measured_value = results["results"][TASK][FILTER]
    assert model_name in EXPECTED_VALUES, (
        f"Cannot find the expected value for the model {model_name=}"
    )
    expected_value = EXPECTED_VALUES[model_name]
    assert (
        measured_value - RTOL < expected_value
        and measured_value + RTOL > expected_value
    ), f"Expected: {expected_value} |  Measured: {measured_value}"


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine(model):
    """Run with the V1 Engine."""

    run_test(model)


@pytest.mark.parametrize("model", FP8_KV_MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(model):
    """Run with the V1 Engine."""

    run_test(model)
