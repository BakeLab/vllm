# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    placeholder_ranges_from_engine_input,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
)
from vllm.multimodal.inputs import (
    PlaceholderRange,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers import renderer_from_config
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.stats import RequestSpecDecodeMetrics

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockSchedulerConfig:
    max_num_seqs: int = 128


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig
    scheduler_config: MockSchedulerConfig = field(default_factory=MockSchedulerConfig)


def _build_renderer(model_config: MockModelConfig):
    return renderer_from_config(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
    )


def _make_request_output(
    request_id: str,
    token_ids: list[int],
    finish_reason: str | None = None,
    stop_reason: int | str | None = None,
    finished: bool = False,
    prompt_token_ids: list[int] | None = None,
    logprobs: list[dict[int, Any] | None] | None = None,
    num_cached_tokens: int | None = None,
    index: int = 0,
    spec_decode_metrics: RequestSpecDecodeMetrics | None = None,
    text: str = "",
) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=logprobs,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                spec_decode_metrics=spec_decode_metrics,
            )
        ],
        finished=finished,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
        num_cached_tokens=num_cached_tokens,
    )


def _mock_engine() -> MagicMock:
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = MockModelConfig()
    engine.vllm_config = MockVllmConfig(
        engine.model_config, parallel_config=MockParallelConfig()
    )
    engine.input_processor = MagicMock()
    engine.renderer = _build_renderer(engine.model_config)
    return engine


def _parse_sse_chunks(chunks: list[str]) -> list[Any]:
    """Parse SSE chunks into dicts (JSON) or raw strings ([DONE])."""
    parsed: list[Any] = []
    for chunk in chunks:
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        payload = chunk[len("data: ") : -len("\n\n")]
        if payload == "[DONE]":
            parsed.append("[DONE]")
        else:
            parsed.append(json.loads(payload))
    return parsed


def test_placeholder_ranges_from_engine_input():
    engine_input: Any = {
        "type": "multimodal",
        "mm_placeholders": {
            "image": [
                PlaceholderRange(offset=1, length=5),
                PlaceholderRange(offset=8, length=2),
            ]
        },
    }

    placeholders = placeholder_ranges_from_engine_input(engine_input)

    assert placeholders is not None
    assert [item.model_dump() for item in placeholders["image"]] == [
        {"offset": 1, "length": 5},
        {"offset": 8, "length": 2},
    ]

    text_only: Any = {"type": "token", "prompt_token_ids": [1, 2, 3]}
    assert placeholder_ranges_from_engine_input(text_only) is None


MM_ENGINE_INPUT: Any = {
    "type": "multimodal",
    "prompt_token_ids": [1, 2, 2, 2, 3],
    "mm_placeholders": {"image": [PlaceholderRange(offset=1, length=3)]},
}
MM_PLACEHOLDERS = {"image": [{"offset": 1, "length": 3}]}


async def _collect_sse(response) -> list[Any]:
    return _parse_sse_chunks([chunk async for chunk in response])


def _text_request(**kwargs) -> GenerateRequest:
    kwargs.setdefault("sampling_params", SamplingParams(max_tokens=10))
    return GenerateRequest(
        token_ids=[1, 2, 3], model=MODEL_NAME, output_mode="text", **kwargs
    )
