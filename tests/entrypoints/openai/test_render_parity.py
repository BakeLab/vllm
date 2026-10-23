# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-API HF render-input parity tests.

Drives the real Chat Completions and Responses prep paths and captures the
arguments each passes to ``BaseRenderer.render_chat_async`` (via shared
``OnlineRenderer.preprocess_chat``):

  - conversation messages
  - ChatParams fields that affect the HF template / media path
    (template, content format, template kwargs including tools,
     media_io_kwargs, mm_processor_kwargs)

``TokenizeParams``, ``prompt_extras``, Harmony / GPT-OSS, prefill / continue,
prompt cache salt, and truncation are out of scope for this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.inputs import tokens_input
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import ChatParams

_MODEL = "test-model"
_USER = [{"role": "user", "content": "Hello"}]

_WEATHER_PARAMETERS = {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"],
}

_CHAT_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": _WEATHER_PARAMETERS,
    },
}

_RESPONSES_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": _WEATHER_PARAMETERS,
}


@dataclass
class MockHFConfig:
    model_type: str = "llama"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = _MODEL
    tokenizer = _MODEL
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
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass(frozen=True)
class CapturedRenderInputs:
    """Args observed at ``render_chat_async`` (HF render boundary)."""

    messages: list[Any]
    chat_params: ChatParams


class RenderCapture:
    """Install a ``render_chat_async`` stub and record its HF-bound inputs."""

    def __init__(
        self, online_renderer: OnlineRenderer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.online_renderer = online_renderer
        self.captured: CapturedRenderInputs | None = None

        async def fake_render_chat_async(
            conversations,
            chat_params,
            tok_params=None,
            *,
            prompt_extras=None,
            skip_mm_cache=False,
        ):
            assert len(conversations) == 1
            self.captured = CapturedRenderInputs(
                messages=list(conversations[0]),
                chat_params=chat_params,
            )
            return [list(conversations[0])], [
                tokens_input(prompt_token_ids=[0]),
            ]

        monkeypatch.setattr(
            online_renderer.renderer, "render_chat_async", fake_render_chat_async
        )

    def take(self) -> CapturedRenderInputs:
        assert self.captured is not None
        captured = self.captured
        self.captured = None
        return captured


async def _capture_chat(
    online_renderer: OnlineRenderer,
    request: ChatCompletionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> CapturedRenderInputs:
    capture = RenderCapture(online_renderer, monkeypatch)
    result = await online_renderer.render_chat(request)
    assert not isinstance(result, ErrorResponse), result
    return capture.take()


def _weather_tools(
    *, overrides: dict[str, Any] | None = None
) -> tuple[list[dict], list[dict]]:
    """Return (chat_tools, responses_tools) for the shared weather function."""
    overrides = overrides or {}
    chat_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": _WEATHER_PARAMETERS,
            **overrides,
        },
    }
    responses_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": _WEATHER_PARAMETERS,
        **overrides,
    }
    return [chat_tool], [responses_tool]


@pytest.fixture
def model_config() -> MockModelConfig:
    return MockModelConfig()


@pytest.fixture
def online_renderer(model_config: MockModelConfig, request) -> OnlineRenderer:
    exclude_tools_when_tool_choice_none = getattr(request, "param", False)
    renderer = MagicMock()
    # Non-Mistral stub; only needed so render_chat / preprocess_chat can
    # inspect tokenizer type before render_chat_async is mocked.
    renderer.tokenizer = MagicMock()

    return OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser="openai",
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        trust_request_mm_kwargs=True,
    )


@pytest.mark.asyncio
class TestReasoningRenderParity:
    async def test_reasoning_effort_none_rejected_by_harmony(self, online_renderer):
        """Harmony rejects reasoning_effort="none" as invalid user input."""
        request = ChatCompletionRequest(
            model=_MODEL,
            messages=_USER,
            reasoning_effort="none",
        )
        with pytest.raises(
            VLLMValidationError, match="Harmony does not support"
        ) as exc_info:
            online_renderer._make_request_with_harmony(request)
        assert exc_info.value.parameter == "reasoning_effort"
