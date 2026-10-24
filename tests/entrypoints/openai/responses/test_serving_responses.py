# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextConfig,
)
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)
from openai_harmony import Message as OpenAIHarmonyMessage
from openai_harmony import Role
from pydantic import ValidationError

import vllm.envs as envs
from tests.entrypoints.openai.chat_completion.test_serving_chat import MockModelConfig
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseCreatedEvent,
    ResponseRawMessageAndToken,
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.inputs import tokens_input
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.harmony import Segment
from vllm.renderers.online_renderer import (
    OnlineRenderer,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import RequestStateStats

pytestmark = pytest.mark.skip_global_cleanup


def _new_online_renderer() -> OnlineRenderer:
    renderer = OnlineRenderer.__new__(OnlineRenderer)
    renderer.exclude_tools_when_tool_choice_none = False
    renderer.parser = None
    renderer.trust_request_chat_template = True
    return renderer


def test_serialize_message_pydantic_model_returns_dict() -> None:
    msg = ResponseRawMessageAndToken(message="hello", tokens=[1, 2, 3])

    serialized = serialize_message(msg)

    assert isinstance(serialized, dict)
    assert serialized["type"] == "raw_message_tokens"
    assert serialized["message"] == "hello"


@pytest.fixture
def mock_exit_stack():
    """Create a mock async exit stack."""
    return MagicMock(spec=AsyncExitStack)


def _echoed_response(**request_fields) -> dict:
    """Build a request with the given fields and return the response as JSON."""
    request = ResponsesRequest(
        model="test-model",
        input="Alice and Bob are going to a science fair on Friday.",
        **request_fields,
    )

    r = ResponsesResponse.from_request(
        request=request,
        sampling_params=request.to_sampling_params(default_max_tokens=16),
        model_name="test-model",
        created_time=0,
        output=[],
        status="completed",
        usage=None,
    )

    return r.model_dump(mode="json")


def test_prompt_cache_key_is_echoed() -> None:
    response = _echoed_response(
        prompt_cache_key="tenant-a",
        safety_identifier="user-1234",
        prompt_cache_retention="in_memory",
    )

    assert response["prompt_cache_key"] == "tenant-a"


def test_safety_identifier_is_echoed() -> None:
    response = _echoed_response(safety_identifier="user-1234")
    assert response["safety_identifier"] == "user-1234"


def test_safety_identifier_defaults_to_none() -> None:
    assert _echoed_response()["safety_identifier"] is None


@pytest.mark.parametrize(
    "sent, echoed",
    [("in_memory", "in_memory"), ("in-memory", "in_memory"), ("24h", "24h")],
)
def test_prompt_cache_retention_is_echoed(sent: str, echoed: str) -> None:
    response = _echoed_response(prompt_cache_retention=sent)
    assert response["prompt_cache_retention"] == echoed


def test_prompt_cache_retention_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest(model="test-model", input="hi", prompt_cache_retention="1w")


def test_prompt_cache_options_echo_applied_defaults() -> None:
    response = _echoed_response(
        prompt_cache_options={"mode": "explicit"},
    )

    assert response["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
        "comparison_response_id": None,
    }


def test_prompt_cache_options_rejects_unknown_ttl() -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest(
            model="test-model",
            input="hi",
            prompt_cache_options={"ttl": "1h"},
        )


def test_prompt_cache_options_request_accepts_prewarm() -> None:
    request = ResponsesRequest(
        model="test-model",
        input="hi",
        prompt_cache_options={"mode": "explicit", "prewarm": True},
    )

    assert request.prompt_cache_options is not None
    assert request.prompt_cache_options.prewarm is True


def test_prompt_cache_options_response_drops_prewarm() -> None:
    response = _echoed_response(
        prompt_cache_options={"mode": "explicit", "prewarm": True},
    )

    assert response["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
        "comparison_response_id": None,
    }


def test_created_event_carries_echoed_fields() -> None:
    event = ResponseCreatedEvent(
        type="response.created",
        sequence_number=0,
        response=_echoed_response(
            prompt_cache_key="tenant-a", safety_identifier="user-1234"
        ),
    )

    assert event.response.prompt_cache_key == "tenant-a"
    assert event.response.safety_identifier == "user-1234"


@pytest.mark.skip_global_cleanup
def test_response_created_event_uses_public_json_schema_alias() -> None:
    schema = {
        "type": "object",
        "properties": {
            "event_name": {"type": "string"},
            "date": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["event_name", "date", "participants"],
        "additionalProperties": False,
    }
    text = ResponseTextConfig()
    text.format = ResponseFormatTextJSONSchemaConfig(
        type="json_schema",
        name="calendar_event",
        schema=schema,
        description="A calendar event.",
        strict=True,
    )
    request = ResponsesRequest(
        model="test-model",
        input="Alice and Bob are going to a science fair on Friday.",
        text=text,
    )
    sampling_params = request.to_sampling_params(default_max_tokens=64)
    initial_response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[],
        status="in_progress",
        usage=None,
    ).model_dump(mode="json", by_alias=True)

    fmt = initial_response["text"]["format"]
    assert fmt["schema"] == schema
    assert "schema_" not in fmt

    event = ResponseCreatedEvent(
        type="response.created",
        sequence_number=0,
        response=initial_response,
    )
    assert event.response.text is not None
    assert event.response.text.format is not None
    assert event.response.text.format.model_dump(by_alias=True)["schema"] == schema


@pytest.mark.asyncio
async def test_online_renderer_renders_non_harmony_responses_with_explicit_history(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _new_online_renderer()
    renderer.use_harmony = False
    renderer.chat_template = "server-template"
    renderer.chat_template_content_format = "string"
    renderer.default_chat_template_kwargs = {"server_default": "kept"}
    renderer.parser = None
    preprocess_chat = AsyncMock(
        return_value=(
            [],
            [tokens_input([11, 12, 13], cache_salt="request-salt")],
        )
    )
    monkeypatch.setattr(renderer, "preprocess_chat", preprocess_chat)
    previous_messages = [
        {"role": "system", "content": "old instructions"},
        {"role": "user", "content": "first"},
    ]
    previous_outputs = [
        ResponseOutputMessage(
            id="msg_1",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="prior answer",
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
    ]
    request = ResponsesRequest(
        input="next",
        instructions="new instructions",
        tools=[],
        cache_salt="request-salt",
        chat_template_kwargs={"request_default": "kept"},
    )

    result = await renderer.render_responses(
        request,
        previous_messages=previous_messages,
        previous_response_outputs=previous_outputs,
    )

    assert not isinstance(result, ErrorResponse)
    assert result.messages == [
        {"role": "system", "content": "new instructions"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "next"},
    ]
    assert result.engine_input["type"] == "token"
    assert result.engine_input["prompt_token_ids"] == [11, 12, 13]
    assert result.engine_input["cache_salt"] == "request-salt"
    assert previous_messages[0]["content"] == "old instructions"
    assert previous_outputs[0].content[0].text == "prior answer"

    preprocess_kwargs = preprocess_chat.call_args.kwargs
    assert preprocess_kwargs["default_template"] == "server-template"
    assert preprocess_kwargs["default_template_content_format"] == "string"
    assert preprocess_kwargs["default_template_kwargs"]["server_default"] == "kept"
    assert preprocess_kwargs["default_template_kwargs"]["request_default"] == "kept"


_REUSED_IDS = [10, 20, 30]
_IMAGE_PART = {
    "type": "input_image",
    "image_url": "https://example.com/a.png",
    "detail": "auto",
}


async def _render_responses_with_reuse(
    request, monkeypatch: pytest.MonkeyPatch, previous_messages=None
):
    online_renderer = OnlineRenderer(
        model_config=MockModelConfig(),
        renderer=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    render_chat = AsyncMock(return_value=([[]], [tokens_input([1, 2])]))
    monkeypatch.setattr(online_renderer.renderer, "render_chat_async", render_chat)

    result = await online_renderer.render_responses(
        request, previous_messages=previous_messages
    )

    assert not isinstance(result, ErrorResponse)
    return render_chat, result.engine_input


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [_REUSED_IDS, [1.5]])
@pytest.mark.parametrize(
    "request_input",
    [
        [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "look"}, _IMAGE_PART],
            }
        ],
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shot",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [_IMAGE_PART],
            },
        ],
    ],
    ids=["input_image", "tool_output_image"],
)
async def test_responses_kv_transfer_prompt_token_ids_ignored_with_non_text_input(
    request_input, ids, monkeypatch: pytest.MonkeyPatch
):
    request = ResponsesRequest(
        input=request_input,
        kv_transfer_params={"do_remote_prefill": True, "prompt_token_ids": ids},
    )

    render_chat, _ = await _render_responses_with_reuse(request, monkeypatch)

    render_chat.assert_awaited_once()
    assert request.kv_transfer_params == {"do_remote_prefill": True}


@pytest.mark.asyncio
async def test_responses_kv_transfer_prompt_token_ids_ignored_with_media_in_history(
    monkeypatch: pytest.MonkeyPatch,
):
    request = ResponsesRequest(
        input="next", kv_transfer_params={"prompt_token_ids": _REUSED_IDS}
    )

    render_chat, _ = await _render_responses_with_reuse(
        request,
        monkeypatch,
        previous_messages=[{"role": "user", "content": [_IMAGE_PART]}],
    )

    render_chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_kv_transfer_prompt_token_ids_kept_with_text_input(
    monkeypatch: pytest.MonkeyPatch,
):
    request = ResponsesRequest(
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"role": "assistant", "content": "ok"},
        ],
        kv_transfer_params={"prompt_token_ids": _REUSED_IDS},
    )

    _, engine_input = await _render_responses_with_reuse(request, monkeypatch)

    assert engine_input["prompt_token_ids"] == _REUSED_IDS


@pytest.mark.asyncio
async def test_online_renderer_rejects_untrusted_responses_chat_template(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _new_online_renderer()
    renderer.use_harmony = False
    renderer.chat_template = "server-template"
    renderer.chat_template_content_format = "string"
    renderer.default_chat_template_kwargs = {}
    renderer.exclude_tools_when_tool_choice_none = False
    renderer.trust_request_chat_template = False
    renderer.parser = None
    preprocess_chat = AsyncMock(
        return_value=([], [tokens_input([1])]),
    )
    monkeypatch.setattr(renderer, "preprocess_chat", preprocess_chat)
    request = ResponsesRequest(
        input="hello",
        chat_template_kwargs={"chat_template": "{{ messages }}"},
    )

    result = await renderer.render_responses(request)

    assert isinstance(result, ErrorResponse)
    assert result.error.code == 400
    assert "untrusted chat template" in result.error.message.lower()
    preprocess_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_online_renderer_rejects_harmony_history_for_non_harmony_model():
    renderer = _new_online_renderer()
    renderer.use_harmony = False
    request = ResponsesRequest(input="next")

    result = await renderer.render_responses(
        request,
        previous_messages=[
            OpenAIHarmonyMessage.from_role_and_content(Role.USER, "first")
        ],
    )

    assert isinstance(result, ErrorResponse)
    assert result.error.param == "previous_response_id"


@pytest.mark.asyncio
async def test_online_renderer_rejects_chat_history_for_harmony_model():
    renderer = _new_online_renderer()
    renderer.use_harmony = True
    request = ResponsesRequest(input="next")

    result = await renderer.render_responses(
        request,
        previous_messages=[{"role": "user", "content": "first"}],
    )

    assert isinstance(result, ErrorResponse)
    assert result.error.param == "previous_response_id"


@pytest.mark.asyncio
async def test_online_renderer_adjusts_harmony_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
):
    class StubParser:
        def __init__(self, tokenizer, tools, *, model_config):
            pass

        def adjust_request(self, request):
            request.cache_salt = "adjusted"
            return request

    renderer = _new_online_renderer()
    renderer.use_harmony = True
    renderer.parser = StubParser
    renderer.model_config = SimpleNamespace(max_model_len=100)
    tokenizer = SimpleNamespace(truncation_side="left")
    renderer.renderer = SimpleNamespace(
        tokenizer=tokenizer,
        get_tokenizer=lambda: tokenizer,
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.render_for_completion",
        lambda messages: [1],
    )
    request = ResponsesRequest(
        input="next",
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        tool_choice={"type": "function", "name": "lookup"},
    )

    result = await renderer.render_responses(request)

    assert not isinstance(result, ErrorResponse)
    assert result.engine_input["type"] == "token"
    assert result.engine_input["cache_salt"] == "adjusted"


@pytest.mark.asyncio
async def test_online_renderer_applies_responses_token_budget_to_harmony_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _new_online_renderer()
    renderer.use_harmony = True
    renderer.model_config = SimpleNamespace(max_model_len=5)
    renderer.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(truncation_side="left")
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.render_for_completion",
        lambda messages: [1, 2, 3, 4, 5, 6],
    )
    request = ResponsesRequest(
        input="hello",
        max_output_tokens=2,
        truncation="auto",
    )

    result = await renderer.render_responses(request)

    assert not isinstance(result, ErrorResponse)
    assert result.engine_input["type"] == "token"
    # `max_tokens` is an output upper bound rather than a reservation off the
    # context window, so the prompt budget is the full max_model_len minus the
    # one token held back for output: 6 tokens truncate to 4, not to
    # max_model_len - max_output_tokens.
    assert result.engine_input["prompt_token_ids"] == [3, 4, 5, 6]


@pytest.mark.asyncio
async def test_online_renderer_renders_harmony_continuation_with_call_linkage(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _new_online_renderer()
    renderer.use_harmony = True
    renderer.model_config = SimpleNamespace(max_model_len=100)
    renderer.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(truncation_side="left")
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.render_for_completion",
        lambda messages: [1],
    )
    previous_messages = [
        OpenAIHarmonyMessage.from_role_and_content(Role.USER, "first"),
    ]
    previous_outputs = [
        ResponseFunctionToolCall(
            arguments='{"city":"Paris"}',
            call_id="call_1",
            name="lookup_weather",
            type="function_call",
        )
    ]
    request = ResponsesRequest(
        input=[
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            }
        ],
        instructions="replacement instructions are ignored on continuations",
        reasoning={"effort": "high"},
        tools=[
            {
                "type": "function",
                "name": "lookup_weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        cache_salt="request-salt",
    )

    result = await renderer.render_responses(
        request,
        previous_messages=previous_messages,
        previous_response_outputs=previous_outputs,
    )

    assert not isinstance(result, ErrorResponse)
    assert result.messages[0] is previous_messages[0]
    tool_output = result.messages[-1]
    assert isinstance(tool_output, OpenAIHarmonyMessage)
    assert tool_output.author.role == "tool"
    assert tool_output.author.name == "functions.lookup_weather"
    assert tool_output.channel == "commentary"
    assert tool_output.recipient == "assistant"
    assert result.engine_input["type"] == "token"
    assert result.engine_input["cache_salt"] == "request-salt"
    assert "arrival_time" in result.engine_input
    assert previous_messages == [
        OpenAIHarmonyMessage.from_role_and_content(Role.USER, "first"),
    ]
    assert previous_outputs[0].call_id == "call_1"


@pytest.mark.asyncio
async def test_online_renderer_returns_typed_error_for_bad_harmony_call_reference():
    renderer = _new_online_renderer()
    renderer.use_harmony = True
    request = ResponsesRequest(
        input=[
            {
                "type": "function_call_output",
                "call_id": "missing",
                "output": "orphaned",
            }
        ],
        tools=[],
    )

    result = await renderer.render_responses(request)

    assert isinstance(result, ErrorResponse)
    assert result.error.type == "invalid_request_error"
    assert result.error.param == "input"
    assert "No call message found for missing" in result.error.message


@pytest.mark.asyncio
async def test_online_renderer_preserves_missing_harmony_builtin_tool_behavior(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _new_online_renderer()
    renderer.use_harmony = True
    renderer.model_config = SimpleNamespace(max_model_len=100)
    renderer.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(truncation_side="left")
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.render_for_completion",
        lambda messages: [1],
    )
    request = ResponsesRequest(
        input="search",
        tools=[{"type": "web_search_preview"}],
    )

    result = await renderer.render_responses(request)

    assert not isinstance(result, ErrorResponse)


@pytest.mark.parametrize(
    "engine_inputs",
    [[], [tokens_input([1]), tokens_input([2])]],
)
def test_responses_render_result_rejects_non_single_prompt(engine_inputs):
    renderer = _new_online_renderer()

    result = renderer._responses_render_result([], engine_inputs)

    assert isinstance(result, ErrorResponse)
    assert f"got {len(engine_inputs)}" in result.error.message


class _Qwen3FakeTokenizer:
    def __init__(self):
        self._vocab = {"<think>": 1, "</think>": 2, "reason": 3, "final": 4}

    def get_vocab(self):
        return self._vocab

    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self._vocab.items()}
        return "".join(id_to_token.get(token_id, "x") for token_id in token_ids)


def _make_text_request_output(text: str, token_ids: list[int]) -> RequestOutput:
    return RequestOutput(
        request_id="req",
        prompt="hi",
        prompt_token_ids=[7, 8],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        finished=True,
        num_cached_tokens=0,
        num_cache_creation_tokens=3,
    )


async def _run_full_generator(serving, request, context, tokenizer):
    async def dummy_result_generator():
        yield None

    return await serving.responses_full_generator(
        request=request,
        sampling_params=SamplingParams(max_tokens=16),
        result_generator=dummy_result_generator(),
        context=context,
        model_name="test-model",
        tokenizer=tokenizer,
        request_metadata=RequestResponseMetadata(request_id="req"),
    )


class TestHarmonyPreambleStreaming:
    """Tests for preamble (commentary with no recipient) streaming events."""

    @staticmethod
    def _make_segment(*, channel, recipient, delta="hello"):
        """Build a lightweight segment for Harmony streaming tests."""
        return Segment(channel=channel, recipient=recipient, delta=delta)

    @staticmethod
    def _make_previous_item(*, channel, recipient, text="preamble text"):
        """Build a lightweight mock previous_item (openai_harmony Message)."""
        content_part = MagicMock()
        content_part.text = text
        item = MagicMock()
        item.channel = channel
        item.recipient = recipient
        item.content = [content_part]
        return item


_PER_REQUEST_STATS = RequestStateStats(
    queued_ts=1.0,
    scheduled_ts=1.5,
    first_token_ts=2.0,
    last_token_ts=3.0,
)


def _make_request_output(
    text,
    token_ids,
    metrics: RequestStateStats | None = None,
    finish_reason: str | None = None,
):
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=token_ids,
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason=finish_reason,
        stop_reason=None,
    )
    return RequestOutput(
        request_id="req",
        prompt="hi",
        prompt_token_ids=[7, 8],
        prompt_logprobs=None,
        outputs=[completion],
        finished=False,
        num_cached_tokens=0,
        metrics=metrics,
    )


async def _empty_context_generator():
    if False:
        yield


def _identity_increment(event):
    """Simple identity callable for _increment_sequence_number_and_return."""
    seq = getattr(_identity_increment, "_counter", 0)
    if hasattr(event, "sequence_number"):
        event.sequence_number = seq
    _identity_increment._counter = seq + 1  # type: ignore
    return event


def _mock_parser_with_reasoning(serving, delta_sequence: list[DeltaMessage]):
    """Set up serving.parser so that it returns a mock parser instance
    with a reasoning parser that returns the given delta_sequence.

    The mock has reasoning_parser set (truthy) but tool_parser as None,
    so the parser's parse_delta enters the reasoning-only branch.
    """
    call_count = 0

    def mock_parse_delta(**kwargs):
        nonlocal call_count
        if call_count >= len(delta_sequence):
            return None
        result = delta_sequence[call_count]
        call_count += 1
        return result

    mock_parser_instance = MagicMock()
    mock_parser_instance.reasoning_parser = MagicMock()  # truthy
    mock_parser_instance.tool_parser = None
    mock_parser_instance.parse_delta = mock_parse_delta
    mock_parser_instance.is_reasoning_end = MagicMock(return_value=False)
    serving.parser = MagicMock(return_value=mock_parser_instance)
    return mock_parser_instance


class TestAutoToolStreaming:
    @pytest.mark.skip_global_cleanup
    @pytest.mark.asyncio
    async def test_auto_multi_tool_streaming_opens_one_item_per_tool(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)

        delta_sequence = [
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id="call_vienna",
                        type="function",
                        index=0,
                        function=DeltaFunctionCall(
                            name="get_weather",
                            arguments="",
                        ),
                    )
                ]
            ),
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(
                            arguments='{"location":"Vienna"}',
                        ),
                    )
                ]
            ),
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id="call_berlin",
                        type="function",
                        index=1,
                        function=DeltaFunctionCall(
                            name="get_weather",
                            arguments='{"location":"Berlin"}',
                        ),
                    )
                ]
            ),
        ]
        events = await self._collect_events(delta_sequence)

        function_items = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert len(function_items) == 2
        assert [event.item.name for event in function_items] == [
            "get_weather",
            "get_weather",
        ]
        assert [event.output_index for event in function_items] == [0, 1]

        argument_deltas = [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ]
        assert argument_deltas == [
            '{"location":"Vienna"}',
            '{"location":"Berlin"}',
        ]

        argument_done = [
            event
            for event in events
            if event.type == "response.function_call_arguments.done"
        ]
        assert [event.arguments for event in argument_done] == [
            '{"location":"Vienna"}',
            '{"location":"Berlin"}',
        ]
        assert [event.output_index for event in argument_done] == [0, 1]

        function_done = [
            event
            for event in events
            if event.type == "response.output_item.done"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert [event.item.arguments for event in function_done] == [
            '{"location":"Vienna"}',
            '{"location":"Berlin"}',
        ]
        assert [event.output_index for event in function_done] == [0, 1]

    @pytest.mark.skip_global_cleanup
    @pytest.mark.asyncio
    async def test_auto_tool_choice_first_delta_tool_call_does_not_duplicate_item(
        self, monkeypatch
    ):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)

        delta_sequence = [
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id="call_test",
                        type="function",
                        index=0,
                        function=DeltaFunctionCall(
                            name="get_weather",
                            arguments="",
                        ),
                    )
                ]
            ),
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(
                            arguments='{"location":"Berlin"}',
                        ),
                    )
                ]
            ),
        ]
        events = await self._collect_events(delta_sequence)

        function_items = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert len(function_items) == 1
        assert function_items[0].item.name == "get_weather"

        argument_deltas = [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ]
        assert "".join(argument_deltas) == '{"location":"Berlin"}'

    @pytest.mark.skip_global_cleanup
    @pytest.mark.asyncio
    async def test_compound_content_and_tool_name_args_same_delta(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)

        tool_args = '{"location":"Berlin"}'

        delta_sequence = [
            DeltaMessage(
                content="Let me check.",
                tool_calls=[
                    DeltaToolCall(
                        id="call_weather",
                        type="function",
                        index=0,
                        function=DeltaFunctionCall(name="get_weather"),
                    ),
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(arguments=tool_args),
                    ),
                ],
            )
        ]

        events = await self._collect_events(delta_sequence)

        text_deltas = [
            event.delta
            for event in events
            if event.type == "response.output_text.delta"
        ]
        assert text_deltas == ["Let me check."]

        argument_deltas = [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ]
        assert argument_deltas == [tool_args]

        types = [event.type for event in events]
        assert types.index("response.output_text.delta") < types.index(
            "response.function_call_arguments.delta"
        )

        argument_done = [
            event
            for event in events
            if event.type == "response.function_call_arguments.done"
        ]
        assert [event.arguments for event in argument_done] == [tool_args]

        function_done = [
            event
            for event in events
            if event.type == "response.output_item.done"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert len(function_done) == 1
        assert function_done[0].item.name == "get_weather"
        assert function_done[0].item.arguments == tool_args


def _harmony_msg(channel: str, text: str, recipient: str | None = None):
    msg = OpenAIHarmonyMessage.from_role_and_content(Role.ASSISTANT, text)
    msg = msg.with_channel(channel)
    return msg.with_recipient(recipient) if recipient else msg


def _harmony_segments(msg, *, streamed: bool = True) -> list[Segment]:
    """Segments the parser yields for one message: a content delta (absent
    for a zero-delta message), then the completed message."""
    delta = [Segment(msg.channel, msg.recipient, msg.content[0].text)]
    completed = Segment(msg.channel, msg.recipient, "", completed_message=msg)
    return (delta if streamed else []) + [completed]
