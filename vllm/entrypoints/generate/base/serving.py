# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import RequestOutput
from vllm.entrypoints.generate.base.protocol import (
    PerRequestMetrics,
    SpeculativeDecodingMetrics,
)
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens
from vllm.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)

RequestT = TypeVar("RequestT", bound=AnyRequest)
_T = TypeVar("_T")
SESSION_ID_HEADER = "X-Session-ID"
PRIORITY_HEADER = "X-Vllm-Priority"


def build_per_request_timing_metrics(
    metrics: RequestStateStats | None,
    num_generation_tokens: int,
) -> PerRequestMetrics:
    """Build per-request timing metrics from ``RequestStateStats``.

    ``generation_time_ms`` is the decode interval only (first output token to
    last output token); it excludes both queue wait and prefill/TTFT.
    ``tokens_per_second`` is overall output throughput: all generated tokens
    over the inference interval (scheduling to last output token), so it counts
    the prefill/TTFT phase and is not simply the reciprocal of ``mean_itl_ms``.
    Each field is left ``None`` when the timestamps it depends on are
    unavailable.
    """
    if metrics is None:
        return PerRequestMetrics()

    queued_ts = metrics.queued_ts
    scheduled_ts = metrics.scheduled_ts
    first_token_ts = metrics.first_token_ts
    last_token_ts = metrics.last_token_ts

    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    queue_time_ms: float | None = None
    mean_itl_ms: float | None = None
    tokens_per_second: float | None = None

    if scheduled_ts > 0 and first_token_ts > 0:
        time_to_first_token_ms = (first_token_ts - scheduled_ts) * 1000

    if first_token_ts > 0 and last_token_ts > 0:
        generation_time_ms = (last_token_ts - first_token_ts) * 1000

    if queued_ts > 0 and scheduled_ts > 0:
        queue_time_ms = (scheduled_ts - queued_ts) * 1000

    if first_token_ts > 0 and last_token_ts > 0 and num_generation_tokens > 1:
        decode_time = last_token_ts - first_token_ts
        mean_itl_ms = decode_time / (num_generation_tokens - 1) * 1000

    if scheduled_ts > 0 and last_token_ts > 0:
        inference_time_ms = (last_token_ts - scheduled_ts) * 1000
        if inference_time_ms > 0:
            tokens_per_second = num_generation_tokens / inference_time_ms * 1000

    return PerRequestMetrics(
        time_to_first_token_ms=time_to_first_token_ms,
        generation_time_ms=generation_time_ms,
        queue_time_ms=queue_time_ms,
        mean_itl_ms=mean_itl_ms,
        tokens_per_second=tokens_per_second,
    )


def build_spec_decoding_metrics(
    final_res: RequestOutput | None,
) -> SpeculativeDecodingMetrics | None:
    """Build per-request spec-decode acceptance metrics from the single output
    sequence, or ``None`` when unavailable (metrics disabled, or no sequence
    yet).

    Only meaningful for single-sequence requests; callers suppress it for n>1.
    """
    if final_res is None or not final_res.outputs:
        return None
    metrics = final_res.outputs[0].spec_decode_metrics
    if metrics is None:
        return None
    return SpeculativeDecodingMetrics(**metrics.to_dict())


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: list[EngineInput] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


def format_token_id_placeholder(token_id: int) -> str:
    return f"token_id:{token_id}"


def decode_token_ids(
    token_ids: list[int], tokenizer: TokenizerLike
) -> list[tuple[str, list[int] | None]]:
    """Decode token ids individually to their token strings and UTF-8 bytes.

    Uses the engine's per-token detokenization, which restores the
    SentencePiece leading space that `convert_tokens_to_string` drops, so the
    strings match the coupled endpoints. Ids are decoded in one batch (callers
    pass a position's sampled id together with its top-k ids). An id with no
    vocab entry decodes to ("", None).
    """
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    known = [tid for tid, piece in zip(token_ids, pieces) if piece is not None]
    decoded = iter(convert_ids_list_to_tokens(tokenizer, known))
    out: list[tuple[str, list[int] | None]] = []
    for tid, piece in zip(token_ids, pieces):
        if piece is None:
            logger.warning_once(
                "decode_token_ids: token_id %d has no vocab entry; "
                "substituting empty string",
                tid,
            )
            out.append(("", None))
            continue
        token_str = next(decoded)
        out.append((token_str, list(token_str.encode("utf-8", errors="replace"))))
    return out


def chat_top_logprobs_limit(
    top_logprobs: int | None, return_all: bool = False
) -> int | None:
    """How many of a position's logprob candidates `/v1/chat/completions`
    returns, in the engine's order (sampled token first): the first
    ``top_logprobs``, none when it is unset, and all of them (``None``) for
    ``top_logprobs=-1`` or ``logprob_token_ids`` (``return_all``). Shared with
    chat derender so the two cannot drift."""
    if return_all or top_logprobs == -1:
        return None
    return top_logprobs or 0


def completion_top_logprobs_limit(
    logprobs: int | None, return_all: bool = False
) -> int | None:
    """How many of a position's logprob candidates `/v1/completions` returns:
    the first ``logprobs + 1`` (the OpenAI completions convention), and all of
    them (``None``) for ``logprobs=-1``, ``logprob_token_ids`` (``return_all``)
    or when ``logprobs`` is unset. Shared with completions derender so the two
    cannot drift."""
    if return_all or logprobs is None or logprobs == -1:
        return None
    return max(logprobs + 1, 0)


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
