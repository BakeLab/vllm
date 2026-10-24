# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens

logger = init_logger(__name__)


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
