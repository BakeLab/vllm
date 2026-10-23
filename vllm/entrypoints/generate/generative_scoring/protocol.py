# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any, Literal

from pydantic import Field, model_validator

from vllm.entrypoints.generate.base.protocol import validate_cache_salt
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.utils import random_uuid


class GenerativeScoringRequest(OpenAIBaseModel):
    """Request for computing generative scoring.

    Attributes:
        model: The model to use for scoring. Optional, follows existing patterns.
        query: The query text or pre-tokenized query token IDs.
        items: The item text(s) or pre-tokenized item token IDs.
        label_token_ids: List of token IDs to compute probabilities for.
        apply_softmax: Whether to normalize probabilities using softmax over only
            the label_token_ids (True) or return true model probabilities over
            the full vocab for those ids (False).
        item_first: If True, prepend items to query. Otherwise append items to query.
        add_special_tokens: Whether to add special tokens when tokenizing.
        cache_salt: Optional salt for prefix caching.

    """

    model: str | None = None
    query: str | list[int] = Field(
        ...,
        description="The query text or pre-tokenized query token IDs.",
    )
    items: list[str] | list[list[int]] = Field(
        ...,
        description="List of item texts or pre-tokenized item token IDs.",
    )
    label_token_ids: list[int] = Field(
        ...,
        description="List of token IDs to compute probabilities for.",
    )
    apply_softmax: bool = Field(
        default=True,
        description=(
            "If True, normalize probabilities using softmax over only the "
            "label_token_ids. If False, return the true model probabilities "
            "over the full vocab for those ids."
        ),
    )
    item_first: bool = Field(
        default=False,
        description="If True, prepend items to query. Otherwise append items to query.",
    )
    add_special_tokens: bool = Field(
        default=True,
        description="Whether to add special tokens when tokenizing.",
    )
    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    priority: int = Field(
        default=0,
        ge=-(2**63),
        le=2**63 - 1,
        description=(
            "The priority of the request (lower means earlier handling; default: 0)."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description="The request_id related to this request.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data: Any) -> Any:
        if isinstance(data, dict):
            validate_cache_salt(data.get("cache_salt"))
        return data


class GenerativeScoringItemResult(OpenAIBaseModel):
    """Result for a single item in the generative scoring response.

    Attributes:
        index: The index of this item in the input items list.
        object: Type of object, always "score".
        score: The probability score for the first label token.

    """

    index: int
    object: Literal["score"] = "score"
    score: float


class GenerativeScoringResponse(OpenAIBaseModel):
    """Response from the generative scoring computation.

    Attributes:
        id: Unique identifier for this response.
        object: Type of object, always "list".
        created: Unix timestamp of when the response was created.
        model: The model used for scoring.
        data: List of scoring results, one per input item.
        usage: Token usage information.

    """

    id: str = Field(default="")
    object: Literal["list"] = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[GenerativeScoringItemResult]
    usage: UsageInfo
