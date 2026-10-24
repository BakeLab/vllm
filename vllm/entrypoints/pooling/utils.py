# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.renderers import TokenizeParams
from vllm.tasks import SupportedTask
from vllm.utils.serial_utils import (
    EMBED_DTYPES,
    EmbedDType,
    Endianness,
    binary2tensor,
)

logger = init_logger(__name__)

JsonEncodingFormat = Literal["float", "base64"]
BytesEncodingFormat = Literal["bytes", "bytes_only"]


@dataclass
class MetadataItem:
    index: int
    embed_dtype: EmbedDType
    endianness: Endianness
    start: int
    end: int
    shape: tuple[int, ...]


def build_metadata_items(
    embed_dtype: EmbedDType,
    endianness: Endianness,
    shape: tuple[int, ...],
    n_request: int,
) -> list[MetadataItem]:
    n_bytes = EMBED_DTYPES[embed_dtype].nbytes
    size = math.prod(shape)

    return [
        MetadataItem(
            index=i,
            embed_dtype=embed_dtype,
            endianness=endianness,
            start=i * size * n_bytes,
            end=(i + 1) * size * n_bytes,
            shape=shape,
        )
        for i in range(n_request)
    ]


def decode_pooling_output(items: list[MetadataItem], body: bytes) -> list[torch.Tensor]:
    return [
        binary2tensor(
            body[item.start : item.end],
            item.shape,
            item.embed_dtype,
            item.endianness,
        )
        for item in sorted(items, key=lambda x: x.index)
    ]


def enable_scoring_api(
    supported_tasks: tuple[SupportedTask, ...],
    model_config: ModelConfig | None = None,
) -> bool:
    if model_config is None:
        return False

    pooling_task = model_config.get_pooling_task(supported_tasks)
    if pooling_task in ("embed", "token_embed"):
        return True

    if pooling_task == "classify":
        num_labels = getattr(model_config.hf_config, "num_labels", 0)
        if num_labels != 1:
            logger.debug_once("Scoring API is only enabled for num_labels == 1.")
            return False
        return True

    return False


def reserve_tokens(tok_params: TokenizeParams, num_tokens: int) -> TokenizeParams:
    """Shrink every length limit by `num_tokens` that are appended afterwards."""

    def shrink(length: int | None) -> int | None:
        return length if length is None or length < 0 else max(length - num_tokens, 0)

    return replace(
        tok_params,
        max_total_tokens=shrink(tok_params.max_total_tokens),
        truncate_prompt_tokens=shrink(tok_params.truncate_prompt_tokens),
        pad_prompt_tokens=shrink(tok_params.pad_prompt_tokens),
    )
