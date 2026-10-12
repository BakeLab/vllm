# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in safetensors loading through cuFile's CPU-buffered compatibility mode."""

import json
import math
import os
import time
from collections.abc import Generator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def cufile_compat_weights_iterator(
    files: list[str], local_expert_ids: set[int] | None = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Read tensor payloads into the current CUDA device, one tensor at a time.

    Each TP rank reads the complete checkpoint. Expert filtering is unsupported.
    Compatibility mode still stages data through CPU memory; this is not GDS DMA.
    """
    if local_expert_ids is not None:
        raise ValueError("cuFile compatibility loading requires TP without EP")
    if os.environ.get("CUFILE_FORCE_COMPAT_MODE", "").lower() != "true":
        raise ValueError("Set CUFILE_FORCE_COMPAT_MODE=true before starting vLLM")
    if os.environ.get("CUFILE_ALLOW_COMPAT_MODE", "").lower() != "true":
        raise ValueError("Set CUFILE_ALLOW_COMPAT_MODE=true before starting vLLM")

    from torch.cuda.gds import GdsFile

    dtypes = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }
    total = 0
    for path in sorted(files):
        with open(path, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            prefix = source.read(8)
            header_size = int.from_bytes(prefix, "little")
            if len(prefix) != 8 or not 0 < header_size < 64 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header length: {path}")
            if 8 + header_size > size:
                raise ValueError(f"Truncated safetensors header: {path}")
            header = json.loads(source.read(header_size))

        handle = GdsFile(path, os.O_RDONLY)
        started = time.monotonic()
        try:
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                dtype = dtypes.get(info["dtype"])
                if dtype is None:
                    raise ValueError(f"Unsupported cuFile dtype: {info['dtype']}")
                begin, end = info["data_offsets"]
                shape = info["shape"]
                if any(type(dim) is not int or dim < 0 for dim in shape):
                    raise ValueError(f"Invalid tensor shape: {path}:{name}")
                if not 0 <= begin <= end <= size - 8 - header_size:
                    raise ValueError(f"Invalid tensor offsets: {path}:{name}")
                if end - begin != math.prod(shape) * dtype.itemsize:
                    raise ValueError(f"Tensor byte count mismatch: {path}:{name}")
                raw = torch.empty(end - begin, dtype=torch.uint8, device="cuda")
                if end > begin:
                    handle.load_storage(
                        raw.untyped_storage(), offset=8 + header_size + begin
                    )
                tensor = raw.view(dtype).reshape(shape)
                total += end - begin
                yield name, tensor
                del tensor, raw
            torch.cuda.synchronize()
        finally:
            handle.close()
        logger.info(
            "BB_CUFILE shard=%s seconds=%.2f cumulative_bytes=%d",
            path,
            time.monotonic() - started,
            total,
        )
    logger.info("BB_CUFILE_COMPLETE bytes=%d", total)
