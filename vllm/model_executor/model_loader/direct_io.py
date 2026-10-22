# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared safetensors iteration for direct storage-to-device loaders."""

import json
import math
import os
import time
from collections.abc import Callable, Generator, Iterable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import _natural_sort_key

logger = init_logger(__name__)

SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}

# Reads `length` bytes at `file_offset` into the uint8 device tensor `raw`.
Reader = Callable[[torch.Tensor, int, int], None]
TensorEntry = tuple[str, torch.dtype, list[int], int, int]
# Compatibility modes accept arbitrary offsets; P2P keeps per-tensor I/O for
# direct-I/O alignment requirements.
COMPAT_COALESCE_BYTES = 64 * 1024 * 1024


def check_compat_env(prefix: str, mode: str) -> None:
    """Require the driver's compat switches to match the selected mode."""
    required = "true" if mode == "compat" else "false"
    for variable in (f"{prefix}_FORCE_COMPAT_MODE", f"{prefix}_ALLOW_COMPAT_MODE"):
        if os.environ.get(variable, "").lower() != required:
            raise ValueError(f"Set {variable}={required} before starting vLLM")


def iter_safetensors_entries(
    path: str,
) -> Generator[tuple[str, torch.dtype, list[int], int, int]]:
    """Yield (name, dtype, shape, file_offset, length) for each tensor."""
    with open(path, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        prefix = source.read(8)
        header_size = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < header_size < 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {path}")
        if 8 + header_size > size:
            raise ValueError(f"Truncated safetensors header: {path}")
        header = json.loads(source.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")

    data_start = 8 + header_size
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {info['dtype']}")
        begin, end = info["data_offsets"]
        shape = info["shape"]
        if any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {path}:{name}")
        if not 0 <= begin <= end <= size - data_start:
            raise ValueError(f"Invalid tensor offsets: {path}:{name}")
        if end - begin != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Tensor byte count mismatch: {path}:{name}")
        yield name, dtype, shape, data_start + begin, end - begin


def _empty(length: int) -> torch.Tensor:
    return torch.empty(length, dtype=torch.uint8, device="cuda")


def _iter_contiguous_groups(
    entries: Iterable[TensorEntry], max_group_bytes: int
) -> Generator[list[TensorEntry]]:
    """Group adjacent entries whose dtype views remain properly aligned."""
    if max_group_bytes <= 0:
        raise ValueError("max_group_bytes must be positive")

    group: list[TensorEntry] = []
    group_bytes = 0
    for entry in entries:
        _, dtype, _, offset, length = entry
        can_append = not group or (
            group[-1][3] + group[-1][4] == offset
            and (offset - group[0][3]) % dtype.itemsize == 0
            and group_bytes + length <= max_group_bytes
        )
        if group and not can_append:
            yield group
            group = []
            group_bytes = 0
        group.append(entry)
        group_bytes += length
        if group_bytes >= max_group_bytes:
            yield group
            group = []
            group_bytes = 0
    if group:
        yield group


def direct_weights_iterator(
    files: Iterable[str],
    *,
    backend: str,
    mode: str,
    open_reader: Callable[[str], Any],
    coalesce_bytes: int | None = None,
    is_unused_weight: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor]]:
    """Read tensor payloads with bounded device staging in compatibility mode.

    `open_reader(path)` returns a context manager yielding a `Reader` for that
    file. When requested, adjacent entries are read into a bounded staging
    buffer to reduce per-read overhead. Only supplied files are read: the
    default loader supplies the complete checkpoint, ShardedStateLoader only
    this TP rank's saved shards.

    Oversized compatibility-mode tensors are assembled on CPU using bounded
    device reads, so TP slicing does not require a full tensor's temporary VRAM.
    """
    if coalesce_bytes is None:
        coalesce_bytes = COMPAT_COALESCE_BYTES if mode == "compat" else 0
    compat_buffer_bytes = coalesce_bytes or COMPAT_COALESCE_BYTES

    logger.info("%s loading mode=%s", backend, mode)
    total = 0
    for path in sorted(files, key=_natural_sort_key):
        started = time.monotonic()
        with open_reader(path) as read:
            entries = iter_safetensors_entries(path)
            if is_unused_weight is not None:
                entries = (entry for entry in entries if not is_unused_weight(entry[0]))
            if coalesce_bytes:
                groups = _iter_contiguous_groups(entries, coalesce_bytes)
            else:
                groups = ([entry] for entry in entries)
            for group in groups:
                first_offset = group[0][3]
                _, _, _, last_offset, last_length = group[-1]
                length = last_offset + last_length - first_offset
                if mode == "compat" and length > compat_buffer_bytes:
                    logger.info(
                        "%s compat bounded staging tensor=%s bytes=%d "
                        "device_buffer_bytes=%d",
                        backend,
                        group[0][0],
                        length,
                        compat_buffer_bytes,
                    )
                    raw = torch.empty(length, dtype=torch.uint8, device="cpu")
                    device_buffer = _empty(compat_buffer_bytes)
                    for begin in range(0, length, compat_buffer_bytes):
                        count = min(compat_buffer_bytes, length - begin)
                        chunk = device_buffer.narrow(0, 0, count)
                        read(chunk, first_offset + begin, count)
                        # Blocking D2H completes before the buffer is reused.
                        raw.narrow(0, begin, count).copy_(chunk)
                    del chunk, device_buffer
                else:
                    raw = _empty(length)
                    if length:
                        read(raw, first_offset, length)
                for name, dtype, shape, offset, tensor_length in group:
                    begin = offset - first_offset
                    tensor = (
                        raw.narrow(0, begin, tensor_length).view(dtype).reshape(shape)
                    )
                    total += tensor_length
                    yield name, tensor
                    del tensor
                del raw
            torch.accelerator.synchronize()
        logger.info(
            "%s mode=%s shard=%s seconds=%.2f cumulative_bytes=%d",
            backend,
            mode,
            path,
            time.monotonic() - started,
            total,
        )
    logger.info("%s loading complete mode=%s bytes=%d", backend, mode, total)
