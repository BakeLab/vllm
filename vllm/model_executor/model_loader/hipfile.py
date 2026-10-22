# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in safetensors loading into ROCm device memory through hipFile."""

import ctypes
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch

from vllm import envs
from vllm.model_executor.model_loader.direct_io import (
    WeightDestination,
    check_compat_env,
    direct_weights_iterator,
)


def hipfile_weights_iterator(
    files: list[str],
    local_expert_ids: set[int] | None = None,
    *,
    is_unused_weight: Callable[[str], bool] | None = None,
    get_destination: WeightDestination | None = None,
) -> Generator[tuple[str, torch.Tensor]]:
    """Read safetensors payloads into the current ROCm device through hipFile."""
    if local_expert_ids is not None:
        raise ValueError("hipFile loading requires TP without EP")
    if torch.version.hip is None:
        raise RuntimeError("hipFile loading requires a ROCm PyTorch build")
    mode = envs.VLLM_HIPFILE_MODE
    check_compat_env("HIPFILE", mode)

    try:
        from hipfile import Buffer, Driver, FileHandle
    except ImportError as error:
        raise RuntimeError(
            "hipFile loading requires the hipfile Python bindings"
        ) from error

    flags = os.O_RDONLY | (os.O_DIRECT if mode == "p2p" else 0)

    @contextmanager
    def open_reader(path: str):
        with FileHandle(path, flags) as handle:

            def read(raw: torch.Tensor, offset: int, length: int) -> None:
                pointer = ctypes.c_void_p(raw.data_ptr())
                with Buffer.from_ctypes_void_p(pointer, length, 0) as buffer:
                    count = handle.read(buffer, length, offset, 0)
                if count != length:
                    raise EOFError(f"Short hipFile read: {path}: {count}/{length}")

            yield read

    with Driver():
        yield from direct_weights_iterator(
            files,
            backend="hipFile",
            mode=mode,
            open_reader=open_reader,
            is_unused_weight=is_unused_weight,
            get_destination=get_destination,
        )
