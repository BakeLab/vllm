# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in safetensors loading through cuFile compatibility or PCI P2PDMA."""

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch

from vllm import envs
from vllm.model_executor.model_loader.direct_io import (
    check_compat_env,
    direct_weights_iterator,
)


def cufile_weights_iterator(
    files: list[str],
    local_expert_ids: set[int] | None = None,
    *,
    is_unused_weight: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor]]:
    """Read safetensors payloads into the current CUDA device through cuFile.

    Expert filtering is unsupported. The selected mode is configured before
    CUDA/cuFile initialization.
    """
    if local_expert_ids is not None:
        raise ValueError("cuFile loading requires TP without EP")
    mode = envs.VLLM_CUFILE_MODE
    check_compat_env("CUFILE", mode)

    from torch.cuda.gds import GdsFile

    flags = os.O_RDONLY | (os.O_DIRECT if mode == "p2p" else 0)

    @contextmanager
    def open_reader(path: str):
        handle = GdsFile(path, flags)
        try:

            def read(raw: torch.Tensor, offset: int, length: int) -> None:
                start = raw.storage_offset()
                storage = raw.untyped_storage()[start : start + length]
                handle.load_storage(storage, offset=offset)

            yield read
        finally:
            handle.close()

    yield from direct_weights_iterator(
        files,
        backend="cuFile",
        mode=mode,
        open_reader=open_reader,
        is_unused_weight=is_unused_weight,
    )
