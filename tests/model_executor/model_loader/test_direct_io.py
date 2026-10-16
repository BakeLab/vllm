# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.model_loader.direct_io import _iter_contiguous_groups


def _names(groups):
    return [[entry[0] for entry in group] for group in groups]


def test_contiguous_groups_split_at_gaps_and_byte_limit():
    entries = [
        ("a", torch.float32, [2], 8, 8),
        ("b", torch.float16, [2], 16, 4),
        ("c", torch.uint8, [4], 20, 4),
        ("gap", torch.float32, [1], 25, 4),
    ]

    assert _names(_iter_contiguous_groups(entries, 16)) == [["a", "b", "c"], ["gap"]]
    assert _names(_iter_contiguous_groups(entries[:3], 8)) == [["a"], ["b", "c"]]


def test_contiguous_groups_preserve_dtype_alignment():
    entries = [
        ("half", torch.float16, [1], 8, 2),
        ("float", torch.float32, [1], 10, 4),
    ]

    assert _names(_iter_contiguous_groups(entries, 16)) == [["half"], ["float"]]
