# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager

import pytest
import torch

from vllm.model_executor.model_loader import direct_io
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


@pytest.mark.parametrize(("mode", "expected_reads"), [("compat", 1), ("p2p", 2)])
def test_direct_iterator_coalesces_compat_but_preserves_p2p_reads(
    monkeypatch, tmp_path, mode, expected_reads
):
    values = {
        "first": torch.arange(4, dtype=torch.uint8),
        "second": torch.arange(4, dtype=torch.uint8) + 4,
    }
    path = tmp_path / "weights.safetensors"
    header = {}
    payload_parts = []
    offset = 0
    for name, value in values.items():
        data = bytes(value.tolist())
        header[name] = {
            "dtype": "U8",
            "shape": [value.numel()],
            "data_offsets": [offset, offset + len(data)],
        }
        payload_parts.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    path.write_bytes(
        len(header_bytes).to_bytes(8, "little") + header_bytes + b"".join(payload_parts)
    )
    payload = path.read_bytes()
    reads = []

    @contextmanager
    def open_reader(_):
        def read(raw, offset, length):
            reads.append((offset, length))
            raw.copy_(torch.tensor(list(payload[offset : offset + length])))

        yield read

    monkeypatch.setattr(
        direct_io,
        "_empty",
        lambda size: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    loaded = dict(
        direct_io.direct_weights_iterator(
            [str(path)], backend="cuFile", mode=mode, open_reader=open_reader
        )
    )

    assert len(reads) == expected_reads
    for name, value in values.items():
        torch.testing.assert_close(loaded[name], value)
