# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from contextlib import contextmanager

import pytest
import torch
from safetensors.torch import save_file

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
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    loaded = dict(
        direct_io.direct_weights_iterator(
            [str(path)], backend="cuFile", mode=mode, open_reader=open_reader
        )
    )

    assert len(reads) == expected_reads
    for name, value in values.items():
        torch.testing.assert_close(loaded[name], value)


@pytest.mark.parametrize("mode", ["compat", "p2p"])
def test_large_cufile_tensor_bounds_compat_memory_and_reads_exact_tail(
    monkeypatch, tmp_path, mode
):
    from torch.cuda import gds

    from vllm.model_executor.model_loader import cufile

    # Split inside float elements and leave a partial final read at EOF.
    value = torch.arange(37, dtype=torch.float32)
    path = tmp_path / "large.safetensors"
    save_file({"weight": value}, path)
    payload = path.read_bytes()
    payload_offset = 8 + int.from_bytes(payload[:8], "little")
    reads, allocated, closed = [], [], []
    limit = 17

    class File:
        def __init__(self, filename, flags):
            assert filename == str(path)
            assert flags == os.O_RDONLY | (os.O_DIRECT if mode == "p2p" else 0)

        def load_storage(self, storage, offset):
            count = storage.nbytes()
            assert offset + count <= len(payload)
            reads.append((offset, count))
            target = torch.empty(0, dtype=torch.uint8).set_(storage)
            target.copy_(
                torch.frombuffer(
                    bytearray(payload[offset : offset + count]), dtype=torch.uint8
                )
            )

        def close(self):
            closed.append(True)

    def device_empty(size):
        allocated.append(size)
        return torch.empty(size, dtype=torch.uint8)

    monkeypatch.setattr(gds, "GdsFile", File)
    monkeypatch.setattr(direct_io, "COMPAT_COALESCE_BYTES", limit)
    monkeypatch.setattr(direct_io, "_empty", device_empty)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setenv("VLLM_CUFILE_MODE", mode)
    monkeypatch.setenv(
        "CUFILE_FORCE_COMPAT_MODE", "true" if mode == "compat" else "false"
    )
    monkeypatch.setenv(
        "CUFILE_ALLOW_COMPAT_MODE", "true" if mode == "compat" else "false"
    )

    loaded = dict(cufile.cufile_weights_iterator([str(path)]))

    torch.testing.assert_close(loaded["weight"], value)
    size = value.numel() * value.element_size()
    if mode == "compat":
        assert max(allocated) <= limit
        assert reads == [
            (payload_offset + begin, min(limit, size - begin))
            for begin in range(0, size, limit)
        ]
    else:
        assert allocated == [size]
        assert reads == [(payload_offset, size)]
    assert closed == [True]


@pytest.mark.parametrize("mode", ["compat", "p2p"])
def test_pp_filter_skips_payloads_before_device_reads(monkeypatch, tmp_path, mode):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict(
        {"1": PPMissingLayer(), "10": torch.nn.Identity()}
    )
    model.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"checkpoint.": "", "mtp.": None}
    )
    model.is_unused_checkpoint_weight = lambda name: name == "unused.weight"
    unused = DefaultModelLoader._get_unused_weight_filter(model)
    values = {
        name: torch.arange(4, dtype=torch.float32)
        for name in (
            "checkpoint.layers.1.weight",
            "checkpoint.layers.10.weight",
            "mtp.weight",
            "unused.weight",
        )
    }
    path = tmp_path / "mixed.safetensors"
    save_file(values, path)
    payload = path.read_bytes()
    reads = []

    @contextmanager
    def open_reader(_):
        def read(raw, offset, length):
            reads.append((offset, length))
            raw.copy_(
                torch.frombuffer(
                    bytearray(payload[offset : offset + length]), dtype=torch.uint8
                )
            )

        yield read

    monkeypatch.setattr(
        direct_io, "_empty", lambda n: torch.empty(n, dtype=torch.uint8)
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    loaded = dict(
        direct_io.direct_weights_iterator(
            [str(path)],
            backend="cuFile",
            mode=mode,
            open_reader=open_reader,
            is_unused_weight=unused,
        )
    )
    kept = "checkpoint.layers.10.weight"
    assert list(loaded) == [kept]
    torch.testing.assert_close(loaded[kept], values[kept])
    entry = next(
        e for e in direct_io.iter_safetensors_entries(str(path)) if e[0] == kept
    )
    assert reads == [(entry[3], entry[4])]
    reads.clear()
    assert not list(
        direct_io.direct_weights_iterator(
            [str(path)],
            backend="cuFile",
            mode=mode,
            open_reader=open_reader,
            is_unused_weight=lambda _: True,
        )
    )
    assert reads == []


@pytest.mark.parametrize("invalid", [False, True])
def test_destination_reads_preserve_storage_offset_and_reject_shape(
    monkeypatch, tmp_path, invalid
):
    value = torch.arange(37, dtype=torch.float32)
    path = tmp_path / "destination.safetensors"
    save_file({"weight": value}, path)
    payload = path.read_bytes()
    storage = torch.full((45,), -1.0)
    destination = storage[4:41] if not invalid else storage[4:40]
    reads = []

    @contextmanager
    def open_reader(_):
        def read(raw, offset, length):
            reads.append(length)
            raw.copy_(
                torch.frombuffer(
                    bytearray(payload[offset : offset + length]), dtype=torch.uint8
                )
            )

        yield read

    monkeypatch.setattr(direct_io, "_can_load_into", lambda _: True)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    iterator = direct_io.direct_weights_iterator(
        [str(path)],
        backend="test",
        mode="compat",
        open_reader=open_reader,
        coalesce_bytes=17,
        get_destination=lambda *_: destination,
    )
    if invalid:
        with pytest.raises(ValueError, match="Invalid direct-load destination"):
            list(iterator)
        assert reads == []
        assert torch.all(storage == -1)
    else:
        loaded = dict(iterator)
        assert loaded["weight"] is destination
        torch.testing.assert_close(destination, value)
        assert torch.all(storage[:4] == -1)
        assert torch.all(storage[41:] == -1)
        assert reads == [17] * 8 + [12]


def test_destination_read_failure_does_not_yield_partial_weight(monkeypatch, tmp_path):
    path = tmp_path / "failed.safetensors"
    save_file({"weight": torch.arange(8, dtype=torch.uint8)}, path)
    destination = torch.empty(8, dtype=torch.uint8)
    calls = []

    @contextmanager
    def open_reader(_):
        def read(raw, offset, length):
            calls.append(length)
            if len(calls) == 2:
                raise EOFError("short read")
            raw.fill_(1)

        yield read

    monkeypatch.setattr(direct_io, "_can_load_into", lambda _: True)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    iterator = direct_io.direct_weights_iterator(
        [str(path)],
        backend="test",
        mode="compat",
        open_reader=open_reader,
        coalesce_bytes=4,
        get_destination=lambda *_: destination,
    )
    with pytest.raises(EOFError, match="short read"):
        next(iterator)
    assert calls == [4, 4]
