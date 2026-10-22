# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import save_file


def test_hipfile_reads_safetensors_payload_at_exact_offset(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader import direct_io, hipfile

    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([1, 2, 3, 4], dtype=torch.uint8)}, path)
    with path.open("rb") as source:
        payload_offset = 8 + int.from_bytes(source.read(8), "little")
    reads = []
    registrations = []
    allocated = []

    class FakeContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class FakeBuffer(FakeContext):
        @classmethod
        def from_ctypes_void_p(cls, pointer, length, flags):
            registrations.append((pointer.value, length, flags))
            return cls()

    class FakeFileHandle(FakeContext):
        def __init__(self, filename, flags):
            assert filename == str(path)
            assert flags == os.O_RDONLY

        def read(self, _buffer, length, file_offset, buffer_offset):
            reads.append((length, file_offset, buffer_offset))
            return length

    def cpu_empty(length):
        allocated.append(torch.empty(length, dtype=torch.uint8))
        return allocated[-1]

    monkeypatch.setattr(direct_io, "_empty", cpu_empty)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.version, "hip", "10.1")
    monkeypatch.setitem(
        sys.modules,
        "hipfile",
        SimpleNamespace(
            Buffer=FakeBuffer, Driver=FakeContext, FileHandle=FakeFileHandle
        ),
    )
    monkeypatch.setenv("VLLM_HIPFILE_MODE", "compat")
    monkeypatch.setenv("HIPFILE_FORCE_COMPAT_MODE", "true")
    monkeypatch.setenv("HIPFILE_ALLOW_COMPAT_MODE", "true")

    weights = list(hipfile.hipfile_weights_iterator([str(path)]))
    assert [(name, list(tensor.shape)) for name, tensor in weights] == [("weight", [4])]
    assert registrations == [(allocated[0].data_ptr(), 4, 0)]
    assert reads == [(4, payload_offset, 0)]


def test_direct_io_rejects_unknown_dtype(tmp_path):
    import json

    import pytest

    from vllm.model_executor.model_loader.direct_io import iter_safetensors_entries

    header = json.dumps(
        {"w": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}
    ).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(8))
    with pytest.raises(ValueError, match="Unsupported safetensors dtype"):
        list(iter_safetensors_entries(str(path)))
