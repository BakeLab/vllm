# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed tensor views for direct-I/O checkpoint loading."""

import math
import weakref

import torch
from torch.utils._pytree import tree_map

from vllm.model_executor.model_loader import direct_io
from vllm.model_executor.model_loader.direct_io import (
    _wait_read,
    iter_safetensors_entries,
)

_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.view.default,
    torch.ops.aten.view.dtype,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.unbind.int,
}


class DirectFile:
    """Keep a reader alive while any checkpoint view still refers to it."""

    def __init__(self, path, open_reader, device):
        self.path = path
        self.open_reader = open_reader
        self.device = torch.device(device)
        self.reader = None

    def read(self, target, offset, length):
        if self.reader is None:
            context = self.open_reader(self.path)
            self.reader = context.__enter__()
            weakref.finalize(self, context.__exit__, None, None, None)
        return self.reader(target, offset, length)


class DirectTensor(torch.Tensor):
    """Defer device reads until a weight loader consumes a sliced tensor."""

    @staticmethod
    def __new__(cls, source, metadata, file_offset, file_length, state=None):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            metadata.shape,
            strides=metadata.stride(),
            storage_offset=metadata.storage_offset(),
            dtype=metadata.dtype,
            device=source.device,
            requires_grad=False,
        )

    def __init__(self, source, metadata, file_offset, file_length, state=None):
        self._source = source
        self._metadata = metadata
        self._file_offset = file_offset
        self._file_length = file_length
        self._loaded = None
        self._state = (
            state if state is not None else {"metadata": metadata, "data": None}
        )

    def _materialize_family(self):
        if self._state["data"] is None:
            root = DirectTensor(
                self._source,
                self._state["metadata"],
                self._file_offset,
                self._file_length,
                self._state,
            )
            self._state["data"] = root._materialize()

    def _materialize(self):
        data = self._state["data"]
        if data is not None:
            meta = self._metadata
            return torch.empty(0, dtype=meta.dtype, device=data.device).set_(
                data.untyped_storage(),
                meta.storage_offset(),
                meta.shape,
                meta.stride(),
            )
        if self._loaded is not None:
            return self._loaded
        meta = self._metadata
        size = meta.numel() * meta.element_size()
        limit = direct_io.COMPAT_COALESCE_BYTES
        raw = (
            direct_io._try_device_buffer(size)
            if self._source.device.type == "cuda"
            else None
        )
        if raw is None:
            result = torch.empty(meta.shape, dtype=meta.dtype, device="cpu")
            raw = result.view(-1).view(torch.uint8)
        else:
            result = raw.view(meta.dtype).reshape(meta.shape)
        staging = None
        if self._source.device.type == "cuda" and raw.device.type == "cpu" and size:
            staging = direct_io._empty(min(size, limit))
        element_size = result.element_size()
        block = 1
        split = meta.ndim
        for axis in range(meta.ndim - 1, -1, -1):
            if meta.shape[axis] != 1 and meta.stride(axis) != block:
                break
            block *= meta.shape[axis]
            split = axis
        if result.device.type == "cuda":
            torch.accelerator.synchronize()
        prefix = meta.shape[:split]
        for index in range(math.prod(prefix)):
            remainder = index
            source_element = meta.storage_offset()
            for axis in range(len(prefix) - 1, -1, -1):
                coordinate = remainder % prefix[axis]
                remainder //= prefix[axis]
                source_element += coordinate * meta.stride(axis)
            source_byte = source_element * element_size
            length = block * element_size
            if source_byte < 0 or source_byte + length > self._file_length:
                raise ValueError("Direct tensor view exceeds checkpoint payload")
            for begin in range(0, length, limit):
                count = min(limit, length - begin)
                target = raw.narrow(0, index * length + begin, count)
                buffer = staging.narrow(0, 0, count) if staging is not None else target
                _wait_read(
                    self._source.read(
                        buffer, self._file_offset + source_byte + begin, count
                    )
                )
                if staging is not None:
                    target.copy_(buffer)
        if result.device.type == "cuda":
            torch.accelerator.synchronize()
        self._loaded = result
        return result

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _VIEW_OPS:
            sources = []

            def metadata(value):
                if isinstance(value, DirectTensor):
                    sources.append(value)
                    return value._metadata
                return value

            meta_args = tree_map(metadata, args)
            meta_kwargs = tree_map(metadata, kwargs)
            if len(sources) == 1:
                source = sources[0]
                result = func(*meta_args, **meta_kwargs)

                def wrap(value):
                    if isinstance(value, torch.Tensor):
                        return cls(
                            source._source,
                            value,
                            source._file_offset,
                            source._file_length,
                            source._state,
                        )
                    return value

                return tree_map(wrap, result)

        returned_aliases = set()
        for result in func._schema.returns:
            if result.alias_info is not None:
                returned_aliases.update(result.alias_info.before_set)
                returned_aliases.update(result.alias_info.after_set)

        def promote(value):
            if isinstance(value, DirectTensor):
                value._materialize_family()
            return value

        for index, argument in enumerate(func._schema.arguments):
            alias = argument.alias_info
            if alias is not None and (
                alias.is_write
                or returned_aliases.intersection(alias.before_set | alias.after_set)
            ):
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                tree_map(promote, value)

        def materialize(value):
            return value._materialize() if isinstance(value, DirectTensor) else value

        return func(*tree_map(materialize, args), **tree_map(materialize, kwargs))


def direct_tensor_iterator(files, open_reader, is_unused_weight=None, device=None):
    if device is None:
        device = direct_io._empty(0).device
    for path in files:
        source = DirectFile(path, open_reader, device)
        for name, dtype, shape, offset, length in iter_safetensors_entries(path):
            if is_unused_weight is not None and is_unused_weight(name):
                continue
            metadata = torch.empty(shape, dtype=dtype, device="meta")
            yield name, DirectTensor(source, metadata, offset, length)
