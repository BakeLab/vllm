# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-first experimental GPU collectives over NIXL/UCX."""

import contextlib
import math
import time
import uuid
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.platforms import current_platform


class NixlCollective:
    """Experimental collectives using registered GPU buffers and NIXL UCX.

    This implementation gathers each rank's input to every rank, then performs
    reductions locally. It is intentionally a simple bring-up path, not an
    efficient collective algorithm. Calls are synchronous and are not
    CUDA-graph capturable.
    """

    def __init__(self, cpu_group: ProcessGroup, device: torch.device, name: str):
        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "The UCX distributed backend requires the platform's NIXL "
                "package (nixl for CUDA or nixl_rocm for ROCm)."
            )
        self.cpu_group = cpu_group
        self.device = device
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.memory_type = current_platform.get_nixl_memory_type() or "VRAM"
        agent_name = f"vllm-{name}-{self.rank}-{uuid.uuid4().hex[:8]}"
        self.nixl = NixlWrapper(
            agent_name, nixl_agent_config(capture_telemetry=False)
        )
        self.remote_agents: dict[int, str] = {}
        metadata = self.nixl.get_agent_metadata()
        all_metadata: list[bytes | None] = [None] * self.world_size
        dist.all_gather_object(all_metadata, metadata, group=cpu_group)
        try:
            for rank, peer_metadata in enumerate(all_metadata):
                if rank != self.rank:
                    assert peer_metadata is not None
                    self.remote_agents[rank] = self.nixl.add_remote_agent(
                        peer_metadata
                    )
        except Exception:
            self.close()
            raise

    def _check_capture(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The experimental UCX collective backend cannot run inside "
                "a CUDA graph. Start vLLM with --enforce-eager."
            )

    def _device_id(self, tensor: torch.Tensor) -> int:
        return int(tensor.device.index or 0)

    def _register(self, tensor: torch.Tensor) -> Any:
        if not tensor.is_contiguous():
            raise ValueError("UCX collective tensors must be contiguous")
        if tensor.numel() == 0:
            raise ValueError("UCX collectives do not support empty tensors yet")
        if tensor.device != self.device:
            raise ValueError(
                f"UCX collective expected {self.device}, got {tensor.device}"
            )
        descs = self.nixl.get_reg_descs(
            [
                (
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                    self._device_id(tensor),
                    "",
                )
            ],
            self.memory_type,
        )
        self.nixl.register_memory(descs, backends=["UCX"])
        return descs

    def _wait(self, handle: int) -> None:
        deadline = time.monotonic() + 300
        while True:
            state = self.nixl.check_xfer_state(handle)
            if state == "DONE":
                return
            if state != "PROC":
                raise RuntimeError(f"NIXL UCX transfer failed: {state}")
            if time.monotonic() >= deadline:
                raise TimeoutError("NIXL UCX collective transfer timed out")
            time.sleep(0.0005)

    def _read(
        self,
        peer: int,
        local_ranges: list[tuple[int, int, int]],
        remote_ranges: list[tuple[int, int, int]],
    ) -> None:
        local_handle = None
        remote_handle = None
        xfer_handle = None
        try:
            local_desc = self.nixl.get_xfer_descs(local_ranges, self.memory_type)
            local_handle = self.nixl.prep_xfer_dlist("NIXL_INIT_AGENT", local_desc)
            remote_desc = self.nixl.get_xfer_descs(remote_ranges, self.memory_type)
            remote_handle = self.nixl.prep_xfer_dlist(
                self.remote_agents[peer], remote_desc
            )
            indices = np.arange(len(local_ranges), dtype=np.int32)
            xfer_handle = self.nixl.make_prepped_xfer(
                "READ", local_handle, indices, remote_handle, indices
            )
            self.nixl.transfer(xfer_handle)
            self._wait(xfer_handle)
        finally:
            if xfer_handle is not None:
                self.nixl.release_xfer_handle(xfer_handle)
            if local_handle is not None:
                self.nixl.release_dlist_handle(local_handle)
            if remote_handle is not None:
                self.nixl.release_dlist_handle(remote_handle)

    def all_gatherv(
        self, input_: torch.Tensor, dim: int = 0, sizes: list[int] | None = None
    ) -> torch.Tensor:
        self._check_capture()
        if input_.dim() == 0:
            raise ValueError("UCX all-gather requires a tensor with at least 1 dim")
        if not -input_.dim() <= dim < input_.dim():
            raise ValueError(f"invalid all-gather dimension {dim}")
        dim %= input_.dim()
        local = input_.movedim(dim, 0).contiguous()
        torch.cuda.synchronize(self.device)
        local_size = local.shape[0]
        if sizes is not None and (
            len(sizes) != self.world_size or sizes[self.rank] != local_size
        ):
            raise ValueError("all_gatherv sizes must match the local input")

        input_descs = self._register(local)
        output: torch.Tensor | None = None
        output_descs: Any = None
        try:
            metadata = (
                local.data_ptr(),
                local.numel() * local.element_size(),
                self._device_id(local),
                local_size,
                tuple(local.shape[1:]),
                str(local.dtype),
            )
            peer_metadata: list[tuple | None] = [None] * self.world_size
            dist.all_gather_object(peer_metadata, metadata, group=self.cpu_group)
            if any(item is None for item in peer_metadata):
                raise RuntimeError("missing UCX all-gather peer metadata")
            shapes = [item[4:] for item in peer_metadata if item is not None]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError("UCX all-gather ranks must use matching tensor types")
            peer_sizes = [int(item[3]) for item in peer_metadata if item is not None]
            if sizes is not None and peer_sizes != sizes:
                raise ValueError("all_gatherv sizes disagree across ranks")
            total_size = sum(peer_sizes)
            output = torch.empty(
                (total_size,) + local.shape[1:], dtype=local.dtype, device=local.device
            )
            output_descs = self._register(output)
            offsets = np.cumsum([0, *peer_sizes[:-1]]).tolist()
            own_count = peer_sizes[self.rank]
            if own_count:
                output.narrow(0, offsets[self.rank], own_count).copy_(local)
            torch.cuda.synchronize(self.device)
            for peer, source in enumerate(peer_metadata):
                assert source is not None
                count = peer_sizes[peer]
                if not count:
                    continue
                if peer == self.rank:
                    continue
                elements_per_item = math.prod(local.shape[1:])
                byte_offset = offsets[peer] * elements_per_item * local.element_size()
                byte_count = count * elements_per_item * local.element_size()
                self._read(
                    peer,
                    [
                        (
                            output.data_ptr() + byte_offset,
                            byte_count,
                            self._device_id(output),
                        )
                    ],
                    [(int(source[0]), int(source[1]), int(source[2]))],
                )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.cpu_group)
            return output.movedim(0, dim).contiguous()
        finally:
            if output_descs is not None:
                self.nixl.deregister_memory(output_descs)
            self.nixl.deregister_memory(input_descs)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        return self.all_gatherv(input_, dim)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if input_.dim() == 0:
            return self.all_reduce(input_.reshape(1)).reshape(())
        gathered = self.all_gather(input_, dim=0)
        return gathered.reshape((self.world_size,) + input_.shape).sum(dim=0)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        self._check_capture()
        if not 0 <= src < self.world_size:
            raise ValueError(f"invalid source rank {src} for broadcast")
        registered = self._register(tensor)
        try:
            torch.cuda.synchronize(self.device)
            metadata = (
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                self._device_id(tensor),
                tuple(tensor.shape),
                str(tensor.dtype),
            )
            peer_metadata: list[tuple | None] = [None] * self.world_size
            dist.all_gather_object(peer_metadata, metadata, group=self.cpu_group)
            if any(item is None for item in peer_metadata):
                raise RuntimeError("missing UCX broadcast peer metadata")
            if any(
                item[3:] != metadata[3:]
                for item in peer_metadata
                if item is not None
            ):
                raise ValueError("UCX broadcast ranks must use matching tensors")
            if self.rank != src:
                source = peer_metadata[src]
                assert source is not None
                self._read(
                    src,
                    [(tensor.data_ptr(), int(source[1]), self._device_id(tensor))],
                    [(int(source[0]), int(source[1]), int(source[2]))],
                )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.cpu_group)
            return tensor
        finally:
            self.nixl.deregister_memory(registered)

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        self._check_capture()
        if not 0 <= dst < self.world_size or dst == self.rank:
            raise ValueError(f"invalid UCX send destination rank {dst}")
        registered = self._register(tensor)
        global_dst = dist.get_global_rank(self.cpu_group, dst)
        try:
            torch.cuda.synchronize(self.device)
            metadata = (
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                self._device_id(tensor),
                tuple(tensor.shape),
                str(tensor.dtype),
            )
            dist.send_object_list([metadata], dst=global_dst, group=self.cpu_group)
            ack: list[Any] = [None]
            dist.recv_object_list(ack, src=global_dst, group=self.cpu_group)
            if ack[0] is not True:
                raise RuntimeError("UCX receiver did not acknowledge the transfer")
        finally:
            self.nixl.deregister_memory(registered)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int,
    ) -> torch.Tensor:
        self._check_capture()
        if not 0 <= src < self.world_size or src == self.rank:
            raise ValueError(f"invalid UCX receive source rank {src}")
        global_src = dist.get_global_rank(self.cpu_group, src)
        metadata: list[Any] = [None]
        dist.recv_object_list(metadata, src=global_src, group=self.cpu_group)
        source = metadata[0]
        if tuple(size) != tuple(source[3]) or str(dtype) != source[4]:
            dist.send_object_list([False], dst=global_src, group=self.cpu_group)
            raise ValueError("UCX send/receive tensor shape or dtype mismatch")
        output = torch.empty(size, dtype=dtype, device=self.device)
        registered = None
        try:
            registered = self._register(output)
            self._read(
                src,
                [
                    (
                        output.data_ptr(),
                        output.numel() * output.element_size(),
                        self._device_id(output),
                    )
                ],
                [(int(source[0]), int(source[1]), int(source[2]))],
            )
            torch.cuda.synchronize(self.device)
            dist.send_object_list([True], dst=global_src, group=self.cpu_group)
            return output
        except Exception:
            dist.send_object_list([False], dst=global_src, group=self.cpu_group)
            raise
        finally:
            if registered is not None:
                self.nixl.deregister_memory(registered)

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if input_.dim() == 0 or not -input_.dim() <= dim < input_.dim():
            raise ValueError(f"invalid reduce-scatter dimension {dim}")
        if dim < 0:
            dim += input_.dim()
        reduced = self.all_reduce(input_)
        if sizes is None:
            if reduced.shape[dim] % self.world_size:
                raise ValueError("reduce_scatter requires evenly sized shards")
            sizes = [reduced.shape[dim] // self.world_size] * self.world_size
        if (
            len(sizes) != self.world_size
            or any(size < 0 for size in sizes)
            or sum(sizes) != reduced.shape[dim]
        ):
            raise ValueError("reduce_scatter sizes must partition the input")
        start = sum(sizes[: self.rank])
        return reduced.narrow(dim, start, sizes[self.rank]).contiguous()

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.reduce_scatterv(input_, dim)

    def close(self) -> None:
        for agent_name in self.remote_agents.values():
            with contextlib.suppress(Exception):
                self.nixl.remove_remote_agent(agent_name)
        self.remote_agents.clear()
