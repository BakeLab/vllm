# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.all_reduce_utils import (
    ROCM_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False

logger = init_logger(__name__)


def _all_ranks_support(group: ProcessGroup, supported: bool) -> bool:
    group_support = torch.tensor(int(supported), dtype=torch.int32, device="cpu")
    dist.all_reduce(group_support, op=dist.ReduceOp.MIN, group=group)
    return bool(group_support.item())


def _rocm_group_can_use_global_heap(group: ProcessGroup, world_size: int) -> bool:
    from vllm.distributed.parallel_state import in_the_same_node_as

    global_world_size = dist.get_world_size()
    if world_size != global_world_size:
        return False
    if dist.get_process_group_ranks(group) != list(range(global_world_size)):
        return False
    return all(in_the_same_node_as(group, source_rank=0))


class SymmMemCommunicator:
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
        "10.3": [6, 8],
        "10.7": [6, 8],  # sm_107 (Rubin): reuse 10.3 thresholds
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        # add options for testing
        force_multimem: bool | None = None,
        max_size_override: int | None = None,
    ):
        self.disabled = True

        if not symm_mem_available:
            return

        is_rocm = current_platform.is_rocm()
        if not (current_platform.is_cuda() or is_rocm):
            logger.warning("SymmMemCommunicator: symmetric memory is not available.")
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.accelerator.set_device_index(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        capability = current_platform.get_device_capability()
        if capability is None:
            logger.warning(
                "SymmMemCommunicator: device capability is unknown, "
                "communicator is not available."
            )
            return
        self.device_capability = capability.as_version_str()
        max_sizes: dict[int, int] | None
        if is_rocm:
            max_sizes = ROCM_SYMM_MEM_ALL_REDUCE_MAX_SIZES
        else:
            max_sizes = SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(self.device_capability)
        if max_sizes is None:
            logger.warning(
                "SymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in max_sizes:
            logger.warning(
                "SymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        if is_rocm:
            if not _rocm_group_can_use_global_heap(self.group, self.world_size):
                logger.warning(
                    "SymmMemCommunicator: rocSHMEM uses a process-global symmetric "
                    "heap, so its process group must contain every rank on one node. "
                    "Communicator is not available."
                )
                return
            backend_supported = (
                torch_symm_mem.is_nvshmem_available()
                and torch_symm_mem.get_backend("cuda") == "NVSHMEM"
            )
            if not _all_ranks_support(self.group, backend_supported):
                logger.warning(
                    "SymmMemCommunicator: rocSHMEM must be available and selected "
                    "with TORCH_SYMMMEM=NVSHMEM on every rank. Communicator is not "
                    "available."
                )
                return
        # Use override max_size if provided, otherwise use default
        if max_size_override is not None:
            self.max_size = max_size_override
            logger.info(
                "SymmMemCommunicator: Using override max_size: %s bytes",
                self.max_size,
            )
        else:
            self.max_size = max_sizes[self.world_size]
        try:
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=self.device,
                dtype=self.dtype,
            )
            handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        except RuntimeError as e:
            if is_rocm:
                raise RuntimeError(
                    "SymmMemCommunicator: rocSHMEM initialization failed. "
                    "All ranks must terminate because its symmetric heap is "
                    "process-global."
                ) from e
            logger.warning_once(
                "SymmMemCommunicator: symmetric memory initialization failed: %s "
                "Communicator is not available. To suppress this warning set "
                "VLLM_ALLREDUCE_USE_SYMM_MEM=0",
                str(e),
            )
            return
        if is_rocm:
            peer_ptrs_valid = all(handle.buffer_ptrs) and all(handle.signal_pad_ptrs)
            if not _all_ranks_support(self.group, peer_ptrs_valid):
                self.buffer = None
                logger.warning(
                    "SymmMemCommunicator: rocSHMEM did not provide directly "
                    "accessible buffer and signal pointers for every rank. "
                    "Communicator is not available."
                )
                return
        self.multimem_supported = handle.multicast_ptr != 0 and not is_rocm
        if not self.multimem_supported and not is_rocm:
            logger.warning(
                "SymmMemCommunicator: symmetric memory "
                "multicast operations are not supported."
            )
            return
        self.force_multimem = force_multimem
        self.disabled = False
        if envs.VLLM_BATCH_INVARIANT:
            self.disabled = True

    def should_use_symm_mem(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size <= self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self.should_use_symm_mem(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))

        # Determine which algorithm to use
        use_multimem = False
        if not self.multimem_supported:
            use_multimem = False
        elif self.force_multimem is not None:
            # Test override: use forced setting
            use_multimem = self.force_multimem
        else:
            # Normal logic: use multimem for supported world sizes
            use_multimem = (
                self.world_size in self._WORLD_SIZES_MULTIMEM[self.device_capability]
            )

        if use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out
