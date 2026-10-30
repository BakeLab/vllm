# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import random
import typing
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators import symm_mem
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.engine.llm_engine import LLMEngine

torch.manual_seed(42)
random.seed(44)

test_size_elements = 1024 * 1024


@pytest.mark.parametrize(
    ("group_world_size", "group_ranks", "same_node", "expected"),
    [
        (2, [0, 1], True, False),
        (4, [0, 1, 2, 3], True, True),
        (4, [0, 1, 2, 4], True, False),
        (4, [0, 1, 2, 3], False, False),
    ],
)
def test_rocm_group_can_use_global_heap(
    monkeypatch, group_world_size, group_ranks, same_node, expected
):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(symm_mem.dist, "get_process_group_ranks", lambda _: group_ranks)
    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        lambda *args, **kwargs: [same_node] * group_world_size,
    )

    assert (
        symm_mem._rocm_group_can_use_global_heap(object(), group_world_size) is expected
    )


def test_rocm_symm_mem_uses_rocshmem_without_multicast(monkeypatch):
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(
        symm_mem,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: True,
            get_device_capability=lambda: SimpleNamespace(
                as_version_str=lambda: "12.0"
            ),
        ),
    )
    monkeypatch.setattr(symm_mem.torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(symm_mem, "_rocm_group_can_use_global_heap", lambda *args: True)
    monkeypatch.setattr(symm_mem, "_all_ranks_support", lambda _, value: value)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "is_nvshmem_available", lambda: True)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "get_backend", lambda _: "NVSHMEM")
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "set_backend",
        lambda _: pytest.fail("vLLM must not mutate the process-global backend"),
    )
    monkeypatch.setattr(
        symm_mem.torch_symm_mem, "empty", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "rendezvous",
        lambda *args, **kwargs: SimpleNamespace(
            multicast_ptr=0,
            buffer_ptrs=[1, 2],
            signal_pad_ptrs=[3, 4],
        ),
    )

    communicator = SymmMemCommunicator(
        SimpleNamespace(group_name="test"),
        "cuda:0",
    )

    assert not communicator.disabled
    assert not communicator.multimem_supported


def test_rocm_unsupported_world_size_does_not_change_backend(monkeypatch):
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(
        symm_mem,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: True,
            get_device_capability=lambda: SimpleNamespace(
                as_version_str=lambda: "12.0"
            ),
        ),
    )
    monkeypatch.setattr(symm_mem.torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _: 3)
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "set_backend",
        lambda _: pytest.fail("unsupported groups must not change the backend"),
    )

    communicator = SymmMemCommunicator(
        SimpleNamespace(group_name="test"),
        "cuda:0",
    )

    assert communicator.disabled


@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_rocm_conflicting_backend_disables_communicator(monkeypatch, backend):
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(
        symm_mem,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: True,
            get_device_capability=lambda: SimpleNamespace(
                as_version_str=lambda: "12.0"
            ),
        ),
    )
    monkeypatch.setattr(symm_mem.torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(symm_mem, "_rocm_group_can_use_global_heap", lambda *args: True)
    monkeypatch.setattr(symm_mem, "_all_ranks_support", lambda _, value: value)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "is_nvshmem_available", lambda: True)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "get_backend", lambda _: backend)
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "set_backend",
        lambda _: pytest.fail("configured backends must not be overwritten"),
    )

    communicator = SymmMemCommunicator(
        SimpleNamespace(group_name="test"),
        "cuda:0",
    )

    assert communicator.disabled


def test_rocm_subgroup_does_not_use_global_rocshmem_heap(monkeypatch):
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(
        symm_mem,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: True,
            get_device_capability=lambda: SimpleNamespace(
                as_version_str=lambda: "12.0"
            ),
        ),
    )
    monkeypatch.setattr(symm_mem.torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(
        symm_mem, "_rocm_group_can_use_global_heap", lambda *args: False
    )
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "get_backend",
        lambda _: pytest.fail("subgroups must return before inspecting the backend"),
    )

    communicator = SymmMemCommunicator(
        SimpleNamespace(group_name="test"),
        "cuda:0",
    )

    assert communicator.disabled


def test_rocm_null_peer_pointer_disables_communicator(monkeypatch):
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(
        symm_mem,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: True,
            get_device_capability=lambda: SimpleNamespace(
                as_version_str=lambda: "12.0"
            ),
        ),
    )
    monkeypatch.setattr(symm_mem.torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(symm_mem, "_rocm_group_can_use_global_heap", lambda *args: True)
    monkeypatch.setattr(symm_mem, "_all_ranks_support", lambda _, value: value)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "is_nvshmem_available", lambda: True)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "get_backend", lambda _: "NVSHMEM")
    monkeypatch.setattr(
        symm_mem.torch_symm_mem, "empty", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "rendezvous",
        lambda *args, **kwargs: SimpleNamespace(
            multicast_ptr=0,
            buffer_ptrs=[1, 0],
            signal_pad_ptrs=[2, 3],
        ),
    )

    communicator = SymmMemCommunicator(
        SimpleNamespace(group_name="test"),
        "cuda:0",
    )

    assert communicator.disabled
    assert communicator.buffer is None


def test_rocm_initialization_failure_is_not_rank_local_fallback(monkeypatch):
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(
        symm_mem,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: True,
            get_device_capability=lambda: SimpleNamespace(
                as_version_str=lambda: "12.0"
            ),
        ),
    )
    monkeypatch.setattr(symm_mem.torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(symm_mem, "_rocm_group_can_use_global_heap", lambda *args: True)
    monkeypatch.setattr(symm_mem, "_all_ranks_support", lambda _, value: value)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "is_nvshmem_available", lambda: True)
    monkeypatch.setattr(symm_mem.torch_symm_mem, "get_backend", lambda _: "NVSHMEM")
    monkeypatch.setattr(
        symm_mem.torch_symm_mem,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("allocation failed")
        ),
    )

    with pytest.raises(RuntimeError, match="All ranks must terminate"):
        SymmMemCommunicator(
            SimpleNamespace(group_name="test"),
            "cuda:0",
        )


@pytest.mark.parametrize("strided_output", [False, True])
def test_rocm_symm_mem_all_reduce_preserves_output(monkeypatch, strided_output):
    monkeypatch.setattr(
        symm_mem, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )
    communicator = object.__new__(SymmMemCommunicator)
    communicator.disabled = False
    communicator.dtype = torch.bfloat16
    communicator.buffer = torch.full((8,), -3, dtype=torch.bfloat16)
    communicator.max_size = communicator.buffer.nbytes
    communicator.group = SimpleNamespace(group_name="test")

    def one_shot_copy(buffer, inp, reduce_op, group_name, out):
        assert (reduce_op, group_name) == ("sum", "test")
        torch.testing.assert_close(buffer, torch.full_like(buffer, -3))
        out.copy_(inp + 1)

    monkeypatch.setattr(
        torch.ops.symm_mem, "one_shot_all_reduce_copy_out", one_shot_copy
    )
    monkeypatch.setattr(
        torch.ops.symm_mem,
        "two_shot_all_reduce_",
        lambda *args: pytest.fail("ROCm must use out-of-place one-shot"),
    )
    inp = torch.ones((2, 4), dtype=torch.bfloat16)
    out = (
        torch.empty((4, 2), dtype=inp.dtype).t()
        if strided_output
        else torch.empty_like(inp)
    )
    output = communicator.all_reduce(inp, out=out)

    assert output is out
    torch.testing.assert_close(output, torch.full_like(inp, 2))
    torch.testing.assert_close(inp, torch.ones_like(inp))


def test_rocm_symm_mem_max_size_boundary():
    communicator = object.__new__(SymmMemCommunicator)
    communicator.disabled = False
    communicator.dtype = torch.bfloat16
    communicator.max_size = 4 * 1024 * 1024

    assert communicator.should_use_symm_mem(
        torch.empty(communicator.max_size // 2, dtype=torch.bfloat16)
    )
    assert not communicator.should_use_symm_mem(
        torch.empty(communicator.max_size // 2 + 1, dtype=torch.bfloat16)
    )


def symm_mem_allreduce_worker(local_rank: int, world_size: int, q: mp.Queue):
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))

    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        symm_mem_comm = cuda_communicator.symm_mem_comm
        if symm_mem_comm is None or symm_mem_comm.disabled:
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator is not available or disabled.")
            return

        inp_direct_symm_mem = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        if not symm_mem_comm.should_use_symm_mem(inp_direct_symm_mem):
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator isn't used for this world and input size.")
            return

        original_inp_direct_symm_mem = inp_direct_symm_mem.clone()
        out_direct_symm_mem = symm_mem_comm.all_reduce(inp_direct_symm_mem)
        assert out_direct_symm_mem is not None

        group = get_tp_group().device_group
        dist.all_reduce(original_inp_direct_symm_mem, group=group)
        torch.testing.assert_close(
            out_direct_symm_mem, original_inp_direct_symm_mem, atol=2.5, rtol=0.1
        )

        # Test tensor_model_parallel_all_reduce which should use symm_mem
        inp_tensor_parallel = torch.randint(
            -23, 1, (test_size_elements,), dtype=dtype, device=device
        )
        original_inp_tensor_parallel = inp_tensor_parallel.clone()
        out_tensor_parallel = tensor_model_parallel_all_reduce(inp_tensor_parallel)
        dist.all_reduce(original_inp_tensor_parallel, group=group)
        torch.testing.assert_close(
            out_tensor_parallel, original_inp_tensor_parallel, atol=2.5, rtol=0.1
        )

        inputs = [
            torch.full(
                (2048,), (local_rank + 1) * (step + 1), dtype=dtype, device=device
            )
            for step in range(16)
        ]
        outputs = [torch.empty_like(inp) for inp in inputs]

        def run():
            for inp, out in zip(inputs, outputs):
                symm_mem_comm.all_reduce(inp, out=out)

        def check(sign):
            rank_sum = world_size * (world_size + 1) // 2
            correct = (
                torch.stack(
                    [
                        (out == sign * rank_sum * (step + 1)).all()
                        for step, out in enumerate(outputs)
                    ]
                )
                .all()
                .to(torch.int32)
            )
            dist.all_reduce(correct, op=dist.ReduceOp.MIN, group=group)
            assert correct.item() == 1

        run()
        check(1)
        device_module = torch.get_device_module(device)
        graph = device_module.CUDAGraph()
        with device_module.graph(graph):
            run()
        sign = 1
        for _ in range(3):
            for inp in inputs:
                inp.neg_()
            sign = -sign
            graph.replay()
            check(sign)


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="SymmMemAllreduce requires a CUDA or ROCm platform.",
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="Only test on CUDA or ROCm",
)
def test_symm_mem_allreduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    q = mp.get_context("spawn").Queue()
    mp.spawn(symm_mem_allreduce_worker, args=(world_size, q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
        if val is not None:
            if current_platform.is_rocm():
                pytest.fail(val)
            pytest.skip(val)


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="SymmMemAllreduce requires a CUDA or ROCm platform.",
)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="Only test on CUDA or ROCm",
)
def test_dp_with_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch):
    world_size = 4
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    # Verify that the DataParallel runs without error
    engine_args = EngineArgs(
        model="distilbert/distilgpt2",
        compilation_config={"mode": 1, "cudagraph_mode": "NONE"},
        enable_prefix_caching=True,
        data_parallel_size=2,
        tensor_parallel_size=2,
        data_parallel_backend="mp",
    )
    LLMEngine.from_engine_args(engine_args)
