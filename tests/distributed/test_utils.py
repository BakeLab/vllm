# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from tests.utils import run_gpu_workers
from vllm.utils.network_utils import get_open_port


def _collective_worker(port):
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        value = torch.tensor(rank + 1)
        dist.all_reduce(value)
        return rank, value.item()
    finally:
        dist.destroy_process_group()


def _failing_worker():
    raise ValueError("worker failure reaches the parent")


def test_spawned_workers_keep_rank_order_and_complete_collectives():
    worker = partial(_collective_worker, get_open_port())
    assert run_gpu_workers([worker, worker], timeout=120) == [(0, 3), (1, 3)]


def test_spawned_worker_failure_reaches_parent():
    with pytest.raises(RuntimeError, match="worker failure reaches the parent"):
        run_gpu_workers([_failing_worker], timeout=120)
