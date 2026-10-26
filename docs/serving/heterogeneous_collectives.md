# Experimental heterogeneous GPU collectives

vLLM has an opt-in experimental communication path for trying one job across
NVIDIA and AMD GPU nodes. It keeps the control process group on Gloo and uses
the platform's NIXL package with its UCX backend to move registered GPU buffers
between ranks. It does not require changes to the PyTorch source tree.

## Enable

Install a NIXL build with the UCX backend on every node (`nixl` for CUDA and
`nixl_rocm` for ROCm), then set this before starting every worker:

```bash
export VLLM_DISTRIBUTED_BACKEND=ucx
```

An external launcher such as `torchrun` must initialize the default process
group with Gloo. Disable split process groups and CUDA graphs for this
prototype; launch vLLM with `--enforce-eager`. For MoE, use
`--all2all-backend naive` or `--all2all-backend allgather_reducescatter`.

## Scope and limitations

The current implementation uses a correctness-first all-gather: each rank
reads every other rank's registered input into a local output buffer. AllReduce
gathers all inputs and sums them locally; ReduceScatter reduces the full input
then slices the result. This uses substantially more network bandwidth and
temporary GPU memory than ring or hierarchical collectives.

Transfers use NIXL's `READ` operation over its UCX backend. UCX documents
full GPU-memory support for tagged sends and receives, while GPU-memory support
for RMA operations is partial. See the
[UCX GPU FAQ](https://github.com/openucx/ucx/blob/master/docs/source/faq.md).
This prototype does not establish which UCX primitive NIXL selects or prove
direct GPU-to-NIC transfers across CUDA and ROCm. Confirm the selected
transport and end-to-end correctness on the target hardware before drawing
performance conclusions.

This mode is synchronous, is not CUDA-graph capturable, rejects empty
collective tensors, and currently does not support batched point-to-point
operations. It is an experimental starting point, not a production collective
library.
