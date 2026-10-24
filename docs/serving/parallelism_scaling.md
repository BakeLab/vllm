# Parallelism and Scaling

## Distributed inference strategies for a single-model replica

To choose a distributed inference strategy for a single-model replica, use the following guidelines:

- **Single GPU (no distributed inference):** if the model fits on a single GPU, distributed inference is probably unnecessary. Run inference on that GPU.
- **Single-node multi-GPU using tensor parallel inference:** if the model is too large for a single GPU but fits on a single node with multiple GPUs, use *tensor parallelism*. For example, set `tensor_parallel_size=4` when using a node with 4 GPUs.
- **Multi-node multi-GPU using tensor parallel and pipeline parallel inference:** if the model is too large for a single node, combine *tensor parallelism* with *pipeline parallelism*. Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes. For example, set `tensor_parallel_size=8` and `pipeline_parallel_size=2` when using 2 nodes with 8 GPUs per node.

Increase the number of GPUs and nodes until there is enough GPU memory for the model. Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes.

After you provision sufficient resources to fit the model, run `vllm`. Look for log messages like:

```text
INFO 07-23 13:56:04 [kv_cache_utils.py:775] GPU KV cache size: 643,232 tokens
INFO 07-23 13:56:04 [kv_cache_utils.py:779] Maximum concurrency for 40,960 tokens per request: 15.70x
```

The `GPU KV cache size` line reports the total number of tokens that can be stored in the GPU KV cache at once. The `Maximum concurrency` line provides an estimate of how many requests can be served concurrently if each request requires the specified number of tokens (40,960 in the example above). The tokens-per-request number is taken from the model configuration's maximum sequence length, `ModelConfig.max_model_len`. If these numbers are lower than your throughput requirements, add more GPUs or nodes to your cluster.

!!! note "Edge case: uneven GPU splits"
    If the model fits within a single node but the GPU count doesn't evenly divide the model size, enable pipeline parallelism, which splits the model along layers and supports uneven splits. In this scenario, set `tensor_parallel_size=1` and `pipeline_parallel_size` to the number of GPUs. Furthermore, if the GPUs on the node do not have NVLINK interconnect (e.g. L40S), leverage pipeline parallelism instead of tensor parallelism for higher throughput and lower communication overhead.

### Distributed serving of *Mixture of Experts* (*MoE*) models

It's often advantageous to exploit the inherent parallelism of experts by using a separate parallelism strategy for the expert layers. vLLM supports large-scale deployment combining Data Parallel attention with Expert or Tensor Parallel MoE layers. For more information, see [Data Parallel Deployment](data_parallel_deployment.md).

## Single-node deployment

vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. The implementation includes [Megatron-LM's tensor parallel algorithm](https://arxiv.org/pdf/1909.08053.pdf).

The default distributed runtime is native Python `multiprocessing` for both single-node and multi-node inference. Set `--distributed-executor-backend mp` explicitly if needed; `external_launcher` is also available for externally managed workers.

For multi-GPU inference, set `tensor_parallel_size` in the `LLM` class to the desired GPU count. For example, to run inference on 4 GPUs:

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

For multi-GPU serving, include `--tensor-parallel-size` when starting the server. For example, to run the API server on 4 GPUs:

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

To enable pipeline parallelism, add `--pipeline-parallel-size`. For example, to run the API server on 8 GPUs with pipeline parallelism and tensor parallelism:

```bash
# Eight GPUs total
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## Multi-node deployment

This fork does not support Ray, Ray Serve, Ray Data, or the `sharded_rdt`
weight-transfer backend. Use `mp` or `external_launcher` instead. Ray-specific
configuration fields, CLI flags, and `VLLM_RAY_*` environment controls have
been removed. OTLP tracing and NCCL/RCCL communication remain available.
The external deployment controller must allocate resources, launch all ranks,
monitor worker health, and coordinate group restarts.


If a single node lacks sufficient GPUs to hold the model, deploy vLLM across multiple nodes. Ensure that every node provides an identical execution environment, including the model path and Python packages. Using container images is recommended because they provide a convenient way to keep environments consistent and to hide host heterogeneity.

### Running vLLM with MultiProcessing

Multi-node vLLM deployments use `multiprocessing` as the runtime engine. Here's an example to deploy model across 2 nodes (8 GPUs per node) with `tp_size=8` and `pp_size=2`.

Choose one node as the head node and run:

```bash
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_NODE_IP>
```

On the other worker node, run:

```bash
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_NODE_IP> --headless
```

## Optimizing network communication for tensor parallelism

Efficient tensor parallelism requires fast internode communication, preferably through high-speed network adapters such as InfiniBand.
Set communication environment variables such as `NCCL_IB_HCA=mlx5` on every node before starting its vLLM process. Ensure containers have access to the required network devices.
Contact your system administrator for more information about the required flags.

## Enabling GPUDirect RDMA

GPUDirect RDMA (Remote Direct Memory Access) is an NVIDIA technology that allows network adapters to directly access GPU memory, bypassing the CPU and system memory. This direct access reduces latency and CPU overhead, which is beneficial for large data transfers between GPUs across nodes.

To enable GPUDirect RDMA with vLLM, configure the following settings:

- `IPC_LOCK` security context: add the `IPC_LOCK` capability to the container's security context to lock memory pages and prevent swapping to disk.
- Shared memory with `/dev/shm`: mount `/dev/shm` in the pod spec to provide shared memory for interprocess communication (IPC).

If you use Docker, set up the container as follows:

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

If you use Kubernetes, set up the pod spec as follows:

```yaml
...
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      resources:
        limits:
          nvidia.com/gpu: 8
        requests:
          nvidia.com/gpu: 8
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
...
```

!!! tip "Confirm GPUDirect RDMA operation"
    To confirm your InfiniBand card is using GPUDirect RDMA, run vLLM with detailed NCCL logs: `NCCL_DEBUG=TRACE vllm serve ...`.

    Then look for the NCCL version and the network used.

    - If you find `[send] via NET/IB/GDRDMA` in the logs, then NCCL is using InfiniBand with GPUDirect RDMA, which *is* efficient.
    - If you find `[send] via NET/Socket` in the logs, NCCL used a raw TCP socket, which *is not* efficient for cross-node tensor parallelism. 

!!! tip "Pre-download Hugging Face models"

## Troubleshooting distributed deployments

For information about distributed debugging, see [Troubleshooting distributed deployments](distributed_troubleshooting.md).
