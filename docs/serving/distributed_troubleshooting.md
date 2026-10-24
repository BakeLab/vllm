# Troubleshooting distributed deployments

For general troubleshooting, see [Troubleshooting](../usage/troubleshooting.md).

## Verify launch configuration

Use the same model path and package versions on every node. For multiprocessing,
set the same `--nnodes`, `--master-addr`, and `--master-port` on every node and a
unique `--node-rank`. Start nonzero ranks with `--headless`, as shown in
[Multi-node deployment](parallelism_scaling.md#multi-node-deployment).

Ensure that each node advertises its reachable private IP with `VLLM_HOST_IP`
and can reach the master and peer communication ports. Collect logs from every
rank; one failed worker can leave the remaining ranks waiting for collectives.

## Verify inter-node GPU communication

Run the [communication test](../usage/troubleshooting.md#incorrect-hardwaredriver)
before loading a model. Set communication variables such as
`NCCL_SOCKET_IFNAME=eth0` and `NCCL_DEBUG=INFO` on every node before starting
workers. A shell environment change on the head node does not propagate to
workers launched on other nodes.

## Recovery and observability

The deployment controller is responsible for resource allocation, process health,
and coordinated restarts. A rank restart does not restore in-flight requests or
KV cache. Keep per-rank logs and use metrics and optional OTLP tracing to
distinguish queueing, model execution, and communication delays.
