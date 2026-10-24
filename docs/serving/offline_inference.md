# Direct inference interfaces

This fork uses Rust for public serving and inference clients. The Python
`vllm.LLM` API and `vllm run-batch` command have been removed. Use the Rust
`vllm-llm` crate for direct engine access, or connect an HTTP/gRPC client to
the Rust frontend. See the [Rust architecture](../../rust/README.md).

The Python/PyTorch engine remains responsible for model loading, Model Runner V2,
torch.compile, Triton kernels, and GPU execution on CUDA/ROCm. Removing the public
Python SDK does not remove this engine or make GPU kernels inherently faster.

Python-only model-runner V1 fallbacks, suffix/custom-class speculative decoding,
and external-launcher pipeline-parallel fallback are no longer available.
Unsupported configurations fail instead of selecting the removed runner.

For benchmarks use `vllm-rs bench serve` or standalone `vllm-bench`.
The standalone binary also provides `vllm-bench mm-processor`.
The former Python `vllm bench` implementation and selector flags are removed.

## Optional Python dependencies

ROCm's base installation no longer installs Tensorizer, Run:ai Model Streamer,
or Python gRPC/reflection. Use `vllm[tensorizer]` or `vllm[runai]` when selecting
those optional loaders. For Python-engine OpenTelemetry tracing, install
`vllm[otel]`; its OTLP exporter supplies its required gRPC transport dependencies.
The Rust frontend's gRPC service does not require Python grpcio/reflection.

Numba and its llvmlite dependency remain required for fused Kimi K2.5 vision
preprocessing, independently of the removed V1 n-gram implementation.
