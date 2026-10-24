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
