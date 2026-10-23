# Installation

vLLM supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [Apple Silicon](gpu.md) (via [vLLM-Metal](https://github.com/vllm-project/vllm-metal))
- [CPU](cpu.md)
    - [Intel/AMD x86](cpu.md#intelamd-x86)
    - [ARM AArch64](cpu.md#arm-aarch64)
    - [Apple silicon](cpu.md#apple-silicon)

## Optional Python HTTP tools

The Rust frontend used by `vllm serve` does not require FastAPI, Uvicorn, or
Starlette. Install the `http` extra when using Python `run-batch`, `preload`,
the multi-port data-parallel supervisor, or the Mooncake/HF3FS metadata servers:

```bash
uv pip install 'vllm[http]'
```

For a source checkout, use `uv pip install -e '.[http]' --no-build-isolation`.
The extra preserves the minimum versions required by these optional tools.

## Hardware Plugins

vLLM supports third-party hardware plugins that live **outside** the main `vllm` repository. These follow the [Hardware-Pluggable RFC](../../design/plugin_system.md).

A list of all supported hardware can be found on the vLLM website, see [Universal Compatibility - Hardware](https://vllm.ai/#compatibility).

If you want to add new hardware, please contact us on [Slack](https://slack.vllm.ai/) or [Email](mailto:collaboration@vllm.ai).
