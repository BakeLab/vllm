# hipFile weight loading

The optional ROCm loader reads safetensors payloads into GPU tensors through
the `hipfile` Python bindings. Install bindings built for the same ROCm runtime
as PyTorch, with `libhipfile.so` available to the dynamic linker. See AMD's
[Python binding instructions](https://rocm.docs.amd.com/projects/hipFile/en/latest/install/python-bindings.html).

Set the mode before starting vLLM workers:

```bash
export VLLM_HIPFILE_LOAD=1
export VLLM_HIPFILE_MODE=compat
export HIPFILE_FORCE_COMPAT_MODE=true
export HIPFILE_ALLOW_COMPAT_MODE=true
vllm serve /mnt/model --load-format safetensors
```

Compatibility mode works with unaligned safetensors payloads, but can copy
through host memory. Adjacent tensor payloads are coalesced into reads capped
at 64 MiB; a larger individual tensor is read on its own. P2P mode keeps
per-tensor reads to preserve direct-I/O alignment requirements. To require
direct storage-to-GPU I/O, set
`VLLM_HIPFILE_MODE=p2p`, `HIPFILE_FORCE_COMPAT_MODE=false`, and
`HIPFILE_ALLOW_COMPAT_MODE=false`, and use files and I/O aligned for the target
filesystem. Typical safetensors payload offsets are not aligned, so strict
mode can reject them. See AMD's [backend requirements](https://rocm.docs.amd.com/projects/hipFile/en/latest/reference/hipFile-io-backends.html).

The loader supports safetensors and tensor parallelism without expert-parallel
filtering. `VLLM_HIPFILE_LOAD` and `VLLM_CUFILE_LOAD` cannot both be enabled.
Unset `VLLM_HIPFILE_LOAD` to use the normal loader.
