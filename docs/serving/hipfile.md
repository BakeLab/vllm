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
through host memory. Reads are deferred until the weight loader consumes a
tensor view, allowing standard TP slicing to select the rank's payload first.
The selected view is read in chunks capped at 64 MiB, with bounded GPU staging
and CPU assembly when device memory is insufficient. Replicated weights are
read by each rank that needs them. Strided views can require many reads;
reduced byte counts do not guarantee a proportional load-time speedup.
The existing direct destination and aligned P2P paths are preserved.
Neither mode uses mmap. To require
direct storage-to-GPU I/O, set
`VLLM_HIPFILE_MODE=p2p`, `HIPFILE_FORCE_COMPAT_MODE=false`, and
`HIPFILE_ALLOW_COMPAT_MODE=false`, and use files and I/O aligned for the target
filesystem. Typical safetensors payload offsets are not aligned, so strict
mode can reject them. See AMD's [backend requirements](https://rocm.docs.amd.com/projects/hipFile/en/latest/reference/hipFile-io-backends.html).

The loader supports safetensors and tensor parallelism without expert-parallel
filtering. `VLLM_HIPFILE_LOAD` and `VLLM_CUFILE_LOAD` cannot both be enabled.
Unset `VLLM_HIPFILE_LOAD` to use the normal loader.
