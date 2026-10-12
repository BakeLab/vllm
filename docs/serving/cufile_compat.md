# cuFile compatibility loading (BakeLab trim)

This experimental loader reads safetensors directly from an existing model
directory, including a read-only virtiofs mount. It avoids a separate guest-disk
copy. cuFile compatibility mode still uses CPU buffers; it does not provide
storage-to-GPU DMA or eliminate the filesystem page cache.

Set these variables before starting any vLLM workers:

```bash
export BB_CUFILE_LOAD=1
export CUFILE_FORCE_COMPAT_MODE=true
export CUFILE_ALLOW_COMPAT_MODE=true
vllm serve /mnt/vfs/model --load-format safetensors --tensor-parallel-size 4
```

Requires a CUDA PyTorch build providing `torch.cuda.gds.GdsFile` and its matching
cuFile runtime libraries. Unset `BB_CUFILE_LOAD` to restore normal loading.
The compatibility loader rejects expert-parallel filtering. Each tensor-parallel
rank reads the full checkpoint, then vLLM applies its normal weight partitioning.
Tensor payloads are allocated on the current CUDA device one at a time; allow
space for the largest source tensor in addition to the model's resident weights.
This path overrides the normal safetensors load strategy when enabled.

The original sandbox implementation loaded all 40 Qwen3-Coder-Next-FP8 shards
on four RTX PRO 5000 GPUs. Each rank read 80,362,904,064 tensor bytes in about
221 seconds. Those observations apply to the sandbox prototype; the source
migration and added validation have not yet been rerun. No speedup or memory
reduction relative to the default loader has been established.
