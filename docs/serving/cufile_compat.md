# cuFile weight loading (BakeLab trim)

This experimental loader reads safetensors directly from an existing model
directory, including a read-only virtiofs mount. It avoids a separate guest-disk
copy. cuFile compatibility mode still uses CPU buffers; it does not provide
storage-to-GPU DMA or eliminate the filesystem page cache.

## Compatibility mode

Set these variables before starting any vLLM workers:

```bash
export VLLM_CUFILE_LOAD=1
export VLLM_CUFILE_MODE=compat
export CUFILE_FORCE_COMPAT_MODE=true
export CUFILE_ALLOW_COMPAT_MODE=true
vllm serve /mnt/vfs/model --load-format safetensors --tensor-parallel-size 4
```

Requires a CUDA PyTorch build providing `torch.cuda.gds.GdsFile` and its matching
cuFile runtime libraries. Unset `VLLM_CUFILE_LOAD` to restore normal loading.
`VLLM_CUFILE_MODE` defaults to `compat`; `p2p` selects strict PCI P2PDMA.
Unknown modes and conflicting compatibility environment flags are rejected.
Deployments using the earlier `BB_CUFILE_LOAD` flag must switch to
`VLLM_CUFILE_LOAD`; the old flag no longer selects this loader.
The loader rejects expert-parallel filtering. With `--load-format safetensors`,
each tensor-parallel rank reads the full checkpoint, then vLLM applies its normal
weight partitioning. With host direct I/O, these repeated reads cannot rely on
host page-cache reuse.

For repeated starts, convert once using
`examples/features/sharded_state/save_sharded_state_offline.py` with
`VLLM_CUFILE_LOAD=0` and the intended `--tensor-parallel-size`. Then enable cuFile
and serve the converted directory with `--load-format sharded_state` and the
same tensor-parallel size. Each rank reads only its saved files through cuFile;
replicated parameters remain in each rank's checkpoint. This avoids rereading
the entire original checkpoint per rank. Conversion requires extra disk space
and an initial model load; it does not accelerate unconverted checkpoints.
Compatibility mode coalesces adjacent tensor payloads into reads capped at
64 MiB; a single tensor larger than that is read on its own. Allow room for this
temporary GPU staging buffer in addition to the model's resident weights. P2P
mode keeps per-tensor reads to preserve direct-I/O alignment requirements. This
path overrides the normal safetensors load strategy when enabled.

## Strict PCI P2PDMA mode

Use a supported NVMe path and ext4 in ordered mode, with kernel PCI P2PDMA
support. An ordinary virtiofs model mount does not provide this path. On x86,
NVIDIA documents loading the GPU driver with:

```bash
modprobe nvidia 'NVreg_RegistryDwords=RMForceStaticBar1=1;RmForceDisableIomapWC=1;'
```

Create an application-specific cuFile configuration before starting workers:

```json
{
  "properties": {
    "allow_compat_mode": false,
    "force_compat_mode": false,
    "use_pci_p2pdma": true
  },
  "block": {
    "nvme": {"use_pci_p2pdma": true}
  }
}
```

```bash
export VLLM_CUFILE_LOAD=1
export VLLM_CUFILE_MODE=p2p
export CUFILE_FORCE_COMPAT_MODE=false
export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_ENV_PATH_JSON=/etc/cufile-p2p.json
vllm serve /mnt/nvme/model --load-format safetensors --tensor-parallel-size 4
```

This mode opens checkpoint files with `O_DIRECT` and lets cuFile failures
propagate. The loader does not retry with compatibility mode. Actual path
selection must be verified in cuFile logs; setting a mode is not proof of DMA.
Safetensors offsets can be unaligned, so the aligned-buffer experiment below
does not establish that this loader can load an entire checkpoint in strict mode.

See NVIDIA's [NVMe P2PDMA troubleshooting guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/#nvme-p2pdma-troubleshooting)
for driver, GPU, filesystem, and configuration requirements.

## Evidence and limits

The original sandbox implementation loaded all 40 Qwen3-Coder-Next-FP8 shards
on four RTX PRO 5000 GPUs. Each rank read 80,362,904,064 tensor bytes in about
221 seconds. Those observations apply to the sandbox prototype; full-model
loading with this source has not yet been rerun. A BF16 scale tensor from the
real checkpoint was read in compatibility mode in the running LLM sandbox and
matched safetensors bit for bit. No speedup or memory
reduction relative to the default loader has been established.

A separate aligned 16 MiB cuFileRead experiment passed with compatibility
fallback disabled on QEMU 11.1.1, Linux 7.2.9-bb, an RTX PRO 5000 Blackwell,
NVIDIA driver 615.78.08, and cuFile 1.19.1.55. The file was on a QEMU-emulated
NVMe backed by a host file, using 4 KiB ext4 blocks and explicit `data=ordered`.
The test GPU was inserted with `x-nv-gpudirect-clique=0`, and the driver used the
BAR1 parameters above. The explicit JSON settings were necessary for the
successful run; the environment flag alone did not enable the path.
The returned data matched its SHA-256, and logs reported `p2p mode: 1 compat: 0`.

This establishes the guest cuFile path for that test configuration. It does not
establish physical SSD-to-GPU DMA, lower host memory consumption, or faster LLM
loading. The host QEMU process still read a regular image file. A warmed 16 MiB
test was slower in strict mode than compatibility mode; it is not a checkpoint
loading benchmark. Full-model strict loading and output equivalence remain
unverified.
