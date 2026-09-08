# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib
import sys
from types import SimpleNamespace


def test_minimax_m3_msa_warmup_skips_nvidia_import_on_rocm(monkeypatch):
    module_name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
    nvidia_model = "vllm.models.minimax_m3.nvidia.model"
    sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def import_without_nvidia(name, *args, **kwargs):
        if name == nvidia_model:
            raise AssertionError("ROCm warmup must not import the NVIDIA model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_nvidia)
    warmup = importlib.import_module(module_name)
    monkeypatch.setattr(
        warmup,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: False),
    )

    warmup.minimax_m3_msa_warmup(SimpleNamespace())
