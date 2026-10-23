# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("args", "command", "expected_text", "exit_code"),
    [
        ([], None, "vLLM CLI", 0),
        (["--help"], None, "vLLM CLI", 0),
        (["--help=all"], None, "vLLM CLI", 0),
        (["-v"], None, importlib.metadata.version("vllm"), 0),
        (["--version"], None, importlib.metadata.version("vllm"), 0),
        (["unknown"], None, "invalid choice", 2),
        (["serve", "--help"], "serve", "Frontend", 0),
        (["serve", "--help=ModelConfig"], "serve", "ModelConfig", 0),
        (["serve", "--enforce-eager"], "serve", "unrecognized arguments", 2),
        (["serve", "--enforce_eager"], "serve", "unrecognized arguments", 2),
        (["chat", "--help=all"], "openai", "--quick", 0),
        (["complete", "--help=all"], "openai", "--max-tokens", 0),
        (["bench", "--help"], "benchmark.main", "throughput", 0),
        (["collect-env", "--help"], "collect_env", "collect-env", 0),
        (["download-kernels", "--help=all"], "download_kernels", "--dry-run", 0),
        (["preload", "--help=all"], "preload", "--weight-cache-socket-dir", 0),
        (["run-batch", "--help"], "run_batch", "INPUT.jsonl", 0),
    ],
)
def test_cli_loads_only_selected_command(args, command, expected_text, exit_code):
    """Help and argument errors must not load unrelated command dependencies."""
    script = """
import json
import sys
from vllm.entrypoints.cli.main import main

sys.argv = ["vllm", *sys.argv[1:]]
code = 0
try:
    main()
except SystemExit as exc:
    code = exc.code
modules = ("openai", "serve", "benchmark.main", "collect_env",
           "download_kernels", "preload", "run_batch")
loaded = sorted(module for module in modules
                if f"vllm.entrypoints.cli.{module}" in sys.modules)
print("LOADED_COMMANDS=" + json.dumps(loaded), file=sys.stderr)
sys.exit(code)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_TARGET_DEVICE": "cpu",
        },
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert expected_text in result.stdout + result.stderr
    loaded = json.loads(result.stderr.rsplit("LOADED_COMMANDS=", 1)[1])
    assert loaded == ([] if command is None else [command])
    if not args or args == ["--help"]:
        assert all(
            name in result.stdout
            for name in (
                "chat",
                "complete",
                "serve",
                "bench",
                "collect-env",
                "download-kernels",
                "preload",
                "run-batch",
            )
        )
