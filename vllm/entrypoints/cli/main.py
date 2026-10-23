# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM.

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import configure_logging_from_args, init_logger

logger = init_logger(__name__)

COMMANDS = {
    "chat": ("openai", "Generate chat completions via the running API server."),
    "complete": (
        "openai",
        "Generate text completions based on the given prompt "
        "via the running API server.",
    ),
    "serve": (
        "serve",
        "Launch a local OpenAI-compatible API server to serve LLM "
        "completions via HTTP.",
    ),
    "bench": ("benchmark.main", "vLLM bench subcommand."),
    "collect-env": ("collect_env", "Start collecting environment information."),
    "download-kernels": (
        "download_kernels",
        "Install precompiled kernels for faster startup.",
    ),
    "preload": (
        "preload",
        "Launch weight cache daemons (one per GPU) for fast engine restarts.",
    ),
    "run-batch": ("run_batch", "Run batch prompts and write results to file."),
}


def maybe_run_omni() -> bool:
    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" not in sys.argv:
        return False

    # NOTE: Check the spec instead of importing directly here, since things could
    # fail with ImportError due to mismatched versions if things are moved around.
    spec = find_spec("vllm_omni")
    if spec is None:
        logger.error(
            "--omni flag requires a valid instance of vllm-omni to be installed."
        )
        sys.exit(1)

    from vllm_omni.entrypoints.cli.main import main as omni_main

    logger.info("Delegating entrypoint handling to vllm-omni")
    omni_main()

    return True


def main():
    if sys.argv[1:2] == ["launch"]:
        import os

        import vllm.envs as envs

        binary = envs.VLLM_RUST_FRONTEND_PATH
        os.execv(binary, [binary, *sys.argv[2:]])
        return

    if maybe_run_omni():
        return

    from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    def make_parser(command=None):
        parser = FlexibleArgumentParser(
            description="vLLM CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm"),
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        for name, (_, help_text) in COMMANDS.items():
            if command is not None and name == command.name:
                command.subparser_init(subparsers).set_defaults(
                    dispatch_function=command.cmd
                )
            else:
                subparsers.add_parser(name, help=help_text, add_help=False)
        return parser

    parser = make_parser()
    selected, _ = parser.parse_known_args(
        [arg for arg in sys.argv[1:] if not arg.startswith("--help=")]
    )
    if selected.subparser is None:
        parser = make_parser()
        parser.parse_args()
        parser.print_help()
        return

    module_name, _ = COMMANDS[selected.subparser]
    cmd_module = importlib.import_module(f"vllm.entrypoints.cli.{module_name}")
    cmd = next(cmd for cmd in cmd_module.cmd_init() if cmd.name == selected.subparser)

    cli_env_setup()

    if cmd.name == "bench":
        cmd_module.maybe_exec_rust_bench()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from vllm import platforms

        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = make_parser(cmd)
    args = parser.parse_args()
    configure_logging_from_args(args)
    cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
