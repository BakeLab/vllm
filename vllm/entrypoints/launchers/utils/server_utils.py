# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from argparse import Namespace

from vllm.config.logging import LogFormatter
from vllm.logger import init_logger

logger = init_logger(__name__)


def load_log_config(log_config_file: str | None) -> dict | None:
    if not log_config_file:
        return None
    try:
        with open(log_config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            "Failed to load log config from file %s: error %s", log_config_file, e
        )
        return None


def get_uvicorn_log_config(args: Namespace) -> dict | None:
    """Get the uvicorn log config based on the provided arguments.

    Priority:
    1. If log_config_file is specified, use it
    2. If disable_access_log_for_endpoints or JSON formatting is specified,
       create a config with the requested filter and format
    3. Otherwise, return None (use uvicorn defaults)
    """
    # First, try to load from file if specified
    logging_config = getattr(args, "logging_config", None)
    log_config_file = (
        logging_config.pylogging_config_file if logging_config is not None else None
    )
    log_config = load_log_config(log_config_file)
    if log_config is not None:
        return log_config

    formatter: LogFormatter = (
        logging_config.formatter if logging_config is not None else "text"
    )
    if args.disable_access_log_for_endpoints or formatter == "json":
        from vllm.logging_utils import create_uvicorn_log_config

        excluded_paths = [
            p.strip()
            for p in (args.disable_access_log_for_endpoints or "").split(",")
            if p.strip()
        ]
        return create_uvicorn_log_config(
            excluded_paths=excluded_paths,
            log_level=args.uvicorn_log_level,
            formatter=formatter,
        )

    return None
