# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import socket
from collections.abc import Generator

import uvicorn

from vllm.entrypoints.common.cli_args import (
    init_parser_plugin,
    validate_api_server_args,
)
from vllm.entrypoints.common.startup import (
    log_non_default_args,
    log_version_and_model,
)
from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class NoSignalServer(uvicorn.Server):
    """Leave SIGINT/SIGTERM handling to the data-parallel supervisor."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


def create_server_socket(
    addr: tuple[str, int],
    *,
    reuse_port: bool,
) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if reuse_port:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def create_server_unix_socket(path: str) -> socket.socket:
    sock = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
    sock.bind(path)
    return sock


@instrument(span_name="API server setup")
def setup_server(args, *, reuse_port: bool):
    """Validate API server args and create the server socket."""
    log_version_and_model(logger, VLLM_VERSION, args.model)
    log_non_default_args(args)
    init_parser_plugin(args)
    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr, reuse_port=reuse_port)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    if args.uds:
        listen_address = f"unix:{args.uds}"
    else:
        addr, port = sock_addr
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock
