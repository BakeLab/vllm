# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from typing import Any

from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.decisions.state import init_decisions_state
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.renderers.chat_utils import load_chat_template
from vllm.tasks import SupportedTask

from .serving import ServingStructuredDecisions


def init_structured_decisions_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple[SupportedTask, ...],
    default_chat_template_kwargs: dict[str, Any],
):
    strategy = init_decisions_state(
        engine_client,
        state,
        args,
        request_logger,
        load_chat_template(args.chat_template),
        default_chat_template_kwargs,
        supported_tasks,
    )
    state.serving_structured_decisions = (
        ServingStructuredDecisions(
            state.openai_serving_models, strategy, request_logger=request_logger
        )
        if strategy is not None
        else None
    )
