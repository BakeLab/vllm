# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from typing import Any

from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.renderers.chat_utils import load_chat_template
from vllm.tasks import SupportedTask

from .serving import ServingStructuredDecisions
from .strategies import ReadContext, select_read_strategy

logger = init_logger(__name__)


def init_structured_decisions_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple[SupportedTask, ...],
    default_chat_template_kwargs: dict[str, Any],
):
    state.serving_structured_decisions = None
    try:
        strategy_cls = select_read_strategy(engine_client.model_config, supported_tasks)
        strategy = strategy_cls(
            ReadContext(
                engine_client=engine_client,
                online_renderer=state.online_renderer,
                chat_template=load_chat_template(args.chat_template),
                chat_template_content_format=args.chat_template_content_format,
                default_chat_template_kwargs=default_chat_template_kwargs,
            )
        )
    except ValueError as e:
        logger.debug("/v1/systemone is disabled: %s", e)
        return
    state.serving_structured_decisions = ServingStructuredDecisions(
        state.openai_serving_models, strategy, request_logger=request_logger
    )
