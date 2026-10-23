# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from typing import TYPE_CHECKING

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

if TYPE_CHECKING:
    from starlette.datastructures import State

    from vllm.entrypoints.anthropic.protocol import (
        AnthropicDisabledThinkingEffortOption,
    )
    from vllm.entrypoints.anthropic.serving import DisabledThinkingEffortResolver
    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
else:
    RequestLogger = object

logger = init_logger(__name__)


def init_render_state(
    state: State,
    request_logger: RequestLogger | None,
    *,
    disabled_thinking_effort: "AnthropicDisabledThinkingEffortOption" = "auto",
    disabled_thinking_effort_resolver: "DisabledThinkingEffortResolver | None" = None,
):
    from .derender.serving import ServingDerender
    from .render.serving import ServingRender

    state.serving_render = ServingRender(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        tool_server=state.tool_server,
        disabled_thinking_effort=disabled_thinking_effort,
        disabled_thinking_effort_resolver=disabled_thinking_effort_resolver,
    )

    state.serving_derender = ServingDerender(
        state.openai_serving_models,
        state.online_derenderer,
        request_logger=request_logger,
    )


def init_scale_out_state(
    state: State,
    args: Namespace,
    engine_client: EngineClient,
    request_logger: RequestLogger | None,
):
    init_render_state(
        state,
        request_logger,
        disabled_thinking_effort_resolver=(
            state.anthropic_serving_messages.disabled_thinking_effort_resolver
        ),
    )

    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

    state.serving_tokens = ServingTokens(
        engine_client,
        state.openai_serving_models,
        state.online_renderer,
        online_derenderer=state.online_derenderer,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_log_outputs=args.enable_log_outputs,
        force_no_detokenize=args.tokens_only,
    )
