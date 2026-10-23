# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.plugins.io_processors import has_io_processor
from vllm.renderers import BaseRenderer
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.tasks import SCORE_TYPE_MAP, SupportedTask

from .base.io_processor import PoolingIOProcessor
from .utils import enable_scoring_api

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.serve.utils.request_logger import RequestLogger

else:
    RequestLogger = object


logger = init_logger(__name__)


def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    vllm_config: VllmConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]:
    model_config = vllm_config.model_config
    processors: dict[str, type[PoolingIOProcessor]] = {}
    pooling_task = model_config.get_pooling_task(supported_tasks)

    if pooling_task == "classify":
        from .classify.io_processor import ClassifyIOProcessor

        processors["classify"] = ClassifyIOProcessor

    if pooling_task == "token_classify":
        from .classify.io_processor import TokenClassifyIOProcessor

        processors["token_classify"] = TokenClassifyIOProcessor

    if pooling_task == "embed":
        from .embed.io_processor import EmbedIOProcessor

        processors["embed"] = EmbedIOProcessor

    if pooling_task == "token_embed":
        from .embed.io_processor import TokenEmbedIOProcessor

        processors["token_embed"] = TokenEmbedIOProcessor

    if pooling_task == "embed&token_classify":
        from .pooling.io_processor import UnsupportedCombinedTaskIOProcessor

        processors[pooling_task] = UnsupportedCombinedTaskIOProcessor

    has_plugin = has_io_processor(
        vllm_config,
        model_config.io_processor_plugin,
    )
    if has_plugin:
        from .pooling.io_processor import PluginWithIOProcessorPlugins

        processors["plugin"] = PluginWithIOProcessorPlugins
    elif pooling_task == "plugin":
        from .pooling.io_processor import PluginWithoutIOProcessorPlugins

        processors["plugin"] = PluginWithoutIOProcessorPlugins

    if enable_scoring_api(supported_tasks, model_config):
        from .scoring.io_processor import ScoringIOProcessors

        score_type: str | None = SCORE_TYPE_MAP.get(pooling_task, None)  # type: ignore[arg-type]
        if score_type is not None and score_type in ScoringIOProcessors:
            processors[score_type] = ScoringIOProcessors[score_type]

    if model_config.architecture == "JinaForRanking":
        from .embed.io_processor import JinaRankingTokenEmbedIOProcessor
        from .scoring.io_processor import ScoringIOProcessors

        processors["token_embed"] = JinaRankingTokenEmbedIOProcessor
        processors["late-interaction"] = ScoringIOProcessors["jina-reranking-scoring"]

    return {
        task: processor_cls(
            vllm_config=vllm_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        for task, processor_cls in processors.items()
    }


def init_pooling_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple[SupportedTask, ...],
):
    model_config = engine_client.model_config
    if model_config is None:
        return

    from vllm.renderers.chat_utils import load_chat_template

    from .embed.serving import ServingEmbedding
    from .scoring.serving import ServingScores

    resolved_chat_template = load_chat_template(args.chat_template)
    pooling_task = model_config.get_pooling_task(supported_tasks)

    chat_template_config = ChatTemplateConfig(
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        trust_request_mm_kwargs=args.trust_request_mm_kwargs,
    )

    state.serving_embedding = (
        ServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if pooling_task == "embed"
        else None
    )
    state.serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            supported_tasks=supported_tasks,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
            enable_flash_late_interaction=getattr(
                args, "enable_flash_late_interaction", True
            ),
        )
        if enable_scoring_api(supported_tasks, model_config)
        else None
    )
