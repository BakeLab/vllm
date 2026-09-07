# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import posixpath
import tempfile
from collections.abc import Callable, Generator
from typing import Any
from urllib.parse import urlsplit

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_pp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_safetensors_files_by_weight_name,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.repo_utils import resolve_revision
from vllm.transformers_utils.runai_utils import list_safetensors

logger = init_logger(__name__)


class RunaiModelStreamerLoader(BaseModelLoader):
    """Model loader that can load safetensors files from the local filesystem
    or the Hugging Face Hub.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        self._is_distributed: bool = False
        if load_config.model_loader_extra_config:
            extra_config = load_config.model_loader_extra_config

            allowed_keys = {"distributed", "concurrency", "memory_limit"}
            if unexpected_keys := set(extra_config) - allowed_keys:
                raise ValueError(
                    "Unexpected extra config keys for runai_streamer: "
                    f"{unexpected_keys}"
                )

            if "distributed" in extra_config:
                distributed = extra_config["distributed"]
                if not isinstance(distributed, bool):
                    raise ValueError(f"distributed must be a bool, got {distributed!r}")
                self._is_distributed = distributed

            # Validate every value before mutating os.environ, so a later
            # invalid key cannot leave an earlier one partially applied.
            env_updates: dict[str, str] = {}
            if "concurrency" in extra_config:
                concurrency = extra_config["concurrency"]
                if (
                    isinstance(concurrency, bool)
                    or not isinstance(concurrency, int)
                    or concurrency <= 0
                ):
                    raise ValueError(
                        f"concurrency must be a positive integer, got {concurrency!r}"
                    )
                env_updates["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)

            if "memory_limit" in extra_config:
                memory_limit = extra_config["memory_limit"]
                if (
                    isinstance(memory_limit, bool)
                    or not isinstance(memory_limit, int)
                    or memory_limit < -1
                ):
                    raise ValueError(
                        f"memory_limit must be an integer >= -1, got {memory_limit!r}"
                    )
                env_updates["RUNAI_STREAMER_MEMORY_LIMIT"] = str(memory_limit)
            os.environ.update(env_updates)

    def _prepare_weights(
        self, model_name_or_path: str, revision: str | None
    ) -> list[str]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        is_local = os.path.isdir(model_name_or_path)
        safetensors_pattern = "*.safetensors"
        index_file = SAFE_WEIGHTS_INDEX_NAME

        if not is_local:
            # `model_weights` can point to another repo than the one `revision` was
            # resolved for, which does not pin this one.
            revision = resolve_revision(model_name_or_path, revision)

        hf_folder = (
            model_name_or_path
            if is_local
            else download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                [safetensors_pattern],
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        )
        hf_weights_files = list_safetensors(path=hf_folder)

        if not is_local:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                cache_dir=self.load_config.download_dir,
                revision=revision,
            )

        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights with `{model_name_or_path}`"
            )

        return hf_weights_files

    def _get_weights_iterator(
        self,
        model_or_path: str,
        revision: str | None,
        is_unused_weight: Callable[[str], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_weights_files = self._prepare_weights(model_or_path, revision)
        selected_files = hf_weights_files
        distributed = self._is_distributed and torch.distributed.is_initialized()
        # Drafts load only on the last PP stage; RunAI is invoked without a
        # model-specific group, so retain its existing full-list path with PP.
        can_select = not distributed or get_pp_group().world_size == 1
        if is_unused_weight is not None and can_select:
            if is_runai_obj_uri(model_or_path):
                selected_files = _filter_remote_weights(
                    hf_weights_files, model_or_path, is_unused_weight
                )
            else:
                selected_files = filter_safetensors_files_by_weight_name(
                    hf_weights_files, is_unused_weight
                )
        if distributed and can_select:
            # RunAI broadcasts across node-local or WORLD groups, not just TP.
            # A rank that cannot filter safely keeps the group on the full list.
            group = get_world_group()
            selections: list[Any] = [None] * group.world_size
            torch.distributed.all_gather_object(
                selections, (hf_weights_files, selected_files), group=group.cpu_group
            )
            if all(files == hf_weights_files for files, _ in selections):
                needed = {path for _, selected in selections for path in selected}
                selected_files = [path for path in hf_weights_files if path in needed]
            else:
                # Do not substitute another node's local checkpoint paths.
                selected_files = hf_weights_files
        return runai_safetensors_weights_iterator(
            selected_files, self.load_config.use_tqdm_on_load, self._is_distributed
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model if necessary."""
        self._prepare_weights(model_config.model, model_config.revision)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        yield from self._get_weights_iterator(
            model_weights,
            model_config.revision,
            getattr(model, "is_unused_checkpoint_weight", None),
        )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model."""
        model.load_weights(self.get_all_weights(model_config, model))
