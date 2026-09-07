# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.utils.import_utils import PlaceholderModule

try:
    from runai_model_streamer import list_safetensors as runai_list_safetensors
except ImportError:
    runai_model_streamer = PlaceholderModule("runai_model_streamer")  # type: ignore[assignment]
    runai_list_safetensors = runai_model_streamer.placeholder_attr("list_safetensors")


def list_safetensors(path: str = "") -> list[str]:
    """
    List full file names from a directory and filter by allow pattern.

    Args:
        path: The directory to list from.

    Returns:
        list[str]: List of full paths allowed by the pattern
    """
    return runai_list_safetensors(path)
