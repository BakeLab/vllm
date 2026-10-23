# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.entrypoints.pooling.embed.protocol import CohereEmbedRequest
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.utils import encode_pooling_output_float_or_ndarray
from vllm.entrypoints.serve.utils.api_utils import create_json_response
from vllm.outputs import PoolingOutput, PoolingRequestOutput

pytestmark = pytest.mark.skip_global_cleanup


def _pooling_output(data):
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


def test_encode_pooling_output_float_or_ndarray_returns_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    encoded = encode_pooling_output_float_or_ndarray(output)

    assert isinstance(encoded, np.ndarray)
    np.testing.assert_allclose(encoded, [1.0, 2.0, 3.0])


def test_json_response_serializes_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    encoded = encode_pooling_output_float_or_ndarray(output)
    response = create_json_response({"embedding": encoded})
    assert json.loads(response.body)["embedding"] == pytest.approx([1.0, 2.0, 3.0])


def test_encode_pooling_output_float_or_ndarray_falls_back_to_list():
    class DataWithUnsupportedNumpy:
        def is_contiguous(self):
            return True

        def numpy(self):
            raise TypeError("unsupported dtype")

        def tolist(self):
            return [1.0, 2.0, 3.0]

    output = _pooling_output(DataWithUnsupportedNumpy())

    assert encode_pooling_output_float_or_ndarray(output) == [1.0, 2.0, 3.0]


@pytest.fixture
def pooling_outputs():
    return [
        PoolingRequestOutput(
            request_id="req",
            outputs=PoolingOutput(data=torch.tensor([1.25, -2.5, 0.0, 3.0])),
            prompt_token_ids=[1, 2],
            num_cached_tokens=0,
            finished=True,
        )
    ]


def test_cohere_json_response_preserves_omitted_fields(pooling_outputs):
    serving = object.__new__(ServingEmbedding)
    ctx = SimpleNamespace(
        request=CohereEmbedRequest(texts=["hello"]),
        final_res_batch=pooling_outputs,
        request_id="req",
    )
    body = json.loads(serving._build_cohere_response_from_ctx(ctx).body)
    assert body["embeddings"] == {"float": [[1.25, -2.5, 0.0, 3.0]]}
    assert body["texts"] == ["hello"]
    assert body["meta"]["billed_units"]["input_tokens"] == 2
