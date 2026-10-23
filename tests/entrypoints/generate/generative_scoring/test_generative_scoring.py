# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.entrypoints.generate.generative_scoring.protocol import (
    GenerativeScoringItemResult,
    GenerativeScoringRequest,
    GenerativeScoringResponse,
)


class TestProtocolModels:
    """Tests for GenerativeScoringRequest and GenerativeScoringResponse."""

    def test_request_and_response_all_fields(self):
        """Test request construction with all field types and response structure."""
        # Test request with string inputs
        req_str = GenerativeScoringRequest(
            query="Is this the capital?",
            items=["Paris", "London"],
            label_token_ids=[9454, 2753],
        )
        assert req_str.query == "Is this the capital?"
        assert req_str.items == ["Paris", "London"]
        assert req_str.label_token_ids == [9454, 2753]
        assert req_str.apply_softmax is True  # default
        assert req_str.item_first is False  # default
        assert req_str.add_special_tokens is True  # default
        assert req_str.cache_salt is None

        # Test request with pre-tokenized inputs and custom options
        req_tok = GenerativeScoringRequest(
            query=[100, 200, 300],
            items=[[400, 500], [600, 700]],
            label_token_ids=[1234, 5678],
            apply_softmax=False,
            item_first=True,
            add_special_tokens=False,
        )
        assert req_tok.query == [100, 200, 300]
        assert req_tok.items == [[400, 500], [600, 700]]
        assert req_tok.apply_softmax is False
        assert req_tok.item_first is True
        assert req_tok.add_special_tokens is False

        # Test response structure
        response = GenerativeScoringResponse(
            model="test-model",
            data=[
                GenerativeScoringItemResult(index=0, score=0.7),
                GenerativeScoringItemResult(index=1, score=0.4),
            ],
            usage={"prompt_tokens": 10, "total_tokens": 12, "completion_tokens": 2},
        )
        assert response.object == "list"
        assert response.model == "test-model"
        assert len(response.data) == 2
        assert response.data[0].score == 0.7
        assert response.data[0].object == "score"
        assert response.data[1].score == 0.4
        assert response.usage.prompt_tokens == 10
