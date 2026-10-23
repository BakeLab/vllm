# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
from pydantic import ValidationError

from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateLogProb


def test_logprob_is_required_on_the_wire():
    """A payload missing logprob is rejected rather than read as -9999."""
    with pytest.raises(ValidationError):
        GenerateLogProb.model_validate({"token_id": 1})
