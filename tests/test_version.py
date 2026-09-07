# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm import version


def test_version_is_defined():
    assert version.__version__ is not None


def test_version_tuple():
    assert len(version.__version_tuple__) in (3, 4, 5)
