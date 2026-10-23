# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# Process-local states used to pause serving during scaling and after this rank
# has been removed from the external-LB topology. A retired API process remains
# available for health and scaling-status queries until its orchestrator exits it.
_scaling_elastic_ep = False
_elastic_ep_rank_retired = False


def get_scaling_elastic_ep():
    return _scaling_elastic_ep


def set_scaling_elastic_ep(value):
    global _scaling_elastic_ep
    _scaling_elastic_ep = value


def get_elastic_ep_rank_retired():
    return _elastic_ep_rank_retired


def set_elastic_ep_rank_retired(value):
    global _elastic_ep_rank_retired
    _elastic_ep_rank_retired = value
