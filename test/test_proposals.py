import numpy as num
import pytest

from beat.sampler import available_proposals, choose_proposal
from beat.sampler.base import multivariate_proposals

N_DRAWS = 10


@pytest.mark.parametrize("proposal", available_proposals())
def test_proposals(proposal):
    if proposal in multivariate_proposals:
        scale = num.eye(2) * 0.5
    else:
        scale = 1

    draw = choose_proposal(proposal, scale=scale)
    samples = num.asarray(draw(N_DRAWS))

    assert N_DRAWS in samples.shape
    assert num.isfinite(samples).all()


def test_unknown_proposal():
    with pytest.raises(Exception):
        choose_proposal("NotAProposal", scale=1)
