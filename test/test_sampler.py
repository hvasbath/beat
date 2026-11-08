import numpy as num
import pytest

from beat.sampler import base

NSAMPLES = 100000


def _iqr(x):
    q25, q75 = num.percentile(x, [25, 75])
    return q75 - q25


def test_normal_proposal():
    draws = num.asarray(base.NormalProposal(num.array([2.0]))(NSAMPLES)).ravel()
    assert draws.size == NSAMPLES
    assert draws.mean() == pytest.approx(0.0, abs=0.05)
    assert draws.std() == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize(
    "proposal",
    [
        base.CauchyProposal(num.array([1.0])),
        base.MultivariateCauchyProposal(num.array([[1.0]])),
    ],
    ids=["cauchy", "multivariate_cauchy"],
)
def test_cauchy_proposals(proposal):
    draws = num.asarray(proposal(NSAMPLES)).ravel()
    assert draws.size == NSAMPLES
    # standard Cauchy: median 0, quartiles at -1 and +1
    assert num.median(draws) == pytest.approx(0.0, abs=0.03)
    assert _iqr(draws) == pytest.approx(2.0, rel=0.05)
