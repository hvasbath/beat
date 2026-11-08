"""
Toy models shared by the sampler and backend tests.
"""

import numpy as num
import pymc as pm
import pytensor.tensor as tt

N_PARAMETERS = 4
MU1 = num.ones(N_PARAMETERS) * 0.5
MU2 = -MU1
STDEV = 0.1
WEIGHT1 = STDEV
WEIGHT2 = 1 - STDEV


def two_gaussians(x, n=N_PARAMETERS):
    """
    Log-likelihood of a bimodal mixture of two narrow Gaussians at +-0.5.
    """
    mu1 = num.ones(n) * 0.5
    mu2 = -mu1
    sigma = num.power(STDEV, 2) * num.eye(n)
    isigma = num.linalg.inv(sigma)
    dsigma = num.linalg.det(sigma)

    log_like1 = (
        -0.5 * n * tt.log(2 * num.pi)
        - 0.5 * tt.log(dsigma)
        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
    )
    log_like2 = (
        -0.5 * n * tt.log(2 * num.pi)
        - 0.5 * tt.log(dsigma)
        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
    )
    return tt.log(WEIGHT1 * tt.exp(log_like1) + WEIGHT2 * tt.exp(log_like2))


def two_gaussians_model(likelihood_name="like", potential=False, n=N_PARAMETERS):
    """
    Uniform prior on [-2, 2]^n with the two_gaussians likelihood stored as
    deterministic. If potential is True the deterministic is called "tmp"
    and a potential named likelihood_name is added (as required by PT).
    """
    with pm.Model() as model:
        X = pm.Uniform(
            "X",
            shape=n,
            lower=-2.0 * num.ones(n),
            upper=2.0 * num.ones(n),
            initval=-1.0 * num.ones(n),
            transform=None,
        )
        if potential:
            like = pm.Deterministic("tmp", two_gaussians(X, n))
            pm.Potential(likelihood_name, like)
        else:
            pm.Deterministic(likelihood_name, two_gaussians(X, n))

    return model
