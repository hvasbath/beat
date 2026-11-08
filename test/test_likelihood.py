"""
Likelihoods of the composites compared to an independent numpy evaluation of
the multivariate normal distribution, with the covariance scaled by the
hyperparameter: C_h = exp(2 h) C.
"""

from types import SimpleNamespace

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pytensor import function, shared

from beat.models.distributions import get_hyper_name, multivariate_normal_chol

LOG_2PI = num.log(2 * num.pi)


def normal_loglike(residual, covariance, hyper):
    n = residual.size
    cov = num.exp(2 * hyper) * covariance
    _, logdet = num.linalg.slogdet(cov)
    return -0.5 * (n * LOG_2PI + logdet + residual.dot(num.linalg.solve(cov, residual)))


@pytest.fixture(scope="module", params=["seismic_project", "geodetic_project"])
def setup(request):
    project = request.getfixturevalue(request.param)
    problem = project.load_problem()
    return project, problem


def evaluate(problem, varname, point):
    """Evaluate model variable for point of the random variables."""
    model = problem.model
    (output,) = model.replace_rvs_by_values([model[varname]])
    fn = function(model.value_vars, output, on_unused_input="ignore")
    return fn(*[point[value_var.name] for value_var in model.value_vars])


def composite_likelihoods(problem, point, name):
    composite = problem.composites[name]
    return composite, evaluate(problem, composite._like_name, point)


def numpy_likelihoods(composite, point):
    if composite.name == "geodetic":
        synths = composite.get_synthetics(point)
        residuals = [d.displacement - s for d, s in zip(composite.datasets, synths)]
    else:
        # likelihood is evaluated in the time window between taper b and c
        synths, obs = composite.get_synthetics(
            point, chop_bounds=["b", "c"], force=True
        )
        residuals = [ob.ydata - st.ydata for st, ob in zip(synths, obs)]

    return [
        normal_loglike(
            residual,
            dataset.covariance.c_total,
            float(point[get_hyper_name(dataset)]),
        )
        for residual, dataset in zip(residuals, composite.datasets)
    ]


@pytest.mark.parametrize("hyper_shift", [0.0, 0.7])
def test_composite_likelihood(setup, hyper_shift):
    project, problem = setup
    name = project.datatypes[0]
    point = problem.model.initial_point()
    for hp in problem.hypernames:
        point[hp] = point[hp] + hyper_shift

    composite, model_llks = composite_likelihoods(problem, point, name)

    full_point = dict(project.point)
    full_point.update(point)
    reference = numpy_likelihoods(composite, full_point)

    assert len(model_llks) == len(composite.datasets)
    assert_allclose(model_llks, reference, rtol=1e-6)


def test_total_likelihood(setup):
    project, problem = setup
    point = problem.model.initial_point()
    composite, model_llks = composite_likelihoods(problem, point, project.datatypes[0])
    like = evaluate(problem, "like", point)
    assert_allclose(like, num.sum(model_llks), rtol=1e-10)


def test_multivariate_normal_chol_many_samples():
    """The number of samples must not overflow the normalisation term."""
    nsamples = 40000
    dataset = SimpleNamespace(
        samples=nsamples,
        typ="SAR",
        covariance=SimpleNamespace(slog_pdet=shared(0.0)),
    )
    weights = [shared(num.eye(3))]
    residuals = [num.zeros(3)]
    logpts = multivariate_normal_chol(
        [dataset], weights, {"h_SAR": 0.0}, residuals
    ).eval()
    assert_allclose(logpts, [-0.5 * nsamples * LOG_2PI])
