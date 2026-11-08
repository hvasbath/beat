"""
Data covariance estimation (beat.covariance noise analysers) through the
composites of the synthetic projects, for the available noise structures.
"""

import numpy as num
import pytest
from numpy.testing import assert_allclose

from beat import config as bconfig
from beat import covariance as cov


def assert_valid_covariance(matrix):
    assert num.isfinite(matrix).all()
    assert_allclose(matrix, matrix.T, rtol=0, atol=1e-12 * num.abs(matrix).max())
    num.linalg.cholesky(matrix)


def test_noise_structures():
    n, dt, tzero = 20, 0.5, 2.0
    exponential = cov.exponential_data_covariance(n, dt, tzero)
    assert_allclose(num.diag(exponential), 1.0)
    assert_allclose(exponential[0, 1], num.exp(-dt / tzero))
    assert_allclose(exponential[0, 5], num.exp(-5 * dt / tzero))
    assert_valid_covariance(exponential)

    assert_allclose(cov.identity_data_covariance(n), num.eye(n))
    assert set(cov.available_noise_structures()) == {
        "variance",
        "exponential",
        "import",
        "non-toeplitz",
    }
    assert set(cov.available_noise_structures_2d()) == {"import", "non-toeplitz"}


def load_with_structure(project, tmp_path, datatype, structure):
    project = project.copy(tmp_path)
    config = project.config
    config["%s_config" % datatype].noise_estimator.structure = structure
    bconfig.dump_config(config)
    return project, project.load_problem(build=False)


@pytest.mark.parametrize("structure", ["variance", "exponential", "non-toeplitz"])
def test_seismic_noise_structures(seismic_project, tmp_path, structure):
    project, problem = load_with_structure(
        seismic_project, tmp_path, "seismic", structure
    )
    sc = problem.composites["seismic"]
    sc.analyse_noise(project.point)

    sigma = project.noise_sigmas["seismic"]
    for wmap in sc.wavemaps:
        n = wmap.config.arrival_taper.nsamples(1.0 / wmap.deltat)
        for dataset in wmap.datasets:
            matrix = dataset.covariance.data
            assert matrix.shape == (n, n)
            assert_valid_covariance(matrix)
            # (filtered) noise level
            assert 0.01 * sigma**2 < num.median(num.diag(matrix)) < 10 * sigma**2

            if structure == "variance":
                assert_allclose(matrix, num.eye(n) * matrix[0, 0])
            elif structure == "exponential":
                _, fmax = wmap.get_taper_frequencies()
                expected = cov.exponential_data_covariance(n, wmap.deltat, 1.0 / fmax)
                assert_allclose(matrix / matrix[0, 0], expected, rtol=1e-6, atol=1e-12)


def test_seismic_import_structure(seismic_project, tmp_path):
    project, problem = load_with_structure(
        seismic_project, tmp_path, "seismic", "import"
    )
    sc = problem.composites["seismic"]
    wmap = sc.wavemaps[0]
    n = wmap.config.arrival_taper.nsamples(1.0 / wmap.deltat)

    # without imported covariances the identity is used
    for dataset in wmap.datasets:
        dataset.covariance = None
    sc.analyse_noise(project.point)
    assert_allclose(wmap.datasets[0].covariance.data, num.eye(n))

    # covariances with wrong size are replaced by mean variance * identity
    dataset = wmap.datasets[0]
    dataset.covariance.data = num.diag(num.linspace(1.0, 3.0, n + 4))
    sc.analyse_noise(project.point)
    assert_allclose(dataset.covariance.data, num.eye(n) * 2.0)


def test_geodetic_non_toeplitz(geodetic_project, tmp_path):
    project, problem = load_with_structure(
        geodetic_project, tmp_path, "geodetic", "non-toeplitz"
    )
    gc = problem.composites["geodetic"]
    gc.analyse_noise(project.point)

    sigma = project.noise_sigmas["geodetic"]
    for dataset in gc.datasets:
        matrix = dataset.covariance.data
        assert matrix.shape == (dataset.samples, dataset.samples)
        assert_valid_covariance(matrix)
        if dataset.typ == "SAR":
            # estimated from the residuals, i.e. the white noise
            assert 0.3 * sigma**2 < num.median(num.diag(matrix)) < 3 * sigma**2
        else:
            # GNSS keeps the given covariances
            assert_allclose(matrix, num.eye(dataset.samples) * sigma**2)


def test_geodetic_import_requires_covariance(geodetic_project, tmp_path):
    _, problem = load_with_structure(geodetic_project, tmp_path, "geodetic", "import")
    gc = problem.composites["geodetic"]
    gc.datasets[0].covariance.data = None
    with pytest.raises(ValueError):
        gc.analyse_noise()
