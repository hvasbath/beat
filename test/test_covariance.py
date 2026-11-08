import numpy as num
import pytest
from conftest import require_external
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

from beat.covariance import non_toeplitz_covariance, non_toeplitz_covariance_2d
from beat.heart import Covariance
from beat.utility import get_random_uniform, load_objects


def assert_valid_covariance(cov, n):
    assert cov.shape == (n, n)
    assert num.isfinite(cov).all()
    assert_allclose(cov, cov.T, rtol=0, atol=1e-10 * num.abs(cov).max())
    assert (num.diag(cov) > 0).all()
    eigvals = num.linalg.eigvalsh(cov)
    assert eigvals.min() > -1e-8 * eigvals.max()


def test_non_toeplitz():
    ws = 500
    a = num.random.normal(scale=2, size=ws)
    cov = non_toeplitz_covariance(a, window_size=int(ws / 5))
    assert_valid_covariance(cov, ws)

    fig, axs = plt.subplots(1, 2)
    im = axs[0].matshow(cov)
    axs[1].plot(num.diag(cov))
    plt.colorbar(im)


def test_non_toeplitz_2d():
    ws = 500
    data = num.random.normal(scale=2, size=ws)
    coords_x = get_random_uniform(-10000, 10000, dimension=ws)
    coords_y = get_random_uniform(-10000, 10000, dimension=ws)
    coords = num.vstack([coords_x, coords_y]).T
    cov = non_toeplitz_covariance_2d(coords, data, max_dist_perc=0.2)
    assert_valid_covariance(cov, ws)

    _, axs = plt.subplots(1, 2)
    im = axs[0].matshow(cov)
    axs[1].scatter(coords_x, coords_y, 5, num.diag(cov))
    plt.colorbar(im)


def test_non_toeplitz_2d_synthetic_sar(geodetic_project):
    from beat import config

    datasets = load_objects(
        f"{geodetic_project.project_dir}/{config.geodetic_data_name}"
    )
    d = datasets[0]
    assert d.typ == "SAR"
    d.update_local_coords(geodetic_project.config.event)
    coords = num.vstack([d.east_shifts, d.north_shifts]).T
    cov = non_toeplitz_covariance_2d(coords, d.displacement, max_dist_perc=0.2)
    assert_valid_covariance(cov, d.displacement.size)


def test_non_toeplitz_2d_data():
    from beat import config

    home = require_external("BEATS", "PeaceRiver", "Alberta2022joint")
    data = load_objects(str(home / "geodetic_data.pkl"))
    d = data[0]
    c = config.load(filename=str(home / "config_geometry.yaml"))
    d.update_local_coords(c.event)
    coords = num.vstack([d.east_shifts, d.north_shifts]).T
    cov = non_toeplitz_covariance_2d(coords, d.displacement, max_dist_perc=0.2)
    assert_valid_covariance(cov, d.displacement.size)


def test_covariance_chol_inverse():
    n = 10
    a = num.random.rand(n**2).reshape(n, n)
    C_d = a.T.dot(a) + num.eye(n) * 0.3

    cov = Covariance(data=C_d)
    chol_ur = cov.chol_inverse
    inverse_from_chol_qr = chol_ur.T.dot(chol_ur)

    assert_allclose(inverse_from_chol_qr, cov.inverse(), rtol=0.0, atol=1e-6)
    assert_allclose(inverse_from_chol_qr, num.linalg.inv(C_d), rtol=0.0, atol=1e-6)
    assert_allclose(cov.chol().dot(cov.chol().T), C_d, rtol=0.0, atol=1e-10)


@pytest.fixture(scope="module")
def ffi_velmod_project(geodetic_project, tmp_path_factory):
    """
    Geodetic finite fault project with linear GFs for 7 velocity models,
    velocity model prediction covariances are calculated for more than 5.
    """
    from synthetic import make_ffi_project

    return make_ffi_project(
        geodetic_project, tmp_path_factory.mktemp("ffi_velmod"), n_variations=7
    )


def test_linear_velmod_covariance(ffi_velmod_project):
    problem = ffi_velmod_project.load_problem(build=False)
    gc = problem.composites["geodetic"]
    point = problem.get_random_point()

    for ds in gc.datasets:
        assert ds.covariance.pred_v is None or not ds.covariance.pred_v.any()

    gc.update_weights(point)

    for ds, weight in zip(gc.datasets, gc.weights):
        pred_v = ds.covariance.pred_v
        assert pred_v.shape == ds.covariance.data.shape
        assert_valid_covariance(pred_v + ds.covariance.data, ds.samples)
        assert num.abs(pred_v).max() > 0
        assert_allclose(weight.get_value(), ds.covariance.chol_inverse)
