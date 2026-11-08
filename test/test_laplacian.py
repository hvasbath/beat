import numpy as num
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

from beat.models import laplacian

X = num.arange(0, 5.0)
Y = num.arange(-5, 2.0)


@pytest.fixture
def coords():
    xs, ys = num.meshgrid(X, Y)
    return num.vstack((xs.ravel(), ys.ravel())).T


def test_distances(coords):
    dists = laplacian.distances(coords, coords)

    n = coords.shape[0]
    assert dists.shape == (n, n)
    assert_allclose(num.diag(dists), 0.0)
    assert_allclose(dists, dists.T)
    # neighbours on the grid have unit distance
    assert dists[0, 1] == pytest.approx(1.0)
    plt.matshow(dists)


def test_uniform_laplacian():
    L = laplacian.get_smoothing_operator_nearest_neighbor(len(X), len(Y), 1.0, 1.0)

    assert L.shape == (X.size * Y.size, X.size * Y.size)
    assert_allclose(L.sum(axis=1), 0.0, rtol=0.0, atol=1e-6)
    assert_allclose(L.sum(), 0.0, rtol=0.0, atol=1e-6)
    plt.matshow(L)


@pytest.mark.parametrize("correlation_function", ["exponential", "gaussian"])
def test_variable_laplacian(coords, correlation_function):
    L = laplacian.get_smoothing_operator_correlated(coords, correlation_function)

    assert L.shape == (coords.shape[0], coords.shape[0])
    assert_allclose(L.sum(axis=1), 0.0, rtol=0.0, atol=1e-6)
    assert_allclose(L.sum(), 0.0, rtol=0.0, atol=1e-6)
    plt.matshow(L, vmin=L.min(), vmax=L.max())
