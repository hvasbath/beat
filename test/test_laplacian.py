import logging
import unittest
from time import time

import numpy as num
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from pyrocko import util

from beat.models import laplacian

logger = logging.getLogger("test_laplacian")


class LaplacianTest(unittest.TestCase):
    def setUp(self):

        self.x = num.arange(0, 5.0)
        self.y = num.arange(-5, 2.0)
        xs, ys = num.meshgrid(self.x, self.y)
        self.coords = num.vstack((xs.ravel(), ys.ravel())).T
        print(self.coords.shape)

    def test_distances(self):

        dists = laplacian.distances(self.coords, self.coords)

        plt.matshow(dists)
        plt.show()

    def test_uniform_laplacian(self):
        L = laplacian.get_smoothing_operator_nearest_neighbor(
            len(self.x), len(self.y), 1.0, 1.0
        )

        im = plt.matshow(L)
        plt.title("Uniform")
        plt.colorbar(im)
        plt.show()
        assert_allclose(L.sum(), 0.0, rtol=0.0, atol=1e-6)

    def test_variable_laplacian(self):
        L_exp = laplacian.get_smoothing_operator_correlated(self.coords, "exponential")
        L_gauss = laplacian.get_smoothing_operator_correlated(self.coords, "gaussian")

        im = plt.matshow(L_exp, vmin=L_exp.min(), vmax=L_exp.max())
        plt.colorbar(im)
        plt.title("Exp")
        im2 = plt.matshow(L_gauss, vmin=L_gauss.min(), vmax=L_gauss.max())
        plt.colorbar(im2)
        plt.title("Gauss")
        plt.show()

        assert_allclose(L_exp.sum(), 0.0, rtol=0.0, atol=1e-6)
        assert_allclose(L_gauss.sum(), 0.0, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    util.setup_logging("test_laplacian", "debug")
    unittest.main()
