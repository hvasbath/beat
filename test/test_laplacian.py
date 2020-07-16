import unittest
import logging
from time import time
from beat.models import laplacian
from matplotlib import pyplot as plt

import numpy as num

from pyrocko import util


logger = logging.getLogger('test_laplacian')


class LaplacianTest(unittest.TestCase):

    def setUp(self):

        self.x = num.arange(0, 5.)
        self.y = num.arange(-5, 2.)
        xs, ys = num.meshgrid(self.x, self.y)
        self.coords = num.vstack((xs.ravel(), ys.ravel())).T
        print(self.coords.shape)

    def test_distances(self):

        dists = laplacian.distances(self.coords, self.coords)
        print(dists)

        plt.matshow(dists)
        plt.show()

    def test_uniform_laplacian(self):
        L = laplacian.get_smoothing_operator_nearest_neighbor(
            len(self.x), len(self.y), 1., 1.)

        print(L.shape)
        print(L)
        im = plt.matshow(L)
        plt.title('Uniform')
        plt.colorbar(im)
        plt.show()

    def test_variable_laplacian(self):
        L_exp = laplacian.get_smoothing_operator_correlated(self.coords, 'exponential')
        L_gauss= laplacian.get_smoothing_operator_correlated(self.coords, 'gaussian')
        print(L_exp.shape)
        print('exp gauss', L_exp, L_gauss)
        im = plt.matshow(L_exp, vmin=L_exp.min(), vmax=L_exp.max())
        plt.colorbar(im)
        plt.title('Exp')
        im2 = plt.matshow(L_gauss, vmin=L_gauss.min(), vmax=L_gauss.max())
        plt.colorbar(im2)
        plt.title('Gauss')
        plt.show()


if __name__ == '__main__':
    util.setup_logging('test_laplacian', 'debug')
    unittest.main()