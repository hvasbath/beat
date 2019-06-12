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

        x = num.arange(0, 5.)
        y = num.arange(-5, 2.)
        xs, ys = num.meshgrid(x, y)
        self.coords = num.vstack((xs.ravel(), ys.ravel())).T
        print(self.coords.shape)

    def test_distances(self):

        dists = laplacian.distances(self.coords, self.coords)
        print(dists)

        plt.matshow(dists)
        plt.show()

    def test_uniform_laplacian(self):
        L = laplacian.get_smoothing_operator_uniform(8, 5, 3., 3.)

        print(L.shape)
        print(L)
        plt.matshow(L)
        plt.show()

    def test_variable_laplacian(self):
        L = laplacian.get_smoothing_operator_variable(self.coords, 'exponential')

        print(L.shape)
        print(L)
        plt.matshow(L)
        plt.show()


if __name__ == '__main__':
    util.setup_logging('test_laplacian', 'debug')
    unittest.main()