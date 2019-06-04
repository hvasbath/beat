import unittest
import logging
from time import time
from beat.model import laplacian

import numpy as num

from pyrocko import util


logger = logging.getLogger('test_laplacian')


class LaplacianTest(unittest.TestCase):

    def setUp(self):

        x = num.arange(0, 5.)
        y = num.arange(-5, 2.)
        xs, ys = num.meshgrid(x, y)
        self.coords = num.hstack(xs.ravel(), ys.ravel())

    def test_variable_laplacian(self):
        L = laplacian.get_smoothing_operator_variable(self.coords)

        print(L.shape)
        print(L)


if __name__ == '__main__':
    util.setup_logging('test_laplacian', 'debug')
    unittest.main()