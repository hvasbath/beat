import numpy as num
from beat.covariance import non_toeplitz_covariance
from pyrocko import util
from matplotlib import pyplot as plt
import unittest
import logging
from time import time


logger = logging.getLogger('test_covariance')


class TestUtility(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_non_toeplitz(self):

        ws = 500
        a = num.random.normal(scale=2, size=ws)
        cov = non_toeplitz_covariance(a, window_size=ws / 5)
        d = num.diag(cov)

        print d.mean()
        fig, axs = plt.subplots(1, 2)
        im = axs[0].matshow(cov)
        axs[1].plot(d)
        plt.colorbar(im)
        plt.show()


if __name__ == '__main__':
    util.setup_logging('test_covariance', 'warning')
    unittest.main()
