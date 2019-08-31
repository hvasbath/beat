import numpy as num
from beat.covariance import non_toeplitz_covariance
from pyrocko import util
from matplotlib import pyplot as plt
import unittest
import logging
from beat.models import load_model
from time import time


logger = logging.getLogger('test_covariance')


class TestUtility(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_non_toeplitz(self):

        ws = 500
        a = num.random.normal(scale=2, size=ws)
        cov = non_toeplitz_covariance(a, window_size=int(ws / 5))
        d = num.diag(cov)

        print(d.mean())
        fig, axs = plt.subplots(1, 2)
        im = axs[0].matshow(cov)
        axs[1].plot(d)
        plt.colorbar(im)
        plt.show()

    def test_linear_velmod_covariance(self):
        print('Warning!: Needs specific project_directory!')
        project_dir ='/home/vasyurhm/BEATS/LaquilaJointPonlyUPDATE_wide_cov'
        problem = load_model(project_dir, mode='ffi', build=False)
        gc = problem.composites['geodetic']
        point = problem.get_random_point()
        gc.update_weights(point)

        fig, axs = plt.subplots(2, 2)
        for i, ds in enumerate(gc.datasets):
            im1 = axs[i, 1].matshow(ds.covariance.data)
            im2 = axs[i, 0].matshow(ds.covariance.pred_v)
            print('predv mena', ds.covariance.pred_v.mean())
            print('data mena', ds.covariance.data.mean())

        plt.colorbar(im1)

        plt.show()

if __name__ == '__main__':
    util.setup_logging('test_covariance', 'info')
    unittest.main()
