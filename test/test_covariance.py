import logging
import unittest
from time import time

import numpy as num
from matplotlib import pyplot as plt
from pyrocko import util

from beat.covariance import non_toeplitz_covariance, non_toeplitz_covariance_2d
from beat.heart import Covariance
from beat.models import load_model
from beat.utility import get_random_uniform


num.random.seed(10)

logger = logging.getLogger("test_covariance")


class TestCovariance(unittest.TestCase):
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

    def test_non_toeplitz_2d(self):

        ws = 500
        data = num.random.normal(scale=2, size=ws)
        coords_x = get_random_uniform(-10000, 10000, dimension=ws)
        coords_y = get_random_uniform(-10000, 10000, dimension=ws)
        coords = num.vstack([coords_x, coords_y]).T
        cov = non_toeplitz_covariance_2d(coords, data, max_dist_perc=0.2)
        d = num.diag(cov)

        print(d.mean())
        _, axs = plt.subplots(1, 2)
        im = axs[0].matshow(cov)
        axs[1].scatter(coords_x, coords_y, 5, d)
        plt.colorbar(im)
        plt.show()

    def test_non_toeplitz_2d_data(self):

        from beat.utility import load_objects
        from beat import config

        home = "/home/vasyurhm/BEATS/PeaceRiver/Alberta2022joint/"
        data = load_objects(home + "geodetic_data.pkl")
        d = data[0]
        c = config.load(filename=home + "config_geometry.yaml")
        d.update_local_coords(c.event)
        coords = num.vstack([d.east_shifts, d.north_shifts]).T
        cov = non_toeplitz_covariance_2d(coords, d.displacement, max_dist_perc=0.2)

        _, axs = plt.subplots(1, 2)
        im = axs[0].matshow(cov)
        axs[1].scatter(coords[:, 0], coords[:, 1], 5, d.displacement)
        plt.colorbar(im)
        plt.show()

    def test_covariance_chol_inverse(self):

        n = 10
        a = num.random.rand(n**2).reshape(n, n)
        C_d = a.T.dot(a) + num.eye(n) * 0.3

        cov = Covariance(data=C_d)
        chol_ur = cov.chol_inverse
        inverse_from_chol_qr = chol_ur.T.dot(chol_ur)

        if 1:
            from matplotlib import pyplot as plt

            fig, axs = plt.subplots(3, 2)
            axs[0, 0].imshow(inverse_from_chol_qr)
            axs[0, 0].set_title("Inverse from QR cholesky")
            axs[0, 1].imshow(cov.inverse)
            axs[0, 1].set_title("Inverse from matrix inversion")

            I_diff = inverse_from_chol_qr - cov.inverse
            print(cov.inverse)
            print("Idiff minmax", I_diff.min(), I_diff.max())
            axs[1, 0].imshow(I_diff)
            axs[1, 0].set_title("Difference")
            # plt.colorbar(im2)

            I_div = num.log(num.abs(inverse_from_chol_qr / cov.inverse))
            print("minmax", I_div.min(), I_div.max())
            axs[1, 1].imshow(I_div)
            axs[1, 1].set_title("Ratio")

            axs[2, 0].imshow(cov.chol)
            axs[2, 0].set_title("Cholesky factor of cov")

            axs[2, 1].imshow(cov.chol_inverse)
            axs[2, 1].set_title("QR Cholesky factor equivalent to chol(C⁻1)")

            plt.show()

        num.testing.assert_allclose(
            inverse_from_chol_qr, cov.inverse, rtol=0.0, atol=1e-6
        )

    def test_linear_velmod_covariance(self):
        print("Warning!: Needs specific project_directory!")
        project_dir = "/home/vasyurhm/BEATS/LaquilaJointPonlyUPDATE_wide_cov"
        problem = load_model(project_dir, mode="ffi", build=False)
        gc = problem.composites["geodetic"]
        point = problem.get_random_point()
        gc.update_weights(point)

        fig, axs = plt.subplots(2, 2)
        for i, ds in enumerate(gc.datasets):
            im1 = axs[i, 1].matshow(ds.covariance.data)
            im2 = axs[i, 0].matshow(ds.covariance.pred_v)
            print("predv mena", ds.covariance.pred_v.mean())
            print("data mena", ds.covariance.data.mean())

        plt.colorbar(im1)

        plt.show()


if __name__ == "__main__":
    util.setup_logging("test_covariance", "info")
    unittest.main()
