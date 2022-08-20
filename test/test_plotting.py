import logging
import unittest

import numpy as num
from matplotlib import pyplot as plt
from pyrocko import util

from beat.models.distributions import vonmises_std
from beat.plotting import draw_line_on_array, lune_plot, spherical_kde_op

logger = logging.getLogger("test_distributed")


class TestPlotting(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def rtest_draw_line_array(self):

        amplitude = 10
        x = num.arange(0, 100, 0.5)
        y = amplitude * num.sin(x)
        y2 = amplitude / 2.0 * num.cos(x)

        grid, extent = draw_line_on_array(x, y, grid_resolution=(600, 200), linewidth=2)
        print(extent)
        grid, extent = draw_line_on_array(
            x, y2, grid=grid, extent=extent, grid_resolution=(600, 200), linewidth=2
        )

        print(extent)
        ax = plt.axes()
        ax.matshow(grid)

        plt.show()

    def test_spherical_kde_op(self):

        nsamples = 10
        lats0 = num.rad2deg(num.random.normal(loc=0.0, scale=0.1, size=nsamples))
        lons0 = num.rad2deg(num.random.normal(loc=-3.14, scale=0.3, size=nsamples))

        kde, lats, lons = spherical_kde_op(lats0, lons0, grid_size=(200, 200))

        ax = plt.axes()
        im = ax.matshow(kde, extent=(-180, 180, -90, 90), origin="lower")
        plt.colorbar(im)

        plt.show()

    def test_lune_plot(self):

        nsamples = 2100
        # latitude
        w = num.random.normal(loc=0.5, scale=0.1, size=nsamples)
        w_bound = 3.0 * num.pi / 8.0
        w[w > w_bound] = w_bound
        w[w < -w_bound] = -w_bound
        # longitude
        v = num.random.normal(loc=-0.1, scale=0.05, size=nsamples)
        v_bound = 1.0 / 3.0
        v[v > v_bound] = v_bound
        v[v < -v_bound] = -v_bound

        gmt = lune_plot(v_tape=v, w_tape=w)
        gmt.save("lune_test.pdf", resolution=300, size=10)


if __name__ == "__main__":

    util.setup_logging("test_plotting", "info")
    unittest.main()
