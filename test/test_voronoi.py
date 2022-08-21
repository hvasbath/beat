import logging
import unittest
from time import time

import matplotlib.pyplot as plt
import numpy as num
from pyrocko import util

from beat.utility import get_random_uniform
from beat.voronoi import voronoi

km = 1000.0

logger = logging.getLogger("test_voronoi")


def plot_voronoi_cell_discretization(
    gfs_dip, gfs_strike, voro_dip, voro_strike, gf2voro_idxs
):

    ax = plt.axes()

    ax.plot(gfs_strike, gfs_dip, "xk")

    for i in range(gfs_dip.size):
        v_i = gf2voro_idxs[i]
        ax.plot([gfs_strike[i], voro_strike[v_i]], [gfs_dip[i], voro_dip[v_i]], "-b")

    ax.plot(voro_strike, voro_dip, "or")
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Distance along dip (km)")
    ax.set_aspect("equal", adjustable="box")

    plt.show()


class VoronoiTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.plot = 0
        self.n_gfs = 1000
        self.n_voro = 100

        fault_length = 30.0 * km
        fault_width = 10.0 * km

        lower = 0.0
        upper_strike = fault_length
        upper_dip = fault_width

        self.gf_points_dip = get_random_uniform(lower, upper_dip, dimension=self.n_gfs)
        self.gf_points_strike = get_random_uniform(
            lower, upper_strike, dimension=self.n_gfs
        )

        self.voronoi_points_dip = get_random_uniform(
            lower, upper_dip, dimension=self.n_voro
        )
        self.voronoi_points_strike = get_random_uniform(
            lower, upper_strike, dimension=self.n_voro
        )

    def test_voronoi_discretization(self):

        t0 = time()
        gf2voro_idxs_c = voronoi.get_voronoi_cell_indexes_c(
            self.gf_points_dip,
            self.gf_points_strike,
            self.voronoi_points_dip,
            self.voronoi_points_strike,
        )
        t1 = time()
        logger.info(
            "Discretization with C on %i GFs with %i "
            "voronoi_nodes took: %f" % (self.n_gfs, self.n_voro, (t1 - t0))
        )

        t2 = time()
        gf2voro_idxs_numpy = voronoi.get_voronoi_cell_indexes_numpy(
            self.gf_points_dip,
            self.gf_points_strike,
            self.voronoi_points_dip,
            self.voronoi_points_strike,
        )
        t3 = time()
        logger.info(
            "Discretization with numpy on %i GFs with %i "
            "voronoi_nodes took: %f" % (self.n_gfs, self.n_voro, (t3 - t2))
        )

        num.testing.assert_allclose(
            gf2voro_idxs_c, gf2voro_idxs_numpy, rtol=0.0, atol=1e-6
        )

        if self.plot:
            plot_voronoi_cell_discretization(
                self.gf_points_dip,
                self.gf_points_strike,
                self.voronoi_points_dip,
                self.voronoi_points_strike,
                gf2voro_idxs,
            )


if __name__ == "__main__":
    util.setup_logging("test_voronoi", "info")
    unittest.main()
