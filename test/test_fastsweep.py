import logging
import unittest
from time import time

import numpy as num
import theano.tensor as tt
from pyrocko import util
from theano import function

from beat import theanof
from beat.fast_sweeping import fast_sweep

km = 1000.0

logger = logging.getLogger("test_fastsweep")


class FastSweepingTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.patch_size = 10.0 * km
        self.nuc_x = 2
        self.nuc_y = 3
        self.n_patch_strike = 4
        self.n_patch_dip = 6

    def get_slownesses(self):
        velo1 = num.ones((self.n_patch_dip, self.n_patch_strike // 2))
        velo2 = num.ones((self.n_patch_dip, self.n_patch_strike // 2)) * 3.5
        velocities = num.concatenate((velo1, velo2), axis=1)
        return 1.0 / velocities

    def _numpy_implementation(self):

        slownesses = self.get_slownesses()

        t0 = time()
        numpy_start_times = fast_sweep.get_rupture_times_numpy(
            slownesses,
            self.patch_size / km,
            self.n_patch_strike,
            self.n_patch_dip,
            self.nuc_x,
            self.nuc_y,
        )
        print("np", numpy_start_times)
        t1 = time()

        logger.info("done numpy fast_sweeping in %f" % (t1 - t0))
        return numpy_start_times

    def _theano_implementation(self):

        Slownesses = self.get_slownesses()

        slownesses = tt.dmatrix("slownesses")
        slownesses.tag.test_value = Slownesses

        nuc_x = tt.lscalar("nuc_x")
        nuc_x.tag.test_value = self.nuc_x

        nuc_y = tt.lscalar("nuc_y")
        nuc_y.tag.test_value = self.nuc_y

        patch_size = tt.cast(self.patch_size / km, "float64")

        theano_start_times = fast_sweep.get_rupture_times_theano(
            slownesses, patch_size, nuc_x, nuc_y
        )

        t0 = time()
        f = function([slownesses, nuc_x, nuc_y], theano_start_times)
        t1 = time()
        theano_start_times = f(Slownesses, self.nuc_x, self.nuc_y)
        t2 = time()

        logger.info("Theano compile time %f" % (t1 - t0))
        logger.info("done Theano fast_sweeping in %f" % (t2 - t1))
        return theano_start_times

    def _theano_c_wrapper(self):

        Slownesses = self.get_slownesses()

        slownesses = tt.dvector("slownesses")
        slownesses.tag.test_value = Slownesses.flatten()

        nuc_x = tt.lscalar("nuc_x")
        nuc_x.tag.test_value = self.nuc_x

        nuc_y = tt.lscalar("nuc_y")
        nuc_y.tag.test_value = self.nuc_y

        cleanup = theanof.Sweeper(
            self.patch_size / km, self.n_patch_dip, self.n_patch_strike, "c"
        )

        start_times = cleanup(slownesses, nuc_y, nuc_x)

        t0 = time()
        f = function([slownesses, nuc_y, nuc_x], start_times)
        t1 = time()
        theano_c_wrap_start_times = f(Slownesses.flatten(), self.nuc_y, self.nuc_x)
        print("tc", theano_c_wrap_start_times)
        t2 = time()
        logger.info("Theano C wrapper compile time %f" % (t1 - t0))
        logger.info("done theano C wrapper fast_sweeping in %f" % (t2 - t1))
        print("Theano C wrapper compile time %f" % (t1 - t0))
        return theano_c_wrap_start_times

    def _c_implementation(self):
        slownesses = self.get_slownesses()

        t0 = time()
        c_start_times = fast_sweep.get_rupture_times_c(
            slownesses.flatten(),
            self.patch_size / km,
            self.n_patch_strike,
            self.n_patch_dip,
            self.nuc_x,
            self.nuc_y,
        )
        t1 = time()
        print("c", c_start_times)
        logger.info("done c fast_sweeping in %f" % (t1 - t0))
        return c_start_times

    def test_differences(self):
        np_i = self._numpy_implementation().flatten()
        t_i = self._theano_implementation().flatten()
        c_i = self._c_implementation()
        tc_i = self._theano_c_wrapper()

        num.testing.assert_allclose(np_i, t_i, rtol=0.0, atol=1e-6)
        num.testing.assert_allclose(np_i, c_i, rtol=0.0, atol=1e-6)
        num.testing.assert_allclose(np_i, tc_i, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    util.setup_logging("test_fast_sweeping", "info")
    unittest.main()
