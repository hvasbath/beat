import unittest
import logging
from time import time

import numpy as num

from beat.fast_sweeping import fast_sweep
from pyrocko import util

import theano.tensor as tt
from theano import function

km = 1000.

logger = logging.getLogger('beat')


class FastSweepingTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.Test.__init__(self, *args, **kwargs)
        self.patch_size = 10.0 * km
        self.nuc_x = 8
        self.nuc_y = 3
        self.n_patch_str = 20
        self.n_patch_dip = 8

    def get_slownesses(self):
        velo1 = num.ones((self.n_patch_dip, self.n_patch_str / 2.))
        velo2 = num.ones((self.n_patch_dip, self.n_patch_str / 2.)) * 3.5
        velocities = num.concatenate((velo1, velo2), axis=1)
        return 1. / velocities

    def test_numpy_implementation(self):

        slownesses = self.get_slownesses()

        t0 = time()
        self.numpy_start_times = fast_sweep.get_rupture_times_numpy(
            slownesses, self.patch_size / km,
            self.n_patch_strike, self.n_patch_dip,
            self.nuc_x, self.nuc_y)
        t1 = time()

        logger.info('done numpy fast_sweeping in %f' % (t1 - t0))

    def test_theano_implementation(self):

        slownesses = tt.dmatrix('slownesses')
        nuc_x = tt.lscalar('nuc_x')
        nuc_y = tt.lscalar('nuc_y')
        patch_size = tt.cast(self.patch_size / km, 'float64')

        theano_start_times = fast_sweep.get_rupture_times_theano(
            slownesses, patch_size, self.nuc_x, self.nuc_y)

        Slownesses = self.get_slownesses()

        t0 = time()
        f = function([slownesses, nuc_x, nuc_y], [theano_start_times])
        t1 = time()
        self.theano_start_times = f(Slownesses, self.nuc_x, self.nuc_y)
        t2 = time()

        logger.info('Theano compile time %f' % (t1 - t0))
        logger.info('done theano fast_sweeping in %f' % (t2 - t1))

    def test_c_implementation(self):
        slownesses = self.get_slownesses()

        t0 = time()
        self.c_start_times = fast_sweep.get_rupture_times_c(
            slownesses, self.patch_size / km,
            self.n_patch_strike, self.n_patch_dip,
            self.nuc_x, self.nuc_y)
        t1 = time()

        logger.info('done c fast_sweeping in %f' % (t1 - t0))

        print 'C', self.c_start_times

    def test_differences(self):
        print self.numpy_start_times - self.c_start_times
        print self.numpy_start_times - self.theano_start_times


if __name__ == '__main__'():
    util.setup_logging('test_fast_sweeping', 'warning')
    unittest.main()
