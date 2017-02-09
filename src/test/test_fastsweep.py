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
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.patch_size = 10.0 * km
        self.nuc_x = 8
        self.nuc_y = 3
        self.n_patch_strike = 20
        self.n_patch_dip = 8

    def get_slownesses(self):
        velo1 = num.ones((self.n_patch_dip, self.n_patch_strike / 2))
        velo2 = num.ones((self.n_patch_dip, self.n_patch_strike / 2)) * 3.5
        velocities = num.concatenate((velo1, velo2), axis=1)
        return 1. / velocities

    def _numpy_implementation(self):

        slownesses = self.get_slownesses()

        t0 = time()
        numpy_start_times = fast_sweep.get_rupture_times_numpy(
            slownesses, self.patch_size / km,
            self.n_patch_strike, self.n_patch_dip,
            self.nuc_x, self.nuc_y)
        t1 = time()

        logger.info('done numpy fast_sweeping in %f' % (t1 - t0))
        return numpy_start_times

    def _theano_implementation(self):

        slownesses = tt.dmatrix('slownesses')
        nuc_x = tt.lscalar('nuc_x')
        nuc_y = tt.lscalar('nuc_y')
        patch_size = tt.cast(self.patch_size / km, 'float64')

        theano_start_times = fast_sweep.get_rupture_times_theano(
            slownesses, patch_size, nuc_x, nuc_y)

        Slownesses = self.get_slownesses()

        t0 = time()
        f = function([slownesses, nuc_x, nuc_y],
                     [theano_start_times])
        t1 = time()
        theano_start_times = f(Slownesses, self.nuc_x, self.nuc_y)
        t2 = time()

        logger.info('Theano compile time %f' % (t1 - t0))
        logger.info('done theano fast_sweeping in %f' % (t2 - t1))
        return theano_start_times

#    def _c_implementation(self):
#        slownesses = self.get_slownesses()

#        t0 = time()
#        self.c_start_times = fast_sweep.get_rupture_times_c(
#            slownesses, self.patch_size / km,
#            self.n_patch_strike, self.n_patch_dip,
#            self.nuc_x, self.nuc_y)
#        t1 = time()

#        logger.info('done c fast_sweeping in %f' % (t1 - t0))

#        print 'C', self.c_start_times

    def test_differences(self):
        npi = self._numpy_implementation()
        thi = self._theano_implementation().reshape
        num.testing.assert_allclose(npi, thi, rtol=0., atol=1e-6)

if __name__ == '__main__':
    util.setup_logging('test_fast_sweeping', 'info')
    unittest.main()
