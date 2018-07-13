import logging
import unittest
from beat.sources import MTQTSource

from pyrocko import util
import numpy as num
from numpy.testing import assert_allclose


pi = num.pi
logger = logging.getLogger('test_sources')


class TestSources(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_MTSourceQT(self):

        # from Tape & Tape 2015 Appendix A:
        (u, v, kappa, sigma, h) = (
            3. / 8. * pi,
            -1. / 9.,
            4. / 5. * pi,
            -pi / 2.,
            3. / 4.)

        mt = MTQTSource(u=u, v=v, kappa=kappa, sigma=sigma, h=h)

        reference_colatlon = num.array([1.571, -0.113])
        reference_theta = num.array([0.723])
        reference_U = num.array([
            [-0.587, -0.809, 0.037],
            [0.807, -0.588, -0.051],
            [0.063, 0., 0.998]])
        reference_lambda = num.array(
            [0.749, -0.092, -0.656])
        reference_m9_nwu = num.array([
            [0.196, -0.397, -0.052],
            [-0.397, 0.455, 0.071],
            [-0.052, 0.071, -0.651]])

        assert_allclose(
            num.array([mt.beta, mt.gamma]),
            reference_colatlon, atol=1e-3, rtol=0.)
        assert_allclose(mt.theta, reference_theta, atol=1e-3, rtol=0.)
        assert_allclose(mt.rot_U, reference_U, atol=1e-3, rtol=0.)
        assert_allclose(mt.lune_lambda, reference_lambda, atol=1e-3, rtol=0.)
        assert_allclose(mt.m9_nwu, reference_m9_nwu, atol=1e-3, rtol=0.)
        print 'M9 NED', mt.m9
        print 'M9 NWU', mt.m9_nwu


if __name__ == '__main__':

    util.setup_logging('test_sources', 'info')
    unittest.main()
