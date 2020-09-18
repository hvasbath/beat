import logging
import unittest
from beat.sources import MTQTSource

from pyrocko import util
import pyrocko.moment_tensor as mtm

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

        magnitude = mtm.moment_to_magnitude(1. / num.sqrt(2.))
        w = 3. / 8. * pi - u

        mt = MTQTSource(
            w=w, v=v, kappa=kappa, sigma=sigma, h=h,
            magnitude=magnitude)

        assert_allclose(mt.rho, 1.)  # check rho

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
        reference_m9_ned = num.array([
            [0.196, 0.397, 0.052],
            [0.397, 0.455, 0.071],
            [0.052, 0.071, -0.651]])

        assert_allclose(
            num.array([mt.beta, mt.gamma]),
            reference_colatlon, atol=1e-3, rtol=0.)
        assert_allclose(mt.theta, reference_theta, atol=1e-3, rtol=0.)
        assert_allclose(mt.rot_U, reference_U, atol=1e-3, rtol=0.)
        assert_allclose(mt.lune_lambda, reference_lambda, atol=1e-3, rtol=0.)
        assert_allclose(mt.m9_nwu, reference_m9_nwu, atol=1e-3, rtol=0.)
        assert_allclose(mt.m9, reference_m9_ned, atol=1e-3, rtol=0.)
        print('M9 NED', mt.m9)
        print('M9 NWU', mt.m9_nwu)

    def test_vs_mtpar(self):
        try:
            import mtpar
        except(ImportError):
            logger.warning(
                'This test needs mtpar to be installed: '
                'https://github.com/rmodrak/mtpar/')
            import sys
            sys.exit()

        reference = {
            'magnitude': 4.8,
            'mnn': 0.84551376,
            'mee': -0.75868967,
            'mdd': -0.08682409,
            'mne': 0.51322155,
            'mnd': 0.14554675,
            'med': -0.25767963,
            'east_shift': 10.,
            'north_shift': 20.,
            'depth': 8.00,
            'time': -2.7,
            'duration': 5.,
        }

        m6s = []
        for var in ['mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']:
            m6s.append(reference[var])

        m6ned = num.array(m6s)     # mT in NED
        m6use = mtpar.change_basis(m6ned, i1=2, i2=1)  # rotate to USE

        rho, v, w, kappa_deg, sigma_deg, h = mtpar.cmt2tt15(m6use)

        magnitude = mtm.moment_to_magnitude(rho / num.sqrt(2.))

        mtqt = MTQTSource(
            w=w, v=v, kappa=kappa_deg * num.pi / 180.,
            sigma=sigma_deg * num.pi / 180., h=h, magnitude=magnitude)

        mt_TT = mtpar.tt152cmt(rho, v, w, kappa_deg, sigma_deg, h)  # MT in USE

        # convert from USE to NED
        mt_TT_ned = mtpar.change_basis(mt_TT, 1, 2)

        print('MTQTSource NED: \n', mtqt.m6)
        print(mtqt)
        print('TT15, USE: \n', mt_TT)
        print('Input NED: \n', m6ned)
        print('MTTT15: \n', mt_TT_ned)

        assert_allclose(mt_TT_ned, m6ned, atol=1e-3, rtol=0.)
        assert_allclose(mtqt.m6, m6ned, atol=1e-3, rtol=0.)
        assert_allclose(mt_TT_ned, mtqt.m6, atol=1e-3, rtol=0.)


if __name__ == '__main__':

    util.setup_logging('test_sources', 'info')
    unittest.main()
