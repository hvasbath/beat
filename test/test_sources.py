import logging
import unittest

import numpy as num
import pyrocko.moment_tensor as mtm
from numpy.testing import assert_allclose
from pyrocko import util

from beat.sources import MTQTSource

pi = num.pi
logger = logging.getLogger("test_sources")


class TestSources(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_MTSourceQT(self):

        # from Tape & Tape 2015 Appendix A:
        (u, v, kappa, sigma, h) = (
            3.0 / 8.0 * pi,
            -1.0 / 9.0,
            4.0 / 5.0 * pi,
            -pi / 2.0,
            3.0 / 4.0,
        )

        magnitude = mtm.moment_to_magnitude(1.0 / num.sqrt(2.0))
        w = 3.0 / 8.0 * pi - u

        mt = MTQTSource(w=w, v=v, kappa=kappa, sigma=sigma, h=h, magnitude=magnitude)

        assert_allclose(mt.rho, 1.0)  # check rho

        reference_colatlon = num.array([1.571, -0.113])
        reference_theta = num.array([0.723])
        reference_U = num.array(
            [[-0.587, -0.809, 0.037], [0.807, -0.588, -0.051], [0.063, 0.0, 0.998]]
        )
        reference_lambda = num.array([0.749, -0.092, -0.656])
        reference_m9_nwu = num.array(
            [[0.196, -0.397, -0.052], [-0.397, 0.455, 0.071], [-0.052, 0.071, -0.651]]
        )
        reference_m9_ned = num.array(
            [[0.196, 0.397, 0.052], [0.397, 0.455, 0.071], [0.052, 0.071, -0.651]]
        )

        assert_allclose(
            num.array([mt.beta, mt.gamma]), reference_colatlon, atol=1e-3, rtol=0.0
        )
        assert_allclose(mt.theta, reference_theta, atol=1e-3, rtol=0.0)
        assert_allclose(mt.rot_U, reference_U, atol=1e-3, rtol=0.0)
        assert_allclose(mt.lune_lambda, reference_lambda, atol=1e-3, rtol=0.0)
        assert_allclose(mt.m9_nwu, reference_m9_nwu, atol=1e-3, rtol=0.0)
        assert_allclose(mt.m9, reference_m9_ned, atol=1e-3, rtol=0.0)
        print("M9 NEED", mt.m9)
        print("M9 NWU", mt.m9_nwu)

    def test_vs_mtpar(self):
        try:
            import mtpar
        except (ImportError):
            logger.warning(
                "This test needs mtpar to be installed: "
                "https://github.com/rmodrak/mtpar/"
            )
            import sys

            sys.exit()

        reference = {
            "magnitude": 4.8,
            "mnn": 0.84551376,
            "mee": -0.75868967,
            "mdd": -0.08682409,
            "mne": 0.51322155,
            "mnd": 0.14554675,
            "med": -0.25767963,
            "east_shift": 10.0,
            "north_shift": 20.0,
            "depth": 8.00,
            "time": -2.7,
            "duration": 5.0,
        }

        m6s = []
        for var in ["mnn", "mee", "mdd", "mne", "mnd", "med"]:
            m6s.append(reference[var])

        m6ned = num.array(m6s)  # mT in NEED
        m6use = mtpar.change_basis(m6ned, i1=2, i2=1)  # rotate to USE

        rho, v, w, kappa_deg, sigma_deg, h = mtpar.cmt2tt15(m6use)

        magnitude = mtm.moment_to_magnitude(rho / num.sqrt(2.0))

        mtqt = MTQTSource(
            w=w,
            v=v,
            kappa=kappa_deg * num.pi / 180.0,
            sigma=sigma_deg * num.pi / 180.0,
            h=h,
            magnitude=magnitude,
        )

        mt_TT = mtpar.tt152cmt(rho, v, w, kappa_deg, sigma_deg, h)  # MT in USE

        # convert from USE to NEED
        mt_TT_ned = mtpar.change_basis(mt_TT, 1, 2)

        print("MTQTSource NEED: \n", mtqt.m6)
        print(mtqt)
        print("TT15, USE: \n", mt_TT)
        print("Input NEED: \n", m6ned)
        print("MTTT15: \n", mt_TT_ned)

        assert_allclose(mt_TT_ned, m6ned, atol=1e-3, rtol=0.0)
        assert_allclose(mtqt.m6, m6ned, atol=1e-3, rtol=0.0)
        assert_allclose(mt_TT_ned, mtqt.m6, atol=1e-3, rtol=0.0)


if __name__ == "__main__":

    util.setup_logging("test_sources", "info")
    unittest.main()
