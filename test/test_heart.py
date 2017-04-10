import numpy as num
from beat import heart
from tempfile import mkdtemp
import shutil
import unittest
from pyrocko import util


class TestUtility(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_backslip(self):
        azimuth = (90., 0.)
        strike = (0., 0.)
        dip = (90., 90.)
        amplitude = (0.1, 0.1)
        locking_depth = (5000., 5000.)

        test_opening = (0.1, 0.)
        test_slip = (0., 0.1)
        test_rake = (180., 0.,)

        for i, (a, s, d, am, ld) in enumerate(
            zip(azimuth, strike, dip, amplitude, locking_depth)):

            d = heart.backslip_params(a, s, d, am, ld)

            num.testing.assert_allclose(
                d['opening'], test_opening[i], rtol=0., atol=1e-6)
            num.testing.assert_allclose(
                d['slip'], test_slip[i], rtol=0., atol=1e-6)
            num.testing.assert_allclose(
                d['rake'], test_rake[i], rtol=0., atol=1e-6)


if __name__ == '__main__':
    util.setup_logging('test_utility', 'warning')
    unittest.main()
