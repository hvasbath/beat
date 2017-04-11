import numpy as num

from beat import interseismic, pscmp

from tempfile import mkdtemp
import os
import shutil
import unittest

from pyrocko import util
from pyrocko import plot


km = 1000.


class TestUtility(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def _get_store_superdir(self):
        return os.path.abspath('../data/')

    def _get_gf_store(self, crust_ind):
        store_superdir = self._get_store_superdir(self)
        return os.path.join(store_superdir, 'psgrn_green_%i' % crust_ind)

    def _get_synthetic_data(self):
        lon = num.linspace(10., 15., 100.)
        lat = num.linspace(40., 50., 100.)

        Lon, Lat = num.meshrid(lon, lat)
        reference = interseismic.heart.ReferenceLocation(
            lon=20., lat=55.)

        return Lon.flatten(), Lat.flatten(), reference

    def _get_sources(self):
        sources = [pscmp.PsCmpRectangularSource(
            lon=12., lat=45., strike=120., dip=90., length=100. * km)]
        return sources

    def test_backslip_params(self):
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

            d = interseismic.backslip_params(a, s, d, am, ld)

            num.testing.assert_allclose(
                d['opening'], test_opening[i], rtol=0., atol=1e-6)
            num.testing.assert_allclose(
                d['slip'], test_slip[i], rtol=0., atol=1e-6)
            num.testing.assert_allclose(
                d['rake'], test_rake[i], rtol=0., atol=1e-6)

    def test_backslip_synthetics(self):
        amplitude = 0.1
        azimuth = 45.
        locking_depth = 5.

        lons, lats, reference = self._get_synthetic_data()

        return interseismic.geo_backslip_synthetics(
            store_superdir=self._get_store_superdir(),
            crust_ind=0,
            sources=self._get_sources(),
            lons=lons,
            lats=lats,
            reference=reference,
            amplitude=amplitude, azimuth=azimuth, locking_depth=locking_depth)

    def test_plot_synthetics(self):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(
            nrows=1, ncols=3,
            figsize=plot.mpl_papersize('a4', 'portrait'))

        fontsize = 12
        sz = 15.

        lons, lats, _ = self._get_synthetic_data()
        disp = self.test_backslip_synthetics()

        for i, comp in enumerate('NEZ'):
            im = ax[i].scatter(lons, lats, sz, disp[:, i])
            cblabel = '%s displacement [m]' % comp
            cbs = plt.colorbar(im, ax=ax[comp],
                orientation='horizontal',
                cmap=plt.cm.jet)
            cbs.set_label(cblabel, fontsize=fontsize)


if __name__ == '__main__':
    util.setup_logging('test_utility', 'info')
    unittest.main()
