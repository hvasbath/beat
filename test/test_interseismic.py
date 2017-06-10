import numpy as num

from beat import interseismic, pscmp
from beat.heart import ReferenceLocation

from tempfile import mkdtemp
import os
import shutil
import unittest

from pyrocko import util
from pyrocko import plot, orthodrome


km = 1000.


class TestInterseismic(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.reference = None
        self.amplitude = 0.02
        self.azimuth = 115.
        self.locking_depth = [6.3, 5.0]

    def _get_store_superdir(self):
        return os.path.abspath('data/')

    def _get_gf_store(self, crust_ind):
        store_superdir = self._get_store_superdir(self)
        return os.path.join(store_superdir, 'psgrn_green_%i' % crust_ind)

    def _get_synthetic_data(self):
        lon = num.linspace(10.5, 13.5, 100.)
        lat = num.linspace(44.0, 46.0, 100.)

        Lon, Lat = num.meshgrid(lon, lat)
        reference = ReferenceLocation(
            lon=5., lat=45.)

        self.lons = Lon.flatten()
        self.lats = Lat.flatten()
        self.reference = reference

    def _get_sources(self, case=1):
        if case == 1:
            sources = [
                pscmp.PsCmpRectangularSource(
                    lon=12., lat=45., strike=20., dip=90., length=125. * km),
                pscmp.PsCmpRectangularSource(
                    lon=11.25, lat=44.35, strike=70., dip=90., length=80. * km)]

        elif case == 2:
            sources = [
                pscmp.PsCmpRectangularSource(
                    lon=12.04, lat=45.000, strike= 329.35 -180, dip=90.,
                    length=117809.04),
                pscmp.PsCmpRectangularSource(
                    lon=11.5, lat=45.75, strike=357.04-180, dip=90.,
                    length=80210.56)]

        for source in sources:
            north_shift, east_shift = orthodrome.latlon_to_ne_numpy(
                self.reference.lat,
                self.reference.lon,
                source.effective_lat,
                source.effective_lon,
                )
            source.update(
                lat=self.reference.lat, lon=self.reference.lon,
                north_shift=north_shift, east_shift=east_shift)
            print source

        return sources

    def test_backslip_params(self):
        azimuth = (90., 0.)
        strike = (0., 0.)
        dip = (90., 90.)
        amplitude = (0.1, 0.1)
        locking_depth = (5000., 5000.)

        test_opening = (-0.1, 0.)
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

    def test_block_geometry(self):

        if self.reference is None:
            self._get_synthetic_data()

        return interseismic.block_geometry(lons=self.lons, lats=self.lats,
            sources=self._get_sources(), reference=self.reference)

    def test_block_synthetics(self):

        if self.reference is None:
            self._get_synthetic_data()

        return interseismic.geo_block_synthetics(
            lons=self.lons, lats=self.lats,
            sources=self._get_sources(),
            amplitude=self.amplitude,
            azimuth=self.azimuth,
            reference=self.reference)

    def _test_backslip_synthetics(self, case=1):

        if self.reference is None:
            self._get_synthetic_data()

        return interseismic.geo_backslip_synthetics(
            store_superdir=self._get_store_superdir(),
            crust_ind=0,
            sources=self._get_sources(case),
            lons=self.lons,
            lats=self.lats,
            reference=self.reference,
            amplitude=self.amplitude,
            azimuth=self.azimuth,
            locking_depth=self.locking_depth)

    def test_plot_synthetics(self):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(
            nrows=1, ncols=3,
            figsize=plot.mpl_papersize('a4', 'portrait'))

        cmap = plt.cm.jet
        fontsize = 12
        sz = 10.

        if self.reference is None:
            self._get_synthetic_data()

#        disp = self.test_block_geometry()
#        disp = self.test_block_synthetics()
        disp = self._test_backslip_synthetics(2)

        for i, comp in enumerate('NEZ'):
            im = ax[i].scatter(self.lons, self.lats, sz, disp[:, i], cmap=cmap)
            cblabel = '%s displacement [m]' % comp
            cbs = plt.colorbar(im, ax=ax[i],
                orientation='horizontal',
                cmap=cmap)
            cbs.set_label(cblabel, fontsize=fontsize)

        plt.show()

if __name__ == '__main__':
    util.setup_logging('test_utility', 'info')
    unittest.main()
