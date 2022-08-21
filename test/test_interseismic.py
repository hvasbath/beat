import logging
import os
import unittest

import numpy as num
from numpy.testing import assert_allclose
from pyrocko import orthodrome, plot, util

from beat import interseismic, pscmp
from beat.heart import (
    ReferenceLocation,
    velocities_from_pole,
    velocities_from_strain_rate_tensor,
)

km = 1000.0

logger = logging.getLogger("test_interseismic")


class TestInterseismic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.reference = None
        self.amplitude = 0.02
        self.azimuth = 115.0
        self.locking_depth = [6.3, 5.0]

    def _get_store_superdir(self):
        return os.path.abspath("data/")

    def _get_gf_store(self, crust_ind):
        store_superdir = self._get_store_superdir(self)
        return os.path.join(store_superdir, "psgrn_green_%i" % crust_ind)

    def _get_synthetic_data(self):
        lon = num.linspace(10.5, 13.5, 100)
        lat = num.linspace(44.0, 46.0, 100)

        Lon, Lat = num.meshgrid(lon, lat)
        reference = ReferenceLocation(lon=5.0, lat=45.0)

        self.lons = Lon.flatten()
        self.lats = Lat.flatten()
        self.reference = reference

    def _get_sources(self, case=1):
        if case == 1:
            sources = [
                pscmp.PsCmpRectangularSource(
                    lon=12.0, lat=45.0, strike=20.0, dip=90.0, length=125.0 * km
                ),
                pscmp.PsCmpRectangularSource(
                    lon=11.25, lat=44.35, strike=70.0, dip=90.0, length=80.0 * km
                ),
            ]

        elif case == 2:
            sources = [
                pscmp.PsCmpRectangularSource(
                    lon=12.04,
                    lat=45.000,
                    strike=329.35 - 180,
                    dip=90.0,
                    length=117809.04,
                ),
                pscmp.PsCmpRectangularSource(
                    lon=11.5, lat=45.75, strike=357.04 - 180, dip=90.0, length=80210.56
                ),
            ]

        for source in sources:
            north_shift, east_shift = orthodrome.latlon_to_ne_numpy(
                self.reference.lat,
                self.reference.lon,
                source.effective_lat,
                source.effective_lon,
            )
            source.update(
                lat=self.reference.lat,
                lon=self.reference.lon,
                north_shift=north_shift,
                east_shift=east_shift,
            )
            print(source)

        return sources

    def old_test_backslip_params(self):
        azimuth = (90.0, 0.0)
        strike = (0.0, 0.0)
        dip = (90.0, 90.0)
        amplitude = (0.1, 0.1)
        locking_depth = (5000.0, 5000.0)

        test_opening = (-0.1, 0.0)
        test_slip = (0.0, 0.1)
        test_rake = (
            180.0,
            0.0,
        )

        for i, (a, s, d, am, ld) in enumerate(
            zip(azimuth, strike, dip, amplitude, locking_depth)
        ):

            d = interseismic.backslip_params(a, s, d, am, ld)

            num.testing.assert_allclose(
                d["opening"], test_opening[i], rtol=0.0, atol=1e-6
            )
            num.testing.assert_allclose(d["slip"], test_slip[i], rtol=0.0, atol=1e-6)
            num.testing.assert_allclose(d["rake"], test_rake[i], rtol=0.0, atol=1e-6)

    def old_test_block_geometry(self):

        if self.reference is None:
            self._get_synthetic_data()

        return interseismic.block_geometry(
            lons=self.lons,
            lats=self.lats,
            sources=self._get_sources(),
            reference=self.reference,
        )

    def old_test_block_synthetics(self):

        if self.reference is None:
            self._get_synthetic_data()

        return interseismic.geo_block_synthetics(
            lons=self.lons,
            lats=self.lats,
            sources=self._get_sources(),
            amplitude=self.amplitude,
            azimuth=self.azimuth,
            reference=self.reference,
        )

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
            locking_depth=self.locking_depth,
        )

    def _old_test_plot_synthetics(self):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=plot.mpl_papersize("a4", "portrait")
        )

        cmap = plt.cm.jet
        fontsize = 12
        sz = 10.0

        if self.reference is None:
            self._get_synthetic_data()

        #        disp = self.test_block_geometry()
        #        disp = self.test_block_synthetics()
        disp = self._test_backslip_synthetics(2)

        for i, comp in enumerate("NEZ"):
            im = ax[i].scatter(self.lons, self.lats, sz, disp[:, i], cmap=cmap)
            cblabel = "%s displacement [m]" % comp
            cbs = plt.colorbar(im, ax=ax[i], orientation="horizontal", cmap=cmap)
            cbs.set_label(cblabel, fontsize=fontsize)

        plt.show()

    def test_plate_rotation(self):
        v1_ref = num.array([-40.33431624537931, 27.59254158624030, 0.0]) / km
        v2_ref = num.array([35.47707891158412, -27.93047805570016, 0.0]) / km
        v1 = velocities_from_pole(37.0, -123.0, 48.7, -78.2, 0.78).ravel()
        from time import time

        t2 = time()
        v2 = velocities_from_pole(34.75, -116.5, 48.7, -78.2, -0.78).ravel()
        t3 = time()
        assert_allclose(v1, v1_ref, atol=1e-3, rtol=0.0)
        assert_allclose(v2, v2_ref, atol=1e-3, rtol=0.0)

        logger.info("One point %f" % (t3 - t2))
        t0 = time()
        v3 = velocities_from_pole([37.0, 37.1], [-123.0, -125.5], 48.7, -78.2, 0.78)
        t1 = time()

        logger.info("Two points %f" % (t1 - t0))
        assert v3.shape == (2, 3)
        assert_allclose(v3[0, :], v1_ref, atol=1e-3, rtol=0.0)

    def test_velocities_from_strain_rate_tensor(self):

        nanostrain = 1e-9
        lats_vec = num.linspace(37.0, 37.5, 5)
        lons_vec = num.linspace(-122.0, -121.0, 5)
        eps_xx = 0.0  # nanostrain South Bay Block from Jolivet et al. 2015
        eps_yy = 58  # - 115.
        eps_xy = 0  # - 58.
        rotation = 0  # 1009.5  # mm/ ( yr * km)

        lons, lats = num.meshgrid(lons_vec, lats_vec)
        print(lats, lons)

        v_x, v_y, v_z = velocities_from_strain_rate_tensor(
            lats.ravel(), lons.ravel(), eps_xx, eps_yy, eps_xy, rotation
        ).T

        print("vmagn", num.sqrt(v_x**2 + v_y**2))
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 1)
        axs.quiver(lons, lats, v_x, v_y)
        plt.show()
        print(v_z)


if __name__ == "__main__":
    util.setup_logging("test_utility", "info")
    unittest.main()
