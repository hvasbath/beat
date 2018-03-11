import unittest
import logging
from time import time
from beat import ffi
from beat.heart import DynamicTarget

import numpy as num

from pyrocko import util, gf, model

import theano.tensor as tt
from theano import function

km = 1000.

logger = logging.getLogger('test_ffi')


class FFITest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        nsamples = 10
        ntargets = 3
        npatches = 4
        nrisetimes = 5
        nstarttimes = 5
        nsamples = 20

        lats = [10., 35., -34.]
        lons = [190., 260., 10.]

        stations = [model.Station(
            lat=lat, lon=lon) for lat, lon in zip(lats, lons)]

        targets = [DynamicTarget(store_id='Test') for i in range(ntargets)]

        self.gfs = ffi.SeismicGFLibrary(
            targets=targets, stations=stations, component='uperp')
        self.gfs.setup(ntargets, npatches, nrisetimes, nstarttimes, nsamples)

        tracedata = num.tile(
            num.arange(nsamples), nrisetimes).reshape((nrisetimes, nsamples))
        for target in targets:
            for patchidx in range(npatches):
                for risetimeidx in range(nrisetimes):
                    starttimeidxs = range(nstarttimes)

                    self.gfs.put(
                        tracedata * risetimeidx, target, patchidx, risetimeidx,
                        starttimeidxs)

    def test_gf_setup(self):
        print self.gfs
        print self.gfs._gfmatrix

    def test_stacking(self):
        def reference_numpy(gfs, risetimeidxs, starttimeidxs, slips):
            u2d = num.tile(
                slips, gfs.nsamples).reshape(gfs.nsamples, gfs.npatches)
            return num.einsum(
                'ijk->ik',
                gfs._gfmatrix[:, :, risetimeidxs, starttimeidxs, :] * u2d.T)


if __name__ == '__main__':
    util.setup_logging('test_ffi', 'info')
    unittest.main()
