import unittest
import logging
from time import time
from beat import ffi
from beat.heart import DynamicTarget

import numpy as num

from pyrocko import util, gf, model

import theano.tensor as tt
from theano import function
from theano import config as tconfig

km = 1000.

logger = logging.getLogger('test_ffi')


class FFITest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        nsamples = 10
        ntargets = 30
        npatches = 40
        nrisetimes = 10
        nstarttimes = 30

        lats = num.random.randint(low=-90, high=90, size=ntargets)
        lons = num.random.randint(low=-180, high=180, size=ntargets)

        stations = [model.Station(
            lat=lat, lon=lon) for lat, lon in zip(lats, lons)]

        targets = [DynamicTarget(store_id='Test') for i in range(ntargets)]

        wavemap = heart.WaveformMapping(
            name='any_P', stations=stations, targets=targets)

        self.gfs = ffi.SeismicGFLibrary(
            targets=targets, stations=stations, component='uperp')
        self.gfs.setup(ntargets, npatches, nrisetimes, nstarttimes, nsamples)

        tracedata = num.tile(
            num.arange(nsamples), nstarttimes).reshape((nstarttimes, nsamples))
        for target in targets:
            for patchidx in range(npatches):
                for risetimeidx in range(nrisetimes):
                    starttimeidxs = range(nstarttimes)

                    self.gfs.put(
                        tracedata * risetimeidx, target, patchidx, risetimeidx,
                        starttimeidxs)

    def test_gf_setup(self):
        print self.gfs
        # print self.gfs._gfmatrix

    def test_stacking(self):
        def reference_numpy(gfs, risetimeidxs, starttimeidxs, slips):
            u2d = num.tile(
                slips, gfs.nsamples).reshape(gfs.nsamples, gfs.npatches)
            t0 = time()
            patchidxs = num.arange(gfs.npatches)
            d = gfs._gfmatrix[:, patchidxs, risetimeidxs, starttimeidxs, :]
            d1 = d.reshape(
                (self.gfs.ntargets, self.gfs.npatches, self.gfs.nsamples))
            out_array = num.einsum('ijk->ik', d1 * u2d.T)
            t1 = time()
            logger.info('Calculation time numpy einsum: %f', (t1 - t0))
            return out_array

        def prepare_theano(gfs, runidx=0):
            theano_rts = tt.vector('rt_indxs_%i' % runidx, dtype='int16')
            theano_stts = tt.vector('start_indxs_%i' % runidx, dtype='int16')
            theano_slips = tt.dvector('slips_%i' % runidx)
            gfs.init_optimization()
            return theano_rts, theano_stts, theano_slips

        def theano_batched_dot(gfs, risetimes, starttimes, slips):
            theano_rts, theano_stts, theano_slips = prepare_theano(gfs, 0)

            outstack = gfs.stack_all(
                starttimes=theano_stts,
                risetimes=theano_rts,
                slips=theano_slips)

            t0 = time()
            f = function([theano_slips, theano_rts, theano_stts], [outstack])
            t1 = time()
            logger.info('Compile time theano batched_dot: %f', (t1 - t0))

            out_array = f(slips, risetimeidxs, starttimeidxs)[0]
            t2 = time()
            logger.info('Calculation time batched_dot: %f', (t2 - t1))
            return out_array.squeeze()

        def theano_for_loop(gfs, risetimeidxs, starttimeidxs, slips):
            theano_rts, theano_stts, theano_slips = prepare_theano(gfs, 1)

            patchidxs = range(gfs.npatches)

            outstack = tt.zeros((gfs.ntargets, gfs.nsamples), tconfig.floatX)
            for i, target in enumerate(gfs.targets):
                synths = gfs.stack(
                    target=target,
                    patchidxs=patchidxs,
                    risetimeidxs=theano_rts,
                    starttimeidxs=theano_stts,
                    slips=theano_slips)
                outstack = tt.set_subtensor(
                    outstack[i:i + 1, 0:gfs.nsamples], synths)

            t0 = time()
            f = function([theano_slips, theano_rts, theano_stts], [outstack])
            t1 = time()
            logger.info('Compile time theano for loop: %f', (t1 - t0))
            out_array = f(slips, risetimeidxs, starttimeidxs)[0]
            t2 = time()
            logger.info('Calculation time for loop: %f', (t2 - t1))
            return out_array.squeeze()

        risetimeidxs = num.random.randint(
            low=0, high=self.gfs.nrisetimes,
            size=self.gfs.npatches, dtype='int16')
        starttimeidxs = num.random.randint(
            low=0, high=self.gfs.nstarttimes,
            size=self.gfs.npatches, dtype='int16')
        slips = num.random.random(self.gfs.npatches)

        outnum = reference_numpy(self.gfs, risetimeidxs, starttimeidxs, slips)
        outtheanobatch = theano_batched_dot(
            self.gfs, risetimeidxs - 0.2, starttimeidxs - 0.3, slips)
        outtheanofor = theano_for_loop(
            self.gfs, risetimeidxs, starttimeidxs, slips)

        num.testing.assert_allclose(outnum, outtheanobatch, rtol=0., atol=1e-6)
        num.testing.assert_allclose(outnum, outtheanofor, rtol=0., atol=1e-6)


if __name__ == '__main__':
    util.setup_logging('test_ffi', 'debug')
    unittest.main()
