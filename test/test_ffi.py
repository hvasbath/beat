import unittest
import logging
from time import time
from beat import ffi
from beat.heart import DynamicTarget, WaveformMapping
from beat.utility import get_random_uniform

import numpy as num

from pyrocko import util, model

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

        self.times = get_random_uniform(300., 500., ntargets)

        self.starttime_min = 0.
        self.starttime_max = 15.
        starttime_sampling = 0.5

        self.duration_min = 5.
        self.duration_max = 10.
        duration_sampling = 0.5

        durations = num.arange(
            self.duration_min, self.duration_max, duration_sampling)
        starttimes = num.arange(
            self.starttime_min, self.starttime_max, starttime_sampling)

        nstarttimes = starttimes.size
        ndurations = durations.size

        lats = num.random.randint(low=-90, high=90, size=ntargets)
        lons = num.random.randint(low=-180, high=180, size=ntargets)

        stations = [model.Station(
            lat=lat, lon=lon) for lat, lon in zip(lats, lons)]

        targets = [DynamicTarget(store_id='Test') for i in range(ntargets)]

        wavemap = WaveformMapping(
            name='any_P', stations=stations, targets=targets)

        self.gfs = ffi.SeismicGFLibrary(
            wavemap=wavemap, component='uperp',
            duration_sampling=duration_sampling,
            starttime_sampling=starttime_sampling,
            starttime_min=self.starttime_min,
            duration_min=self.duration_min)
        self.gfs.setup(ntargets, npatches, ndurations, nstarttimes, nsamples)

        tracedata = num.tile(
            num.arange(nsamples), nstarttimes).reshape((nstarttimes, nsamples))
        for i, target in enumerate(targets):
            for patchidx in range(npatches):
                for duration in durations:
                    tmin = self.times[i]
                    self.gfs.put(
                        tracedata * i, tmin, target, patchidx, duration,
                        starttimes)

    def test_gf_setup(self):
        print self.gfs
        # print self.gfs._gfmatrix

    def test_stacking(self):
        def reference_numpy(gfs, durations, starttimes, slips):
            t0 = time()
            out_array = gfs.stack_all(
                starttimes=starttimes,
                durations=durations,
                slips=slips)

            t1 = time()
            logger.info('Calculation time numpy einsum: %f', (t1 - t0))
            return out_array

        def prepare_theano(gfs, runidx=0):
            theano_rts = tt.dvector('durations_%i' % runidx)
            theano_stts = tt.dvector('starttimes_%i' % runidx)
            theano_slips = tt.dvector('slips_%i' % runidx)
            gfs.init_optimization()
            return theano_rts, theano_stts, theano_slips

        def theano_batched_dot(gfs, durations, starttimes, slips):
            theano_rts, theano_stts, theano_slips = prepare_theano(gfs, 0)

            outstack = gfs.stack_all(
                starttimes=theano_stts,
                durations=theano_rts,
                slips=theano_slips)

            t0 = time()
            f = function([theano_slips, theano_rts, theano_stts], outstack)
            t1 = time()
            logger.info('Compile time theano batched_dot: %f', (t1 - t0))

            out_array = f(slips, durationidxs, starttimeidxs)
            t2 = time()
            logger.info('Calculation time batched_dot: %f', (t2 - t1))
            return out_array.squeeze()

        def theano_for_loop(gfs, durationidxs, starttimeidxs, slips):
            theano_rts, theano_stts, theano_slips = prepare_theano(gfs, 1)

            patchidxs = range(gfs.npatches)

            outstack = tt.zeros((gfs.ntargets, gfs.nsamples), tconfig.floatX)
            for i, target in enumerate(gfs.targets):
                synths = gfs.stack(
                    target=target,
                    patchidxs=patchidxs,
                    durationidxs=theano_rts,
                    starttimeidxs=theano_stts,
                    slips=theano_slips)
                outstack = tt.set_subtensor(
                    outstack[i:i + 1, 0:gfs.nsamples], synths)

            t0 = time()
            f = function([theano_slips, theano_rts, theano_stts], outstack)
            t1 = time()
            logger.info('Compile time theano for loop: %f', (t1 - t0))
            out_array = f(slips, durationidxs, starttimeidxs)
            t2 = time()
            logger.info('Calculation time for loop: %f', (t2 - t1))
            return out_array.squeeze()

        durations = get_random_uniform(
            self.duration_min, self.duration_max, dimension=self.gfs.npatches)
        starttimes = get_random_uniform(
            self.starttime_min,
            self.starttime_max,
            dimension=self.gfs.npatches)
        slips = num.random.random(self.gfs.npatches)

        outnum = reference_numpy(self.gfs, durations, starttimes, slips)
        outtheanobatch = theano_batched_dot(
            self.gfs, durations, starttimes, slips)

        self.gfs.set_stack_mode('numpy')
        durationidxs = self.gfs.durations2idxs(durations)
        starttimeidxs = self.gfs.starttimes2idxs(starttimes)
        outtheanofor = theano_for_loop(
            self.gfs, durationidxs, starttimeidxs, slips)

        num.testing.assert_allclose(outnum, outtheanobatch, rtol=0., atol=1e-6)
        num.testing.assert_allclose(outnum, outtheanofor, rtol=0., atol=1e-6)


if __name__ == '__main__':
    util.setup_logging('test_ffi', 'debug')
    unittest.main()
