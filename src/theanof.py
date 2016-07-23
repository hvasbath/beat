'''
Package for wrapping various functions into Theano-Ops to be able to include
them into theano graphs as is needed by the pymc3 models.
Far future:
    include a 'def grad:' -method to each Op in order to enable the use of
    gradient based optimization algorithms
'''
from beat import heart

import theano.tensor as tt
import theano

import numpy as num

km = 1000.


class GeoLayerSynthesizer(theano.Op):

    __props__ = ('store_superdir', 'crust_ind', 'sources')

    itypes = [tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, store_superdir, crust_ind, sources):
        self.store_superdir = store_superdir
        self.crust_ind = crust_ind
        self.sources = tuple(sources)

    def perform(self, node, inputs, output):

        lons, lats, ess, nss, ds, sts, dis, ras, ls, ws, sls, ops = inputs
        z = output[0]

        for es, ns, d, st, di, ra, l, w, sl, op, source in \
            zip(ess, nss, ds, sts, dis, ras, ls, ws, sls, ops, self.sources):
            source.update(east_shift=es * km,
                          north_shift=ns * km, depth=d,
                          strike=st, dip=di, rake=ra,
                          length=l, width=w, slip=sl,
                          opening=op)

        displ = heart.geo_layer_synthetics(
            self.store_superdir,
            self.crust_ind, lons, lats, self.sources)

        z[0] = displ[0]

    def infer_shape(self, node, input_shapes):
        return [(input_shapes[0][0], 3)]


class SeisSynthesizer(theano.Op):

    __props__ = ('engine', 'sources', 'targets', 'event',
                 'arrival_taper', 'filterer')

    itypes = [tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix, tt.dvector]

    def __init__(self, engine, sources, targets, event, arrival_taper,
                 filterer):
        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.event = event
        self.arrival_taper = arrival_taper
        self.filterer = filterer

    def perform(self, node, inputs, output):

        ess, nss, ds, sts, dis, ras, ls, ws, sls, ts, rts = inputs
        synths = output[0]
        tmins = output[1]

        for es, ns, d, st, di, ra, l, w, sl, t, rt, source in \
            zip(ess, nss, ds, sts, dis, ras, ls, ws, sls, ts, rts, self.sources):
            source.update(east_shift=es * km, north_shift=ns * km, depth=d * km,
                          strike=st, dip=di, rake=ra,
                          length=l * km, width=w * km, slip=sl,
                          time=(self.event.time + t))
            source.stf.duration = rt

        synths[0], tmins[0] = heart.seis_synthetics(self.engine, self.sources,
                                              self.targets,
                                              self.arrival_taper,
                                              self.filterer)

    def infer_shape(self, node, input_shapes):
        nrow = len(self.targets)
        store = self.engine.get_store(self.targets[0].store_id)
        ncol = int(num.ceil(store.config.sample_rate * \
                (self.arrival_taper.d + self.arrival_taper.a)))
        return [(nrow, ncol), (nrow,)]


class SeisDataChopper(theano.Op):

    __props__ = ('sample_rate', 'traces', 'arrival_taper', 'filterer')

    itypes = [tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, sample_rate, traces, arrival_taper, filterer):
        self.sample_rate = sample_rate
        self.traces = tuple(traces)
        self.arrival_taper = arrival_taper
        self.filterer = filterer

    def perform(self, node, inputs, output):
        tmins = inputs[0]
        z = output[0]

        z[0] = heart.taper_filter_traces(self.traces, self.arrival_taper,
                                         self.filterer, tmins)

    def infer_shape(self, node, input_shapes):
        nrow = len(self.traces)
        ncol = int(num.ceil(self.sample_rate * \
                (self.arrival_taper.d + self.arrival_taper.a)))
        return [(nrow, ncol)]
