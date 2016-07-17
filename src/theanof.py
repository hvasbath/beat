'''
Package for wrapping various functions into Theano-Ops to be able to include
them into theano graphs as is needed by the pymc3 models.
Far future:
    include a 'def grad:' -method to each Op in order to enable the use of
    gradient based optimization algorithms
'''

import theano.tensor as tt
import theano
import heart
import numpy as num


class GeoLayerSynthesizer(theano.Op):

    __props__ = ('superdir', 'crust_ind')

    itypes = [tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, superdir, crust_ind):
        self.superdir = superdir
        self.crust_ind = crust_ind

    def perform(self, node, inputs, output):
        ### update use syntax from covariance calculation?
        lons, lats, o_lons, o_lats, ds, st, di, ra, ls, ws, sl, op = inputs
        z = output[0]
        displ = heart.geo_layer_synthetics(
                  superdir=self.superdir, crust_ind=self.crust_ind,
                  lons=lons, lats=lats,
                  o_lons=o_lons, o_lats=o_lats, ds=ds,
                  strikes=st, dips=di, rakes=ra,
                  ls=ls, ws=ws, slips=sl, opns=op)
        z[0] = displ[0]

    def infer_shape(self, node, input_shapes):
        return [(input_shapes[0][0], 3)]


class SeisSynthesizer(theano.Op):

    __props__ = ('engine', 'sources', 'targets', 'event',
                 'arrival_taper', 'filter_char')

    itypes = [tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, engine, sources, targets, event, arrival_taper,
                 filter_char):
        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.event = event
        self.arrival_taper = arrival_taper
        self.filter_char = filter_char

    def perform(self, node, inputs, output):
        ### update use syntax from covariance calculation?
        lons, lats, ds, sts, dis, ras, ls, ws, sls, ts = inputs
        z = output[0]

        for lon, lat, d, st, di, ra, l, w, sl, t, source in \
            zip(lons, lats, ds, sts, dis, ras, ls, ws, sls, ts, self.sources):
            source.update(lon=lon, lat=lat, depth=d,
                          strike=st, dip=di, rake=ra,
                          length=l, width=w, slip=sl,
                          time=(self.event.time + t))

        synths, tmins = heart.seis_synthetics(self.engine,
                                  self.sources, self.targets,
                                  self.arrival_taper, self.filter_char)
        z[0] = synths
        z[1] = tmins

    def infer_shape(self, node, input_shapes):
        nrow = len(self.targets)
        store = self.engine.get_store(self.targets[0].store_id)
        ncol = int(num.ceil(store.config.sample_rate * \
                (self.arrival_taper.d + self.arrival_taper.a)))
        return [(nrow, ncol)]


class SeisDataChopper(theano.Op):

    __props__ = ('sample_rate', 'traces', 'arrival_taper', 'filter_char')

    itypes = [tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, sample_rate, traces, arrival_taper, filter_char):
        self.sample_rate = sample_rate
        self.traces = tuple(traces)
        self.arrival_taper = arrival_taper
        self.filter_char = filter_char

    def perform(self, node, inputs, output):
        ### update use syntax from covariance calculation?
        tmin = inputs
        z = output[0]

        z[0] = heart.taper_filter_traces(traces, arrival_taper, filter_char, tmin)

    def infer_shape(self, node, input_shapes):
        nrow = len(self.traces)
        ncol = int(num.ceil(self.sample_rate * \
                (self.arrival_taper.d + self.arrival_taper.a)))
        return [(nrow, ncol)]
