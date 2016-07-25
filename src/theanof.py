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


class GeoLayerSynthesizerFree(theano.Op):
    '''
    Theano wrapper for a geodetic forward model for variable observation
    points.

    Inputs have to be in order!
    Type Numpy arrays:
    Observation|             Source parameters (RectangularSource)
    lons, lats | east_shifts, north_shifts, top_depths, strikes, dips, rakes,
                 lengths, widths, slips, openings
    '''

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

        lons, lats, ess, nss, tds, sts, dis, ras, ls, ws, sls, ops = inputs
        z = output[0]

        for es, ns, td, st, di, ra, l, w, sl, op, source in \
            zip(ess, nss, tds, sts, dis, ras, ls, ws, sls, ops, self.sources):
            source.update(east_shift=es * km,
                          north_shift=ns * km,
                          strike=st, dip=di, rake=ra,
                          length=l * km, width=w * km, slip=sl,
                          opening=op)
            heart.update_center_coords(source, td * km)

        displ = heart.geo_layer_synthetics(
            self.store_superdir,
            self.crust_ind, lons, lats, self.sources)

        z[0] = displ[0]

    def infer_shape(self, node, input_shapes):
        return [(input_shapes[0][0], 3)]


class GeoLayerSynthesizerStatic(theano.Op):
    '''
    Theano wrapper for a geodetic forward model for static observation
    points.

    Inputs have to be in order!
    Type Numpy arrays:
                    Source parameters (RectangularSource)
    east_shifts, north_shifts, top_depths, strikes, dips, rakes,
    lengths, widths, slips
    '''
    __props__ = ('lats', 'lons', 'store_superdir', 'crust_ind', 'sources')

    itypes = [tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, lats, lons, store_superdir, crust_ind, sources):
        self.lats = tuple(lats)
        self.lons = tuple(lons)
        self.store_superdir = store_superdir
        self.crust_ind = crust_ind
        self.sources = tuple(sources)

    def perform(self, node, inputs, output):

        ess, nss, tds, sts, dis, ras, ls, ws, sls = inputs
        z = output[0]

        for es, ns, td, st, di, ra, l, w, sl, source in \
            zip(ess, nss, tds, sts, dis, ras, ls, ws, sls, self.sources):
            source.update(east_shift=float(es * km),
                          north_shift=float(ns * km),
                          strike=float(st), dip=float(di), rake=float(ra),
                          length=float(l * km), width=float(w * km),
                          slip=float(sl))
            heart.update_center_coords(source, td * km)

        displ = heart.geo_layer_synthetics(
            store_superdir=self.store_superdir,
            crust_ind=self.crust_ind,
            lons=self.lons,
            lats=self.lats,
            sources=self.sources)

        z[0] = displ[0]

    def infer_shape(self, node, input_shapes):
        return [(len(self.lats), 3)]


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

        ess, nss, tds, sts, dis, ras, ls, ws, sls, ts, rts = inputs
        synths = output[0]
        tmins = output[1]

        for es, ns, td, st, di, ra, l, w, sl, t, rt, source in \
            zip(ess, nss, tds, sts, dis, ras, ls, ws, sls, ts, rts, self.sources):
            source.update(east_shift=float(es * km),
                          north_shift=float(ns * km),
                          strike=float(st), dip=float(di), rake=float(ra),
                          length=float(l * km), width=float(w * km),
                          slip=float(sl),
                          time=float(self.event.time + t))
            heart.update_center_coords(source, td * km)
            source.stf.duration = float(rt)

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
