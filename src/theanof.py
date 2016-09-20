'''
Package for wrapping various functions into Theano-Ops to be able to include
them into theano graphs as is needed by the pymc3 models.
Far future:
    include a 'def grad:' -method to each Op in order to enable the use of
    gradient based optimization algorithms
'''
from beat import heart, utility, config

import theano.tensor as tt
import theano

import numpy as num

km = 1000.


class GeoLayerSynthesizerFree(theano.Op):
    '''
    Theano wrapper for a geodetic forward model for variable observation
    points including opening- can be used for dike/sill modeling.

    Inputs have to be in order!
    Type Numpy arrays:
    Observation|             Source parameters (RectangularSource)
    lons, lats | east_shifts, north_shifts, top_depths, strikes, dips, rakes,
                 lengths, widths, slips, openings
    '''

    __props__ = ('store_superdir', 'crust_ind', 'sources')

    def __init__(self, store_superdir, crust_ind, sources):
        self.store_superdir = store_superdir
        self.crust_ind = crust_ind
        self.sources = tuple(sources)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, *inputs):
        inlist = []
        for i in inputs:
            inlist.append(tt.as_tensor_variable(i))

        outm = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [outm.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):

        z = output[0]

        lons = inputs.pop(0)
        lats = inputs.pop(0)

        point = {var: inp for var, inp in zip(
                    config.geo_vars_magma, inputs)}

        point = utility.adjust_point_units(point)

        source_points = utility.split_point(point)

        for i, source in enumerate(self.sources):
            source.update(**source_points[i])
            heart.adjust_fault_reference(source, input_depth='top')

        z[0] = heart.geo_layer_synthetics(
            self.store_superdir,
            self.crust_ind, lons, lats, self.sources)

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

    def __init__(self, lats, lons, store_superdir, crust_ind, sources):
        self.lats = tuple(lats)
        self.lons = tuple(lons)
        self.store_superdir = store_superdir
        self.crust_ind = crust_ind
        self.sources = tuple(sources)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, *inputs):
        inlist = []
        for i in inputs:
            inlist.append(tt.as_tensor_variable(i))

        out = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [out.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):

        z = output[0]

        point = {var: inp for var, inp in zip(
                    config.geo_vars_geometry, inputs)}

        point = utility.adjust_point_units(point)

        source_points = utility.split_point(point)

        for i, source in enumerate(self.sources):
            source.update(**source_points[i])
            heart.adjust_fault_reference(source, input_depth='top')

        z[0] = heart.geo_layer_synthetics(
            store_superdir=self.store_superdir,
            crust_ind=self.crust_ind,
            lons=self.lons,
            lats=self.lats,
            sources=self.sources)

    def infer_shape(self, node, input_shapes):
        return [(len(self.lats), 3)]


class SeisSynthesizer(theano.Op):

    __props__ = ('engine', 'sources', 'targets', 'event',
                 'arrival_taper', 'filterer')

    def __init__(self, engine, sources, targets, event, arrival_taper,
                 filterer):
        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.event = event
        self.arrival_taper = arrival_taper
        self.filterer = filterer

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, *inputs):
        inlist = []
        for i in inputs:
            inlist.append(tt.as_tensor_variable(i))

        outm = tt.as_tensor_variable(num.zeros((2, 2)))
        outv = tt.as_tensor_variable(num.zeros((2)))
        outlist = [outm.type(), outv.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):

        synths = output[0]
        tmins = output[1]

        point = {var: inp for var, inp in zip(
                    config.joint_vars_geometry, inputs)}

        mpoint = utility.adjust_point_units(point)

        source_points = utility.split_point(mpoint)

        for i, source in enumerate(self.sources):
            source.update(**source_points[i])
            source.time += self.event.time
            heart.adjust_fault_reference(source, input_depth='top')

        synths[0], tmins[0] = heart.seis_synthetics(
                self.engine, self.sources,
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

    def __init__(self, sample_rate, traces, arrival_taper, filterer):
        self.sample_rate = sample_rate
        self.traces = tuple(traces)
        self.arrival_taper = arrival_taper
        self.filterer = filterer

    def make_node(self, *inputs):
        inlist = []
        for i in inputs:
            inlist.append(tt.as_tensor_variable(i))

        outm = tt.as_tensor_variable(num.zeros((2, 2)))

        outlist = [outm.type()]
        return theano.Apply(self, inlist, outlist)

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
