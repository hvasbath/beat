"""
Package for wrapping various functions into Theano-Ops to be able to include
them into theano graphs as is needed by the pymc3 models.

Far future:
    include a 'def grad:' -method to each Op in order to enable the use of
    gradient based optimization algorithms
"""
from beat import heart, utility, interseismic
from beat.fast_sweeping import fast_sweep

from pymc3.model import FreeRV

import theano.tensor as tt
import theano

import numpy as num

km = 1000.


class GeoSynthesizer(theano.Op):
    """
    Theano wrapper for a geodetic forward model with synthetic displacements.
    Uses pyrocko engine and fomosto GF stores.
    Input order does not matter anymore! Did in previous version.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    sources : List
        containing :class:`pyrocko.gf.seismosizer.Source` Objects
    targets : List
        containing :class:`pyrocko.gf.targets.StaticTarget` Objects
    """

    __props__ = ('engine', 'sources', 'targets')

    def __init__(self, engine, sources, targets):
        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.nobs = sum([target.lats.size for target in self.targets])

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, inputs):
        """
        Transforms theano tensors to node and allocates variables accordingly.

        Parameters
        ----------
        inputs : dict
            keys being strings of source attributes of the
            :class:`pscmp.RectangularSource` that was used to initialise
            the Operator
            values are :class:`theano.tensor.Tensor`
        """
        inlist = []

        self.varnames = inputs.keys()

        for i in inputs.values():
            inlist.append(tt.as_tensor_variable(i))

        outm = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [outm.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Perform method of the Operator to calculate synthetic displacements.

        Parameters
        ----------
        inputs : list
            of :class:`numpy.ndarray`
        output : list
            1) of synthetic waveforms of :class:`numpy.ndarray`
               (n x nsamples)
            2) of start times of the first waveform samples
               :class:`numpy.ndarray` (n x 1)
        """
        synths = output[0]

        point = {vname: i for vname, i in zip(self.varnames, inputs)}

        mpoint = utility.adjust_point_units(point)

        source_points = utility.split_point(mpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])

        synths[0] = heart.geo_synthetics(
            engine=self.engine,
            targets=self.targets,
            sources=self.sources,
            outmode='stacked_array')

    def infer_shape(self, node, input_shapes):
        return [(self.nobs, 3)]


class GeoLayerSynthesizerPsCmp(theano.Op):
    """
    Theano wrapper for a geodetic forward model for static observation
    points. Direct call to PsCmp, needs PsGrn Greens Function store!
    Deprecated, currently not used in composites.

    Parameters
    ----------
    lats : n x 1 :class:`numpy.ndarray`
        with latitudes of observation points
    lons : n x 1 :class:`numpy.ndarray`
        with longitudes of observation points
    store_superdir : str
        with absolute path to the GF store super directory
    crust_ind : int
        with the index to the GF store
    sources : :class:`pscmp.RectangularSource`
        to be used in generating the synthetic displacements
    """

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

    def make_node(self, inputs):
        """
        Transforms theano tensors to node and allocates variables accordingly.

        Parameters
        ----------
        inputs : dict
            keys being strings of source attributes of the
            :class:`pscmp.RectangularSource` that was used to initialise
            the Operator
            values are :class:`theano.tensor.Tensor`
        """
        inlist = []
        self.varnames = inputs.keys()

        for i in inputs.values():
            inlist.append(tt.as_tensor_variable(i))

        out = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [out.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Perform method of the Operator to calculate synthetic displacements.

        Parameters
        ----------
        inputs : list
            of :class:`numpy.ndarray`
        output : list
            of synthetic displacements of :class:`numpy.ndarray` (n x 1)
        """
        z = output[0]

        point = {vname: i for vname, i in zip(self.varnames, inputs)}

        point = utility.adjust_point_units(point)

        source_points = utility.split_point(point)

        for i, source in enumerate(self.sources):
            source.update(**source_points[i])

        z[0] = heart.geo_layer_synthetics_pscmp(
            store_superdir=self.store_superdir,
            crust_ind=self.crust_ind,
            lons=self.lons,
            lats=self.lats,
            sources=self.sources)

    def infer_shape(self, node, input_shapes):
        return [(len(self.lats), 3)]


class GeoInterseismicSynthesizer(theano.Op):
    """
    Theano wrapper to transform the parameters of block model to
    parameters of a fault.
    """
    __props__ = (
        'lats', 'lons', 'store_superdir', 'crust_ind', 'sources', 'reference')

    def __init__(
        self, lats, lons, store_superdir, crust_ind, sources, reference):
        self.lats = tuple(lats)
        self.lons = tuple(lons)
        self.store_superdir = store_superdir
        self.crust_ind = crust_ind
        self.sources = tuple(sources)
        self.reference = reference

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, inputs):
        """
        Transforms theano tensors to node and allocates variables accordingly.

        Parameters
        ----------
        inputs : dict
            keys being strings of source attributes of the
            :class:`pscmp.RectangularSource` that was used to initialise
            the Operator
            values are :class:`theano.tensor.Tensor`
        """
        inlist = []

        self.fixed_values = {}
        self.varnames = []

        for k, v in inputs.iteritems():
            if isinstance(v, FreeRV):
                self.varnames.append(k)
                inlist.append(tt.as_tensor_variable(v))
            else:
                self.fixed_values[k] = v

        out = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [out.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Perform method of the Operator to calculate synthetic displacements.

        Parameters
        ----------
        inputs : list
            of :class:`numpy.ndarray`
        output : list
            of synthetic displacements of :class:`numpy.ndarray` (n x 1)
        """
        z = output[0]

        point = {vname: i for vname, i in zip(self.varnames, inputs)}
        point.update(self.fixed_values)

        point = utility.adjust_point_units(point)
        spoint, bpoint = interseismic.seperate_point(point)

        source_points = utility.split_point(spoint)

        for i, source_point in enumerate(source_points):
            self.sources[i].update(**source_point)

        z[0] = interseismic.geo_backslip_synthetics(
            store_superdir=self.store_superdir,
            crust_ind=self.crust_ind,
            sources=self.sources,
            lons=num.array(self.lons),
            lats=num.array(self.lats),
            reference=self.reference,
            **bpoint)

        def infer_shape(self, node, input_shapes):
            return [(len(self.lats), 3)]


class SeisSynthesizer(theano.Op):
    """
    Theano wrapper for a seismic forward model with synthetic waveforms.
    Input order does not matter anymore! Did in previous version.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    sources : List
        containing :class:`pyrocko.gf.seismosizer.Source` Objects
    targets : List
        containing :class:`pyrocko.gf.seismosizer.Target` Objects

    arrival_taper : :class:`heart.ArrivalTaper`
    filterer : :class:`heart.Filterer`
    """

    __props__ = ('engine', 'sources', 'targets', 'event',
                 'arrival_taper', 'filterer', 'pre_stack_cut')

    def __init__(self, engine, sources, targets, event, arrival_taper,
                 filterer, pre_stack_cut):
        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.event = event
        self.arrival_taper = arrival_taper
        self.filterer = filterer
        self.pre_stack_cut = pre_stack_cut

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, inputs):
        """
        Transforms theano tensors to node and allocates variables accordingly.

        Parameters
        ----------
        inputs : dict
            keys being strings of source attributes of the
            :class:`pscmp.RectangularSource` that was used to initialise
            the Operator
            values are :class:`theano.tensor.Tensor`
        """
        inlist = []

        self.varnames = inputs.keys()

        for i in inputs.values():
            inlist.append(tt.as_tensor_variable(i))

        outm = tt.as_tensor_variable(num.zeros((2, 2)))
        outv = tt.as_tensor_variable(num.zeros((2)))
        outlist = [outm.type(), outv.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Perform method of the Operator to calculate synthetic displacements.

        Parameters
        ----------
        inputs : list
            of :class:`numpy.ndarray`
        output : list
            1) of synthetic waveforms of :class:`numpy.ndarray`
               (n x nsamples)
            2) of start times of the first waveform samples
               :class:`numpy.ndarray` (n x 1)
        """
        synths = output[0]
        tmins = output[1]

        point = {vname: i for vname, i in zip(
                    self.varnames, inputs)}

        mpoint = utility.adjust_point_units(point)

        source_points = utility.split_point(mpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            source.time += self.event.time

        synths[0], tmins[0] = heart.seis_synthetics(
            engine=self.engine,
            sources=self.sources,
            targets=self.targets,
            arrival_taper=self.arrival_taper,
            filterer=self.filterer,
            pre_stack_cut=self.pre_stack_cut)

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
                (self.arrival_taper.d + num.abs(self.arrival_taper.a))))
        return [(nrow, ncol)]


class Sweeper(theano.Op):
    """
    Theano Op for C implementation of the fast sweep algorithm.

    Parameters
    ----------
    patch_size : float
        size of fault patches [km]
    n_patch_strike : int
        number of patches in strike direction
    n_patch_dip : int
        number of patches in dip-direction
    """

    __props__ = ('patch_size', 'n_patch_dip', 'n_patch_strike')

    def __init__(self, patch_size, n_patch_strike, n_patch_dip):
        self.patch_size = num.float64(patch_size)
        self.n_patch_dip = n_patch_dip
        self.n_patch_strike = n_patch_strike

    def make_node(self, *inputs):
        inlist = []
        for i in inputs:
            inlist.append(tt.as_tensor_variable(i))

        outv = tt.as_tensor_variable(num.zeros((2)))
        outlist = [outv.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        slownesses, nuc_strike, nuc_dip = inputs

        z = output[0]

        z[0] = fast_sweep.fast_sweep_ext.fast_sweep(
            slownesses, self.patch_size,
            int(nuc_strike), int(nuc_dip),
            self.n_patch_strike, self.n_patch_dip)

    def infer_shape(self, node, input_shapes):
        return [(self.n_patch_dip * self.n_patch_strike, )]
