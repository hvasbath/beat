"""
Package for wrapping various functions into Theano-Ops to be able to include
them into theano graphs as is needed by the pymc3 models.

Far future:
    include a 'def grad:' -method to each Op in order to enable the use of
    gradient based optimization algorithms
"""
import copy
import logging
from collections import OrderedDict

import numpy as num
import theano
import theano.tensor as tt
from pymc3.model import FreeRV
from pyrocko.trace import nextpow2

from beat import heart, interseismic, utility
from beat.fast_sweeping import fast_sweep

km = 1000.0
logger = logging.getLogger("theanof")


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

    __props__ = ("engine", "sources", "targets")

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

        self.varnames = list(inputs.keys())

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
            # reset source time may result in store error otherwise
            source.time = 0.0

        synths[0] = heart.geo_synthetics(
            engine=self.engine,
            targets=self.targets,
            sources=self.sources,
            outmode="stacked_array",
        )

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

    __props__ = ("lats", "lons", "store_superdir", "crust_ind", "sources")

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
        self.varnames = list(inputs.keys())

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
            sources=self.sources,
        )

    def infer_shape(self, node, input_shapes):
        return [(len(self.lats), 3)]


class GeoInterseismicSynthesizer(theano.Op):
    """
    Theano wrapper to transform the parameters of block model to
    parameters of a fault.
    """

    __props__ = ("lats", "lons", "engine", "targets", "sources", "reference")

    def __init__(self, lats, lons, engine, targets, sources, reference):
        self.lats = tuple(lats)
        self.lons = tuple(lons)
        self.engine = engine
        self.targets = tuple(targets)
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
            :class:`pyrocko.gf.seismosizer.RectangularSource` that was used
            to initialise the Operator.
            values are :class:`theano.tensor.Tensor`
        """
        inlist = []

        self.fixed_values = {}
        self.varnames = []

        for k, v in inputs.items():
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
            of synthetic displacements of :class:`numpy.ndarray` (n x 3)
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
            engine=self.engine,
            targets=self.targets,
            sources=self.sources,
            lons=num.array(self.lons),
            lats=num.array(self.lats),
            reference=self.reference,
            **bpoint
        )

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
    arrival_times : :class:`ǹumpy.NdArray`
        with synthetic arrival times wrt reference event
    filterer : :class:`heart.Filterer`
    """

    __props__ = (
        "engine",
        "sources",
        "targets",
        "event",
        "arrival_taper",
        "arrival_times",
        "wavename",
        "filterer",
        "pre_stack_cut",
        "station_corrections",
        "domain",
    )

    def __init__(
        self,
        engine,
        sources,
        targets,
        event,
        arrival_taper,
        arrival_times,
        wavename,
        filterer,
        pre_stack_cut,
        station_corrections,
        domain,
    ):
        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.event = event
        self.arrival_taper = arrival_taper
        self.arrival_times = tuple(arrival_times.tolist())
        self.wavename = wavename
        self.filterer = tuple(filterer)
        self.pre_stack_cut = pre_stack_cut
        self.station_corrections = station_corrections
        self.domain = domain
        self.sample_rate = self.engine.get_store(
            self.targets[0].store_id
        ).config.sample_rate

        if self.domain == "spectrum":
            nsamples = nextpow2(self.arrival_taper.nsamples(self.sample_rate))
            taper_frequencies = heart.filterer_minmax(filterer)
            deltaf = self.sample_rate / nsamples

            self.valid_spectrum_indices = utility.get_valid_spectrum_data(
                deltaf, taper_frequencies
            )

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

        self.varnames = list(inputs.keys())

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

        point = {vname: i for vname, i in zip(self.varnames, inputs)}

        mpoint = utility.adjust_point_units(point)

        if self.station_corrections:
            time_shifts = mpoint.pop("time_shift").ravel()
            arrival_times = num.array(self.arrival_times) + time_shifts
        else:
            arrival_times = num.array(self.arrival_times)

        source_points = utility.split_point(mpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            source.time += self.event.time

        synthetics, tmins[0] = heart.seis_synthetics(
            engine=self.engine,
            sources=self.sources,
            targets=self.targets,
            arrival_taper=self.arrival_taper,
            wavename=self.wavename,
            filterer=self.filterer,
            pre_stack_cut=self.pre_stack_cut,
            arrival_times=arrival_times,
        )

        if self.domain == "time":
            synths[0] = synthetics

        elif self.domain == "spectrum":
            synths[0] = heart.fft_transforms(
                time_domain_signals=synthetics,
                valid_spectrum_indices=self.valid_spectrum_indices,
                outmode="array",
                pad_to_pow2=True,
            )
        else:
            ValueError('Domain "%" not supported!' % self.domain)

    def infer_shape(self, node, input_shapes):
        nrow = len(self.targets)

        if self.domain == "time":
            ncol = self.arrival_taper.nsamples(self.sample_rate)
        elif self.domain == "spectrum":
            ncol = self.valid_spectrum_indices[1] - self.valid_spectrum_indices[0]
        return [(nrow, ncol), (nrow,)]


class PolaritySynthesizer(theano.Op):

    __props__ = ("engine", "source", "pmap", "is_location_fixed", "always_raytrace")

    def __init__(self, engine, source, pmap, is_location_fixed, always_raytrace):
        self.engine = engine
        self.source = source
        self.pmap = pmap
        self.is_location_fixed = is_location_fixed
        self.always_raytrace = always_raytrace

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, inputs):
        inlist = []
        self.varnames = list(inputs.keys())
        for i in inputs.values():
            inlist.append(tt.as_tensor_variable(i))

        out = tt.as_tensor_variable(num.zeros((2)))
        outlist = [out.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        synths = output[0]
        point = {vname: i for vname, i in zip(self.varnames, inputs)}
        mpoint = utility.adjust_point_units(point)
        source_points = utility.split_point(mpoint)

        utility.update_source(self.source, **source_points[self.pmap.config.event_idx])

        if not self.is_location_fixed:
            self.pmap.update_targets(
                self.engine,
                self.source,
                always_raytrace=self.always_raytrace,
                check=False,
            )
            self.pmap.update_radiation_weights()

        synths[0] = heart.pol_synthetics(
            self.source, radiation_weights=self.pmap.get_radiation_weights()
        )

    def infer_shape(self, node, input_shapes):
        return [(self.pmap.n_t,)]


class SeisDataChopper(theano.Op):
    """
    Deprecated!
    """

    __props__ = ("sample_rate", "traces", "arrival_taper", "filterer")

    def __init__(self, sample_rate, traces, arrival_taper, filterer):
        self.sample_rate = sample_rate
        self.traces = tuple(traces)
        self.arrival_taper = arrival_taper
        self.filterer = tuple(filterer)

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

        z[0] = heart.taper_filter_traces(
            self.traces, self.arrival_taper, self.filterer, tmins, outmode="array"
        )

    def infer_shape(self, node, input_shapes):
        nrow = len(self.traces)
        ncol = self.arrival_taper.nsamples(self.sample_rate)
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

    __props__ = ("patch_size", "n_patch_dip", "n_patch_strike", "implementation")

    def __init__(self, patch_size, n_patch_dip, n_patch_strike, implementation):

        self.patch_size = num.float64(patch_size)
        self.n_patch_dip = n_patch_dip
        self.n_patch_strike = n_patch_strike
        self.implementation = implementation

    def make_node(self, *inputs):
        inlist = []
        for i in inputs:
            inlist.append(tt.as_tensor_variable(i))

        outv = tt.as_tensor_variable(num.zeros((2)))
        outlist = [outv.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Return start-times of rupturing patches with respect to
        given hypocenter.

        Parameters
        ----------
        slownesses : float, vector
            inverse of the rupture velocity across each patch
        nuc_dip : int, scalar
            rupture nucleation point on the fault in dip-direction,
            index to patch
        nuc_strike : int, scalar
            rupture nucleation point on the fault in strike-direction,
            index to patch

        Returns
        -------
        starttimes : float, vector

        Notes
        -----
        Here we call the C-implementation on purpose with swapped
        strike and dip directions, because we need the
        fault dipping in row directions of the array.
        The C-implementation has it along columns!!!
        """
        slownesses, nuc_dip, nuc_strike = inputs
        z = output[0]
        logger.debug("Fast sweeping ..%s." % self.implementation)
        if self.implementation == "c":
            #
            z[0] = fast_sweep.fast_sweep_ext.fast_sweep(
                slownesses,
                self.patch_size,
                int(nuc_dip),
                int(nuc_strike),
                self.n_patch_dip,
                self.n_patch_strike,
            )

        elif self.implementation == "numpy":
            z[0] = fast_sweep.get_rupture_times_numpy(
                slownesses.reshape((self.n_patch_dip, self.n_patch_strike)),
                self.patch_size,
                self.n_patch_strike,
                self.n_patch_dip,
                nuc_strike,
                nuc_dip,
            ).flatten()

        else:
            raise NotImplementedError(
                "Fast sweeping for implementation %s not"
                " implemented!" % self.implementation
            )

        logger.debug("Done sweeping!")

    def infer_shape(self, node, input_shapes):
        return [(self.n_patch_dip * self.n_patch_strike,)]


class EulerPole(theano.Op):
    """
    Theano Op for rotation of geodetic observations around Euler Pole.

    Parameters
    ----------
    lats : float, vector
        of Latitudes [deg] of points to be rotated
    lons : float, vector
        of Longitudes [deg] of points to be rotated
    data_mask : bool, vector
        of indexes to points to be masked
    """

    __props__ = ("lats", "lons", "data_mask")

    def __init__(self, lats, lons, data_mask):

        self.lats = tuple(lats)
        self.lons = tuple(lons)
        self.data_mask = tuple(data_mask)

    def make_node(self, inputs):
        inlist = []

        self.fixed_values = OrderedDict()
        self.varnames = []

        for k, v in inputs.items():
            varname = k.split("_")[-1]  # split of dataset naming
            if isinstance(v, FreeRV):
                self.varnames.append(varname)
                inlist.append(tt.as_tensor_variable(v))
            else:
                self.fixed_values[varname] = v

        outv = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [outv.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):

        z = output[0]
        point = {vname: i for vname, i in zip(self.varnames, inputs)}
        point.update(self.fixed_values)

        pole_lat = point["lat"]  # pole parameters
        pole_lon = point["lon"]
        omega = point["omega"]

        velocities = heart.velocities_from_pole(
            num.array(self.lats), num.array(self.lons), pole_lat, pole_lon, omega
        )

        if self.data_mask:
            velocities[num.array(self.data_mask), :] = 0.0

        z[0] = velocities

    def infer_shape(self, node, input_shapes):
        return [(len(self.lats), 3)]


class StrainRateTensor(theano.Op):
    """
    TheanoOp for internal block deformation through 2d area strain rate tensor.

    Parameters
    ----------
    lats : float, vector
        of Latitudes [deg] of points to be strained
    lons : float, vector
        of Longitudes [deg] of points to be strained
    data_mask : bool, vector
        of indexes to points to be masked
    """

    __props__ = ("lats", "lons", "data_mask")

    def __init__(self, lats, lons, data_mask):

        self.lats = tuple(lats)
        self.lons = tuple(lons)
        self.data_mask = tuple(data_mask)
        self.ndata = len(self.lats)

        station_idxs = [
            station_idx
            for station_idx in range(self.ndata)
            if station_idx not in data_mask
        ]

        self.station_idxs = tuple(station_idxs)

    def make_node(self, inputs):
        inlist = []

        self.fixed_values = OrderedDict()
        self.varnames = []

        for k, v in inputs.items():
            varname = k.split("_")[-1]  # split of dataset naming
            if isinstance(v, FreeRV):
                self.varnames.append(varname)
                inlist.append(tt.as_tensor_variable(v))
            else:
                self.fixed_values[varname] = v

        outv = tt.as_tensor_variable(num.zeros((2, 2)))
        outlist = [outv.type()]
        return theano.Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):

        z = output[0]

        point = {vname: i for vname, i in zip(self.varnames, inputs)}
        point.update(self.fixed_values)

        exx = point["exx"]  # tensor params
        eyy = point["eyy"]
        exy = point["exy"]
        rotation = point["rotation"]

        valid = num.array(self.station_idxs)

        v_xyz = heart.velocities_from_strain_rate_tensor(
            num.array(self.lats)[valid],
            num.array(self.lons)[valid],
            exx=exx,
            eyy=eyy,
            exy=exy,
            rotation=rotation,
        )

        v_xyz_all = num.zeros((self.ndata, 3))
        v_xyz_all[valid, :] = v_xyz
        z[0] = v_xyz_all

    def infer_shape(self, node, input_shapes):
        return [(self.ndata, 3)]
