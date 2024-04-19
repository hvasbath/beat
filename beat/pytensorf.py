"""
Package for wrapping various functions into pytensor-Ops to be able to include
them into pytensor graphs as is needed by the pymc models.

Far future:
    include a 'def grad:' -method to each Op in order to enable the use of
    gradient based optimization algorithms
"""
import logging
from collections import OrderedDict

import numpy as num
import pytensor.tensor as tt
from pyrocko.gf import LocalEngine
from pyrocko.trace import nextpow2
from pytensor.graph import Apply

from beat import heart, utility
from beat.fast_sweeping import fast_sweep

km = 1000.0
logger = logging.getLogger("pytensorf")


class GeoSynthesizer(tt.Op):
    """
    pytensor wrapper for a geodetic forward model with synthetic displacements.
    Uses pyrocko engine and fomosto GF stores.
    Input order does not matter anymore! Did in previous version.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    sources : List
        containing :class:`pyrocko.gf.seismosizer.Source` Objects
    targets : List
        containing :class:`pyrocko.gf.targets.StaticTarget` Objects
    mapping : Dict
        variable names and list of integers how they map to source objects
    """

    __props__ = ("engine", "sources", "targets", "mapping")

    def __init__(self, engine, sources, targets, mapping):
        if isinstance(engine, LocalEngine):
            self.outmode = "stacked_array"
        else:
            self.outmode = "array"

        self.engine = engine
        self.sources = tuple(sources)
        self.targets = tuple(targets)
        self.nobs = sum([target.lats.size for target in self.targets])
        self.mapping = mapping

    def __getstate__(self):
        if isinstance(self.engine, LocalEngine):
            self.engine.close_cashed_stores()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def make_node(self, inputs):
        """
        Transforms pytensor tensors to node and allocates variables accordingly.

        Parameters
        ----------
        inputs : dict
            keys being strings of source attributes of the
            :class:`pscmp.RectangularSource` that was used to initialise
            the Operator
            values are :class:`pytensor.tensor.Tensor`
        """
        inlist = []

        self.varnames = list(inputs.keys())

        for i in inputs.values():
            inlist.append(tt.as_tensor_variable(i))

        outm_shape = self.infer_shape()[0]

        outm = tt.as_tensor_variable(num.zeros(outm_shape))
        outlist = [outm.type()]
        return Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Perform method of the Operator to calculate synthetic displacements.

        Parameters
        ----------
        inputs : list
            of :class:`numpy.ndarray`
        output : list
            1) of synthetic waveforms of :class:`numpy.ndarray` (n x nsamples)
            2) of start times of the first waveform samples :class:`numpy.ndarray` (n x 1)
        """
        synths = output[0]

        point = {vname: i for vname, i in zip(self.varnames, inputs)}

        mpoint = utility.adjust_point_units(point)

        source_points = utility.split_point(
            mpoint,
            mapping=self.mapping,
            weed_params=True,
        )

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            # reset source time may result in store error otherwise
            source.time = 0.0

        synths[0] = heart.geo_synthetics(
            engine=self.engine,
            targets=self.targets,
            sources=self.sources,
            outmode=self.outmode,
        )

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        return [(self.nobs, 3)]


class SeisSynthesizer(tt.Op):
    """
    pytensor wrapper for a seismic forward model with synthetic waveforms.
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
        "mapping",
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
        mapping,
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
        self.mapping = mapping

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
        Transforms pytensor tensors to node and allocates variables accordingly.

        Parameters
        ----------
        inputs : dict
            keys being strings of source attributes of the
            :class:`pscmp.RectangularSource` that was used to initialise
            the Operator
            values are :class:`pytensor.tensor.Tensor`
        """
        inlist = []

        self.varnames = list(inputs.keys())

        for i in inputs.values():
            inlist.append(tt.as_tensor_variable(i))

        outm_shape, outv_shape = self.infer_shape()

        outm = tt.as_tensor_variable(num.zeros(outm_shape))
        outv = tt.as_tensor_variable(num.zeros(outv_shape))
        outlist = [outm.type(), outv.type()]
        return Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        """
        Perform method of the Operator to calculate synthetic displacements.

        Parameters
        ----------
        inputs : list
            of :class:`numpy.ndarray`
        output : list
            1) of synthetic waveforms of :class:`numpy.ndarray` (n x nsamples)
            2) of start times of the first waveform samples :class:`numpy.ndarray` (n x 1)
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

        source_points = utility.split_point(
            mpoint,
            mapping=self.mapping,
            weed_params=True,
        )

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
            ValueError('Domain "%s" not supported!' % self.domain)

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        nrow = len(self.targets)

        if self.domain == "time":
            ncol = self.arrival_taper.nsamples(self.sample_rate)
        elif self.domain == "spectrum":
            ncol = self.valid_spectrum_indices[1] - self.valid_spectrum_indices[0]
        return [(nrow, ncol), (nrow,)]


class PolaritySynthesizer(tt.Op):
    __props__ = ("engine", "source", "pmap", "is_location_fixed", "always_raytrace")

    def __init__(self, engine, source, pmap, is_location_fixed, always_raytrace):
        self.engine = engine
        self.source = source
        self.pmap = pmap
        self.is_location_fixed = is_location_fixed
        self.always_raytrace = always_raytrace
        # TODO check if mutli-source-type refactoring did not break anything

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

        outv_shape = self.infer_shape()[0]
        outv = tt.as_tensor_variable(num.zeros(outv_shape))

        outlist = [outv.type()]
        return Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        synths = output[0]
        point = {vname: i for vname, i in zip(self.varnames, inputs)}
        mpoint = utility.adjust_point_units(point)

        source_points = utility.split_point(
            mpoint,
            n_sources_total=1,
        )
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

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        return [(self.pmap.n_t,)]


class SeisDataChopper(tt.Op):
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

        outm_shape = self.infer_shape()[0]
        outm = tt.as_tensor_variable(num.zeros(outm_shape))

        outlist = [outm.type()]
        return Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        tmins = inputs[0]
        z = output[0]

        z[0] = heart.taper_filter_traces(
            self.traces, self.arrival_taper, self.filterer, tmins, outmode="array"
        )

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        nrow = len(self.traces)
        ncol = self.arrival_taper.nsamples(self.sample_rate)
        return [(nrow, ncol)]


class Sweeper(tt.Op):
    """
    pytensor Op for C implementation of the fast sweep algorithm.

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

        outv_shape = self.infer_shape()[0]
        outv = tt.as_tensor_variable(num.zeros(outv_shape))

        outlist = [outv.type()]
        return Apply(self, inlist, outlist)

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

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        return [(self.n_patch_dip * self.n_patch_strike,)]


class EulerPole(tt.Op):
    """
    pytensor Op for rotation of geodetic observations around Euler Pole.

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
            if isinstance(v, tt.TensorVariable):
                self.varnames.append(varname)
                inlist.append(tt.as_tensor_variable(v))
            else:
                self.fixed_values[varname] = v

        outm_shape = self.infer_shape()[0]
        outm = tt.as_tensor_variable(num.zeros(outm_shape))
        outlist = [outm.type()]
        return Apply(self, inlist, outlist)

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

        if len(self.data_mask) > 0:
            velocities[num.array(self.data_mask), :] = 0.0

        z[0] = velocities

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        return [(len(self.lats), 3)]


class StrainRateTensor(tt.Op):
    """
    pytensorOp for internal block deformation through 2d area strain rate tensor.

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

    def make_node(self, inputs):
        inlist = []

        self.fixed_values = OrderedDict()
        self.varnames = []

        for k, v in inputs.items():
            varname = k.split("_")[-1]  # split of dataset naming
            if isinstance(v, tt.TensorVariable):
                self.varnames.append(varname)
                inlist.append(tt.as_tensor_variable(v))
            else:
                self.fixed_values[varname] = v

        outm_shape = self.infer_shape()[0]

        outm = tt.as_tensor_variable(num.zeros(outm_shape))
        outlist = [outm.type()]
        return Apply(self, inlist, outlist)

    def perform(self, node, inputs, output):
        z = output[0]

        point = {vname: i for vname, i in zip(self.varnames, inputs)}
        point.update(self.fixed_values)

        exx = point["exx"]  # tensor params
        eyy = point["eyy"]
        exy = point["exy"]
        rotation = point["rotation"]

        v_xyz = heart.velocities_from_strain_rate_tensor(
            num.array(self.lats),
            num.array(self.lons),
            exx=exx,
            eyy=eyy,
            exy=exy,
            rotation=rotation,
        )

        if self.data_mask:
            v_xyz[num.array(self.data_mask), :] = 0.0

        z[0] = v_xyz

    def infer_shape(self, fgraph=None, node=None, input_shapes=None):
        return [(self.ndata, 3)]
