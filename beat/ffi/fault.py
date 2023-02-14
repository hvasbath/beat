import copy
import os
from collections import OrderedDict, namedtuple
from logging import getLogger

import numpy as num
from matplotlib import pyplot as plt
from pyrocko.gf.seismosizer import Cloneable
from pyrocko.guts import Dict, Float, Int, List, Object, dump, load
from pyrocko.moment_tensor import moment_to_magnitude
from pyrocko.orthodrome import latlon_to_ne_numpy, ne_to_latlon
from pyrocko.plot import mpl_papersize
from pyrocko.util import ensuredir
from scipy.linalg import block_diag, svd
from theano import shared

from beat.config import (
    ResolutionDiscretizationConfig,
    UniformDiscretizationConfig,
    discretization_dir_name,
    fault_geometry_name,
)
from beat.fast_sweeping import fast_sweep
from beat.heart import velocities_from_pole
from beat.models.laplacian import (
    get_smoothing_operator_correlated,
    get_smoothing_operator_nearest_neighbor,
)
from beat.utility import (
    Counter,
    check_point_keys,
    distances,
    dump_objects,
    find_elbow,
    kmtypes,
    list2string,
    load_objects,
    mod_i,
    positions2idxs,
    rotate_coords_plane_normal,
    split_off_list,
    split_point,
    update_source,
)

from .base import geo_construct_gf_linear_patches, get_backend

logger = getLogger("ffi.fault")


d2r = num.pi / 180.0
r2d = 180.0 / num.pi


__all__ = [
    "FaultGeometry",
    "FaultOrdering",
    "discretize_sources",
    "get_division_mapping",
    "optimize_discretization",
    "optimize_damping",
    "ResolutionDiscretizationResult",
    "write_fault_to_pscmp",
    "euler_pole2slips",
    "backslip2coupling",
]


slip_directions = {
    "uparr": {"slip": 1.0, "rake": 0.0},
    "uperp": {"slip": 1.0, "rake": -90.0},
    "utens": {"slip": 1.0, "rake": 0.0, "opening_fraction": 1.0},
}


PatchMap = namedtuple("PatchMap", "count, slc, shp, npatches, indexmap")


km = 1000.0


class FaultGeometry(Cloneable):
    """
    Object to construct complex fault-geometries with several subfaults.
    Subfaults have uniform patch discretization.
    Stores information for subfault geometries and
    inversion variables (e.g. slip-components).
    Yields patch objects for requested subfault, dataset and component.

    Parameters
    ----------
    datatypes : list
        of str of potential dataset fault geometries to be stored
    components : list
        of str of potential inversion variables (e.g. slip-components) to
        be stored
    ordering : :class:`FaultOrdering`
        comprises patch information related to subfaults
    config : :class: `config.DiscretizationConfig`
    """

    def __init__(self, datatypes, components, ordering, config=None):
        self.datatypes = datatypes
        self.components = components
        self._ext_sources = OrderedDict()
        self._discretized_patches = OrderedDict()
        self._model_resolution = None
        self.ordering = ordering
        self.config = config

    def __str__(self):
        s = """
Complex Fault Geometry
number of subfaults: %i
total number of patches: %i """ % (
            self.nsubfaults,
            self.npatches,
        )
        return s

    def _check_datatype(self, datatype):
        if datatype not in self.datatypes:
            raise TypeError('Datatype "%s" not included in FaultGeometry' % datatype)

    def _check_component(self, component):
        if component not in self.components:
            raise TypeError("Component not included in FaultGeometry")

    def _check_index(self, index):
        if index > self.nsubfaults - 1:
            raise TypeError("Subfault with index %i not defined!" % index)

    def set_model_resolution(self, model_resolution):
        self._model_resolution = model_resolution

    def get_model_resolution(self):
        if hasattr(self, "_model_resolution"):
            if self._model_resolution is None:
                logger.warning(
                    "Model resolution matrix has not been calculated! "
                    "Please run beat.ffi.optimize_discretization on this fault! "
                    "Returning None!"
                )
            return self._model_resolution
        else:
            return None

    def get_subfault_key(self, index, datatype, component):

        if datatype is not None:
            self._check_datatype(datatype)
        else:
            datatype = self.datatypes[0]

        if component is not None:
            self._check_component(component)
        else:
            component = self.components[0]

        self._check_index(index)

        return datatype + "_" + component + "_" + str(index)

    def setup_subfaults(self, datatype, component, ext_sources, replace=False):

        if len(ext_sources) != self.nsubfaults:
            raise FaultGeometryError("Setup does not match fault ordering!")

        for i, source in enumerate(ext_sources):
            source_key = self.get_subfault_key(i, datatype, component)

            if source_key not in list(self._ext_sources.keys()) or replace:
                self._ext_sources[source_key] = copy.deepcopy(source)
            else:
                raise FaultGeometryError("Subfault already specified in geometry!")

    def _assign_datatype(self, datatype=None):
        if datatype is None:
            return self.datatypes[0]
        else:
            return datatype

    def _assign_component(self, component=None):
        if component is None:
            return self.components[0]
        else:
            return component

    def iter_subfaults(self, idxs=None, datatype=None, component=None):
        """
        Iterator over subfaults.

        Parameters
        ----------
        idxs : tuple
            start and end-index of subfaults to iterate over
        """
        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        if idxs is None:
            idxs = [0, self.nsubfaults]

        for i in range(*idxs):
            yield self.get_subfault(index=i, datatype=datatype, component=component)

    def get_subfault(self, index, datatype=None, component=None):

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key in list(self._ext_sources.keys()):
            return self._ext_sources[source_key]
        else:
            raise FaultGeometryError("Requested subfault not defined!")

    def get_all_subfaults(self, datatype=None, component=None):
        """
        Return list of all reference faults
        """
        subfaults = []
        for i in range(self.nsubfaults):
            subfaults.append(
                self.get_subfault(index=i, datatype=datatype, component=component)
            )

        return subfaults

    def set_subfault_patches(self, index, patches, datatype, component, replace=False):

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key not in list(self._discretized_patches.keys()) or replace:
            self._discretized_patches[source_key] = copy.deepcopy(patches)
        else:
            raise FaultGeometryError(
                "Discretized Patches already specified in geometry!"
            )

    def get_subfault_patches(self, index, datatype=None, component=None):
        """
        Get all Patches to a subfault in the geometry.

        Parameters
        ----------
        index : int
            to subfault
        datatype : str
            to return 'seismic' or 'geodetic'
        """
        self._check_index(index)

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key in list(self._discretized_patches.keys()):
            return self._discretized_patches[source_key]
        else:
            raise FaultGeometryError("Requested Patches not defined!")

    def get_all_patches(self, datatype=None, component=None):
        """
        Get all RectangularSource patches for the full complex fault.

        Parameters
        ----------
        datatype : str
            'geodetic' or 'seismic'
        component : str
            slip component to return may be %s
        """ % list2string(
            slip_directions.keys()
        )

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        patches = []
        for i in range(self.nsubfaults):
            patches += self.get_subfault_patches(
                i, datatype=datatype, component=component
            )

        return patches

    def get_subfault_patch_moments(
        self, index, slips=None, store=None, target=None, datatype="seismic"
    ):
        """
        Get the seismic moments on each Patch of the complex fault

        Parameters
        ----------
        slips : list or array-like
            of slips on each fault patch, if not provided uses in-place values
        store : string
            greens function store to use for velocity model extraction
        target : :class:`pyrocko.gf.targets.Target`
            with interpolation method to use for GF interpolation
        datatype : string
            which RectangularSOurce patches to extract
        """

        moments = []
        for i, rs in enumerate(
            self.get_subfault_patches(index=index, datatype=datatype, component="uparr")
        ):
            if slips is not None:
                rs.update(slip=slips[i])

            pm = rs.get_moment(target=target, store=store)
            moments.append(pm)

        return moments

    def get_moment(self, point=None, store=None, target=None, datatype="geodetic"):
        """
        Get total moment of the fault.
        """
        moments = []
        for index in range(self.nsubfaults):
            slips = self.get_total_slip(index, point)

            sf_moments = self.get_subfault_patch_moments(
                index=index, slips=slips, store=store, target=target, datatype=datatype
            )
            moments.extend(sf_moments)

        return num.array(moments).sum()

    def get_magnitude(self, point=None, store=None, target=None, datatype="geodetic"):
        """
        Get total moment magnitude after Hanks and Kanamori 1979
        """
        return moment_to_magnitude(
            self.get_moment(point=point, store=store, target=target, datatype=datatype)
        )

    def get_total_slip(self, index=None, point={}, components=None):
        """
        Get total slip on patches summed over components.
        """
        if components is None:
            components = self.components

        if index is None:
            npatches = self.npatches
        else:
            npatches = self.subfault_npatches[index]

        slips = num.zeros(npatches)
        for comp in components:
            slips += self.var_from_point(index=index, point=point, varname=comp) ** 2

        return num.sqrt(slips)

    def get_subfault_patch_stfs(
        self, index, durations, starttimes, store=None, target=None, datatype="seismic"
    ):
        """
        Get the seismic moments on each Patch of the complex fault

        Parameters
        ----------
        index : list or int
            of subfault(s) to request
        durations : list or array-like
            of slips on each fault patch
        store : string
            greens function store to use for velocity model extraction
        target : :class:`pyrocko.gf.targets.Target`
            with interpolation method to use for GF interpolation
        datatype : string
            which RectangularSOurce patches to extract
        """

        patch_times = []
        patch_amplitudes = []

        if isinstance(index, list):
            pass
        else:
            index = [index]

        for idx in index:
            for i, rs in enumerate(
                self.get_subfault_patches(index=idx, datatype=datatype)
            ):

                if starttimes.size != self.subfault_npatches[idx]:
                    starttimes_idx = self.vector2subfault(index=idx, vector=starttimes)
                    durations_idx = self.vector2subfault(index=idx, vector=durations)
                else:
                    starttimes_idx = starttimes
                    durations_idx = durations

                rs.stf.duration = durations_idx[i]
                times, amplitudes = rs.stf.discretize_t(
                    store.config.deltat, starttimes_idx[i]
                )

                patch_times.append(times)
                patch_amplitudes.append(amplitudes)

        return patch_times, patch_amplitudes

    def get_subfault_moment_rate_function(self, index, point, target, store):

        deltat = store.config.deltat
        slips = self.get_total_slip(index, point, components=["uparr", "uperp"])
        starttimes = self.point2starttimes(point, index=index).ravel()
        tmin = num.floor((starttimes.min() / deltat)) * deltat
        tmax = (
            num.ceil((starttimes.max() + point["durations"].max()) / deltat) + 1
        ) * deltat
        durations = self.vector2subfault(index, point["durations"])

        mrf_times = num.arange(tmin, tmax, deltat)
        mrf_rates = num.zeros_like(mrf_times)

        moments = self.get_subfault_patch_moments(
            index=index, slips=slips, store=store, target=target, datatype="seismic"
        )

        patch_times, patch_amplitudes = self.get_subfault_patch_stfs(
            index=index,
            durations=durations,
            starttimes=starttimes,
            store=store,
            target=target,
            datatype="seismic",
        )

        for m, pt, pa in zip(moments, patch_times, patch_amplitudes):
            tmoments = pa * m
            slc = slice(
                int((pt.min() - tmin) / deltat), int((pt.max() - tmin) / deltat + 1)
            )
            mrf_rates[slc] += tmoments

        return mrf_rates, mrf_times

    def get_moment_rate_function(self, index, point, target, store):

        if isinstance(index, list):
            pass
        else:
            index = [index]

        sf_rates = []
        sf_times = []
        for idx in index:
            rates, times = self.get_subfault_moment_rate_function(
                index=idx, point=point, target=target, store=store
            )
            sf_rates.append(rates)
            sf_times.append(times)

        # add subfault MRFs to total function
        min_times = min(map(num.min, sf_times))
        max_times = max(map(num.max, sf_times))

        deltat = store.config.deltat
        mrf_times = num.arange(min_times, max_times + deltat, deltat)
        mrf_rates = num.zeros_like(mrf_times)
        for sf_rate, sf_time in zip(sf_rates, sf_times):
            slc = slice(
                int((sf_time.min() - min_times) / deltat),
                int((sf_time.max() - min_times) / deltat + 1),
            )
            mrf_rates[slc] += sf_rate

        return mrf_rates, mrf_times

    def get_rupture_geometry(
        self, point, target, store=None, event=None, datatype="geodetic"
    ):
        def duplicate_property(array):
            ndims = len(array.shape)
            if ndims == 1:
                return num.hstack((array, array))
            elif ndims == 2:
                return num.vstack((array, array))
            else:
                raise TypeError("Only 1-2d data supported!")

        def patches2vertices(patches):
            verts = []
            for patch in patches:
                patch.anchor = "top"
                xyz = patch.outline()
                latlon = num.ones((5, 2)) * num.array([patch.lat, patch.lon])
                patchverts = num.hstack((latlon, xyz))
                verts.append(patchverts[:-1, :])  # last vertex double

            return num.vstack(verts)

        slips = self.get_total_slip(index=None, point=point)

        if datatype == "seismic":
            durations = point["durations"]
            deltat = store.config.deltat

            sts = []
            indexs = list(range(self.nsubfaults))
            for index in indexs:
                sts.append(self.point2starttimes(point, index=index).ravel())

            starttimes = num.hstack(sts)
            tmax = (
                num.ceil((starttimes.max() + durations.max()) / deltat) + 1
            ) * deltat
            tmin = num.floor(starttimes.min() / deltat) * deltat

            srf_times = num.arange(tmin, tmax, deltat)
            srf_slips = num.zeros((slips.size, srf_times.size))

            patch_times, patch_amplitudes = self.get_subfault_patch_stfs(
                index=indexs,
                durations=durations,
                starttimes=starttimes,
                store=store,
                target=target,
                datatype=datatype,
            )

            assert slips.size == len(patch_times)

            for i, (slip, pt, pa) in enumerate(
                zip(slips, patch_times, patch_amplitudes)
            ):
                tslips = pa * slip
                slc = slice(
                    int((pt.min() - tmin) / deltat), int((pt.max() - tmin) / deltat + 1)
                )
                srf_slips[i, slc] += tslips

            sub_headers = tuple([str(i) for i in num.arange(srf_times.size)])
            coupling = None

        elif datatype == "geodetic":
            srf_slips = slips.ravel()
            srf_times = num.zeros(1)
            sub_headers = []

            has_pole, _ = check_point_keys(point, phrase="*_pole_lat")
            if has_pole:
                logger.info("Found Euler pole in point also exporting coupling ...!")
                euler_slips = euler_pole2slips(point=point, fault=self, event=event)
                coupling = backslip2coupling(point, euler_slips)
            else:
                coupling = None
        else:
            logger.warning(
                "Datatype %s is not supported for rupture geometry!" % datatype
            )
            return None

        ncorners = 4

        vertices = patches2vertices(self.get_all_patches(datatype))

        outlines = []
        for sf in self.iter_subfaults():
            outlines.append(patches2vertices([sf]))

        faces1 = num.arange(ncorners * self.npatches, dtype="int64").reshape(
            self.npatches, ncorners
        )
        faces2 = num.fliplr(faces1)
        faces = num.vstack((faces1, faces2))

        srf_slips = duplicate_property(srf_slips)

        from pyrocko.model import Geometry

        geom = Geometry(times=srf_times, event=event)
        geom.setup(vertices, faces, outlines=outlines)
        geom.add_property((("slip", "float64", sub_headers)), srf_slips)
        if coupling is not None:
            coupling = duplicate_property(coupling)
            geom.add_property((("coupling", "float64", sub_headers)), coupling)
        return geom

    def get_patch_indexes(self, index):
        """
        Return indexes for sub-fault patches that translate to the solution
        array.

        Parameters
        ----------
        index : int
            to the sub-fault

        Returns
        -------
        slice : slice
            to the solution array that is being extracted from the related
            :class:`pymc3.backends.base.MultiTrace`
        """
        self._check_index(index)
        return slice(
            self.cum_subfault_npatches[index], self.cum_subfault_npatches[index + 1]
        )

    def vector2subfault(self, index, vector):
        sf_patch_indexs = self.cum_subfault_npatches[index : index + 2]
        return vector[sf_patch_indexs[0] : sf_patch_indexs[1]]

    def point2starttimes(self, point, index=0):
        """
        Calculate starttimes for point in solution space for given subfault.
        """

        nuc_dip = point["nucleation_dip"][index]
        nuc_strike = point["nucleation_strike"][index]
        time = point["time"][index]

        velocities = self.vector2subfault(index, point["velocities"])

        nuc_dip_idx, nuc_strike_idx = self.fault_locations2idxs(
            index, positions_dip=nuc_dip, positions_strike=nuc_strike, backend="numpy"
        )

        return (
            self.get_subfault_starttimes(index, velocities, nuc_dip_idx, nuc_strike_idx)
            + time
        )

    def var_from_point(self, index=None, point={}, varname=None):

        try:
            rv = point[varname]
        except KeyError:
            rv = num.zeros(self.npatches)
            logger.debug(
                "Variable %s is not contained in point returning" " zeros!" % varname
            )

        if index is not None:
            return self.vector2subfault(index, rv)
        else:
            return rv

    def point2sources(self, point, events=[]):
        """
        Return source objects (patches) updated by parameters from point.

        Parameters
        ----------
        point : dict
            of numpy arrays of random variables
        events : list
            of :class:`pyrocko.model.Event, their times are reference times
            for the subfault patch times, should have either length of 1 or
            length equal to nsubfaults
        """
        nevents = len(events)
        if nevents:
            assert nevents == 1 or nevents == self.nsubfaults

        if "durations" in point:
            datatype = "seismic"
        else:
            datatype = "geodetic"

        sources = []
        for index in range(self.nsubfaults):
            try:
                sf = self.get_subfault(index, datatype=datatype, component="uparr")
                component = "uparr"
            except TypeError:
                sf = self.get_subfault(index, datatype=datatype, component="utens")
                component = "utens"

            sf_patches = self.get_subfault_patches(
                index, datatype=datatype, component=component
            )

            ucomps = {}
            for comp in slip_directions.keys():
                ucomps[comp] = self.var_from_point(index, point, comp)

            slips = self.get_total_slip(index, point)
            rakes = num.arctan2(-ucomps["uperp"], ucomps["uparr"]) * r2d + sf.rake
            opening_fractions = ucomps["utens"] / slips

            sf_point = {
                "slip": slips,
                "rake": rakes,
                "opening_fraction": opening_fractions,
            }

            try:
                durations = point["durations"]
                starttimes = self.point2starttimes(point, index=index).ravel()
                if nevents > 1:
                    starttimes += events[index].time
                else:
                    starttimes += events[0].time

                sf_point.update({"time": starttimes, "duration": durations})
            except KeyError:
                pass

            patch_points = split_point(sf_point)
            assert len(patch_points) == len(sf_patches)

            for patch, patch_point in zip(sf_patches, patch_points):
                update_source(patch, **patch_point)

            sources.extend(sf_patches)

        return sources

    def get_subfault_starttimes(
        self, index, rupture_velocities, nuc_dip_idx, nuc_strike_idx
    ):
        """
        Get maximum bound of start times of extending rupture along
        the sub-fault.

        Parameters
        ----------
        index : int
            index to the subfault
        rupture_velocities : :class:`numpy.NdArray`
            of rupture velocities for each patch, (N x 1) for N patches [km/s]
        nuc_dip_idx : int
            rupture nucleation idx to patch in dip-direction
        nuc_strike_idx : int
            rupture nucleation idx to patch in strike-direction
        """
        self._check_index(index)
        npw, npl = self.ordering.get_subfault_discretization(index)
        slownesses = 1.0 / rupture_velocities.reshape((npw, npl))

        start_times = fast_sweep.get_rupture_times_numpy(
            slownesses,
            self.ordering.patch_sizes_dip[index],
            n_patch_strike=npl,
            n_patch_dip=npw,
            nuc_x=nuc_strike_idx,
            nuc_y=nuc_dip_idx,
        )
        return start_times

    def get_event_relative_patch_centers(self, event, index=None, datatype=None):
        """
        Returns list of arrays of requested attributes.
        If attributes have several fields they are concatenated to 2d arrays

        Parameters
        ----------
        index: int or list of ints

        Returns
        -------
        :class:`numpy.Ndarray`
            (n_patches x 3)
        """
        if datatype is None:
            datatype = self._assign_datatype()

        if index is None:
            subfault_idxs = list(range(self.nsubfaults))

        centers = self.get_subfault_patch_attributes(
            subfault_idxs, datatype, attributes=["center"]
        )

        lats, lons = self.get_subfault_patch_attributes(
            subfault_idxs, datatype, attributes=["lat", "lon"]
        )

        north_shifts_wrt_event, east_shifts_wrt_event = latlon_to_ne_numpy(
            event.lat, event.lon, lats, lons
        )

        centers[:, 0] += east_shifts_wrt_event / km
        centers[:, 1] += north_shifts_wrt_event / km
        return centers

    def get_smoothing_operator(self, event, correlation_function="nearest_neighbor"):
        """
        Get second order Laplacian smoothing operator.

        This is being used to smooth the slip-distribution
        in the optimization.


        Returns
        -------
        :class:`numpy.Ndarray`
            (n_patch_strike + n_patch_dip) x (n_patch_strike + n_patch_dip)
        """

        if correlation_function == "nearest_neighbor":
            if isinstance(self.config, UniformDiscretizationConfig):
                Ls = []
                for ns in range(self.nsubfaults):
                    self._check_index(ns)
                    npw, npl = self.ordering.get_subfault_discretization(ns)
                    # no smoothing across sub-faults!
                    L = get_smoothing_operator_nearest_neighbor(
                        n_patch_strike=npl,
                        n_patch_dip=npw,
                        patch_size_strike=self.ordering.patch_sizes_strike[ns],
                        patch_size_dip=self.ordering.patch_sizes_dip[ns],
                    )
                    Ls.append(L)
                return block_diag(*Ls)
            else:
                raise InvalidDiscretizationError(
                    "Nearest neighbor correlation Laplacian is only "
                    'available for "uniform" discretization! Please change'
                    " either correlation_function or the discretization."
                )

        else:
            centers = self.get_event_relative_patch_centers(event)
            return get_smoothing_operator_correlated(centers, correlation_function)

    def get_subfault_patch_attributes(
        self, index, datatype=None, component=None, attributes=[""]
    ):
        """
        Returns list of arrays of requested attributes.
        If attributes have several fields they are concatenated to 2d arrays

        Parameters
        ----------
        index: int or list of ints
        """
        if isinstance(index, list):
            patches = []
            for i in index:
                patches += self.get_subfault_patches(i, datatype, component)
        else:
            patches = self.get_subfault_patches(index, datatype, component)

        ats_wanted = []
        for attribute in attributes:
            dummy = [getattr(patch, attribute) for patch in patches]
            if isinstance(dummy[0], num.ndarray):
                dummy = num.vstack(dummy)
            else:
                dummy = num.array(dummy)

            if attribute in kmtypes:
                dummy = dummy / km

            ats_wanted.append(dummy)

        if len(attributes) > 1:
            return ats_wanted
        else:
            return ats_wanted[0]

    def fault_locations2idxs(
        self, index, positions_dip, positions_strike, backend="numpy"
    ):
        """
        Return patch indexes for given location on the fault.

        Parameters
        ----------
        index : int
            to subfault
        positions_dip : :class:`numpy.NdArray` float
            of positions in dip direction of the fault [km]
        positions_strike : :class:`numpy.NdArray` float
            of positions in strike direction of the fault [km]
        backend : str
            which implementation backend to use [numpy/theano]
        """
        backend = get_backend(backend)
        dipidx = positions2idxs(
            positions=positions_dip,
            cell_size=self.ordering.patch_sizes_dip[index],
            backend=backend,
        )
        strikeidx = positions2idxs(
            positions=positions_strike,
            cell_size=self.ordering.patch_sizes_strike[index],
            backend=backend,
        )
        return dipidx, strikeidx

    def patchmap(self, index, dipidx, strikeidx):
        """
        Return mapping of strike and dip indexes to patch index.
        """
        return self.ordering.vmap[index].indexmap[dipidx, strikeidx]

    def spatchmap(self, index, dipidx, strikeidx):
        """
        Return mapping of strike and dip indexes to patch index.
        """
        return self.ordering.smap[index][dipidx, strikeidx]

    @property
    def nsubfaults(self):
        return len(self.ordering.vmap)

    @property
    def subfault_npatches(self):
        if len(self._discretized_patches) > 0:
            npatches = []
            for index in range(self.nsubfaults):
                key = self.get_subfault_key(index, datatype=None, component=None)
                try:
                    patches = self._discretized_patches[key]
                    npatches.append(len(patches))
                except KeyError:
                    logger.debug("Sub-fault %i not discretized yet" % index)
                    npatches.append(0)

            return npatches
        else:
            return [0 for _ in range(self.nsubfaults)]

    @property
    def cum_subfault_npatches(self):
        return num.cumsum([0] + self.subfault_npatches)

    @property
    def npatches(self):
        return sum(self.subfault_npatches)

    @property
    def needs_optimization(self):
        return isinstance(self.config, ResolutionDiscretizationConfig)

    @property
    def is_discretized(self):
        return True if self.npatches else False

    def get_derived_parameters(self, point=None, store=None, target=None, event=None):

        has_pole, _ = check_point_keys(point, phrase="*_pole_lat")
        if has_pole:
            euler_slips = euler_pole2slips(point=point, fault=self, event=event)
            coupling = backslip2coupling(point, euler_slips)
        else:
            coupling = []

        magnitude = self.get_magnitude(point=point, store=store, target=target)
        return num.hstack([magnitude, coupling])


def write_fault_to_pscmp(
    filename, fault, point=None, force=False, export_patch_idxs=False
):
    """
    Dump the fault geometry to ascii file according to the pscmp format.
    """
    from beat import info

    if fault.needs_optimization and not export_patch_idxs:
        raise TypeError(
            "PSCMP only supports uniform discretized rectangular sources, "
            "cannot dump irregularly discretized sources"
        )

    if not fault.is_discretized:
        raise TypeError("Fault needs to be discretized for export!")

    def sf_to_string(index, subfault, np_strike, np_dip, time_days=0.0):
        ul_lat, ul_lon = sf.outline("latlon")[0, :]
        return "{}    {} {} {} {} {} {} {} {} {} {}\n".format(
            index + 1,
            ul_lat,
            ul_lon,
            sf.depth / km,
            sf.length / km,
            sf.width / km,
            sf.strike,
            sf.dip,
            np_strike,
            np_dip,
            time_days,
        )

    def get_template(nsubfaults):
        return (
            """
#===============================================================================
# RECTANGULAR SUBFAULTS
# =====================
# 1. number of subfaults (<= NSMAX in pscglob.h), latitude [deg] and east
#    longitude [deg] of the regional reference point as  origin of the Cartesian
#    coordinate system: ns, lat0, lon0
#
# 2. parameters for the 1. rectangular subfault: geographic coordinates
#    (O_lat, O_lon) [deg] and O_depth [km] of the local reference point on
#    the present fault plane, length (along strike) [km] and width (along down
#    dip) [km], strike [deg], dip [deg], number of equi-size fault
#    patches along the strike (np_st) and along the dip (np_di) (total number of
#    fault patches = np_st x np_di), and the start time of the rupture; the
#    following data lines describe the slip distribution on the present sub-
#    fault:
#
#    pos_s[km]  pos_d[km]  slip_along_strike[m]  slip_along_dip[m]  opening[m]
#
#    where (pos_s,pos_d) defines the position of the center of each patch in
#    the local coordinate system with the origin at the reference point:
#    pos_s = distance along the length (positive in the strike direction)
#    pos_d = distance along the width (positive in the down-dip direction)
#
#
# 3. ... for the 2. subfault ...
# ...
#                   N
#                  /
#                 /| strike
#                +------------------------
#                |\        p .            \ W
#                :-\      i .              \ i
#                |  \    l .                \ d
#                :90 \  S .                  \ t
#                |-dip\  .                    \ h
#                :     \. | rake               \
#                Z      -------------------------
#                              L e n g t h
#
#    Note that a point inflation can be simulated by three point opening
#    faults (each causes a third part of the volume of the point inflation)
#    with orientation orthogonal to each other. the results obtained should
#    be multiplied by a scaling factor 3(1-nu)/(1+nu), where nu is the Poisson
#    ratio at the source. The scaling factor is the ratio of the seismic
#    moment (energy) of an inflation source to that of a tensile source inducing
#    a plate opening with the same volume change.
#===============================================================================
# n_faults
#-------------------------------------------------------------------------------
%i
#-------------------------------------------------------------------------------
# n   O_lat   O_lon    O_depth length  width strike dip   np_st np_di start_time
# [-] [deg]   [deg]    [km]    [km]     [km] [deg]  [deg] [-]   [-]   [day]
#     pos_s   pos_d    slp_stk slp_dip open
#     [km]    [km]     [m]     [m]     [m]
#-------------------------------------------------------------------------------
"""
            % nsubfaults
        )

    # get slip components from result point
    uparr = point["uparr"]
    try:
        uperp = point["uperp"]
    except KeyError:
        logger.info("No uperp component in solution setting to zero!")
        uperp = num.zeros_like(uparr)

    # open file and write header
    if not os.path.exists(filename) or force:
        logger.info(
            "Writing fault geometry to pscmp formatted text" " under: \n %s" % filename
        )
        with open(filename, "wb") as fh:
            header = (
                "# BEAT version %s complex fault geometry \n"
                "# for use with PSCMP from Wang et al. 2008\n"
                "#-----------------------------------------\n" % info.version
            )
            fh.write(header.encode("ascii"))
            fh.write(get_template(fault.nsubfaults).encode("ascii"))

    else:
        raise IOError("File %s exists! Please use --force to overwrite!" % filename)

    # assemble fault geometry
    for index, sf in enumerate(fault.iter_subfaults()):
        npw, npl = fault.ordering.get_subfault_discretization(index)
        subfault_string = sf_to_string(index, sf, np_strike=npl, np_dip=npw)

        # write subfault info
        with open(filename, mode="a+") as fh:
            fh.write(subfault_string)

        centers = fault.get_subfault_patch_attributes(index, attributes=["center"])
        rot_centers = rotate_coords_plane_normal(centers, sf)[:, 1::-1]
        rot_centers[:, 1] -= sf.width / km
        rot_centers[:, 1] *= -1.0

        uparr_sf = fault.vector2subfault(index, uparr)
        uperp_sf = fault.vector2subfault(index, uperp)

        angles = num.arctan2(-uperp_sf, uparr_sf) * r2d + sf.rake
        slips = num.sqrt(uparr_sf**2 + uperp_sf**2)

        strike_slips = num.atleast_2d(num.cos(angles * d2r)) * slips
        dip_slips = num.atleast_2d(num.sin(angles * d2r)) * slips
        opening = num.zeros_like(dip_slips)

        outarray = num.hstack([rot_centers, strike_slips.T, dip_slips.T, opening.T])

        if export_patch_idxs:
            patch_idxs = num.atleast_2d(
                num.arange(*fault.cum_subfault_npatches[index : index + 2])
            ).T
            print(patch_idxs, outarray)
            outarray = num.hstack([patch_idxs, outarray])

        # write patch info
        with open(filename, mode="a+") as fh:
            num.savetxt(fh, outarray, fmt="%g")


class FaultOrdering(object):
    """
    A mapping of source patches to the arrays of optimization results.
    For faults with uniform gridsize.

    Parameters
    ----------
    npls : list
        of number of patches in strike-direction
    npws : list
        of number of patches in dip-direction
    patch_sizes_strike : list of floats
        patch size in strike-direction [km]
    patch_sizes_dip : list of floats
        patch size in dip-direction [km]
    """

    def __init__(self, npls, npws, patch_sizes_strike, patch_sizes_dip):

        self.patch_sizes_dip = patch_sizes_dip
        self.patch_sizes_strike = patch_sizes_strike
        self.vmap = []
        self.smap = []
        dim = 0
        count = 0

        for npl, npw in zip(npls, npws):
            npatches = npl * npw
            slc = slice(dim, dim + npatches)
            shp = (npw, npl)
            indexes = num.arange(npatches, dtype="int16").reshape(shp)
            self.vmap.append(PatchMap(count, slc, shp, npatches, indexes))
            self.smap.append(
                shared(indexes, name="patchidx_array_%i" % count, borrow=True).astype(
                    "int16"
                )
            )
            dim += npatches
            count += 1

        self.npatches = dim

    def get_subfault_discretization(self, index):
        """
        Return number of patches in strike and dip-directions of a subfault.

        Parameters
        ----------
        index : int
            index to the subfault

        Returns
        -------
        tuple (dip, strike)
            number of patches in fault direction
        """
        return self.vmap[index].shp


class FaultGeometryError(Exception):
    pass


def initialise_fault_geometry(
    config,
    sources=None,
    extension_widths=[0.1],
    extension_lengths=[0.1],
    patch_widths=[5.0],
    patch_lengths=[5.0],
    datatypes=["geodetic"],
    varnames=[""],
):
    """
    Build complex discretized fault.

    Extend sources into all directions and discretize sources into uniformly
    discretized patches. Rounds dimensions to have no half-patches.

    Parameters
    ----------
    config : :class: `config.DiscretizationConfig`
    sources : :class:`sources.RectangularSource`
        Reference plane, which is being extended and
    extension_width : float
        factor to extend source in width (dip-direction)
    extension_length : float
        factor extend source in length (strike-direction)
    patch_width : float
        Width [km] of subpatch in dip-direction
    patch_length : float
        Length [km] of subpatch in strike-direction
    varnames : list
        of str with variable names that are being optimized for

    Returns
    -------
    :class:'FaultGeometry'
    """

    def check_subfault_consistency(a, nsources, parameter):
        na = len(a)
        if na != nsources:
            raise ValueError(
                '"%s" have to be specified for each subfault! Only %i set,'
                " but %i subfaults are configured!" % (parameter, na, nsources)
            )

    for i, (pl, pw) in enumerate(zip(patch_lengths, patch_widths)):
        if pl != pw and "seismic" in datatypes:
            raise ValueError(
                "Finite fault optimization including seismic data does only"
                " support square patches (yet)! Please adjust the"
                " discretization for subfault %i: patch-length: %f"
                " != patch-width %f!" % (i, pl, pw)
            )

    nsources = len(sources)
    if "seismic" in datatypes and nsources > 1:
        logger.warning(
            "Seismic kinematic finite fault optimization does"
            " not support rupture propagation across sub-faults yet!"
        )

    check_subfault_consistency(patch_lengths, nsources, "patch_lengths")
    check_subfault_consistency(patch_widths, nsources, "patch_widths")
    check_subfault_consistency(extension_lengths, nsources, "extension_lengths")
    check_subfault_consistency(extension_widths, nsources, "extension_widths")

    npls = []
    npws = []
    for i, source in enumerate(sources):
        s = copy.deepcopy(source)
        patch_length_m = patch_lengths[i] * km
        patch_width_m = patch_widths[i] * km
        ext_source = s.extent_source(
            extension_widths[i], extension_lengths[i], patch_width_m, patch_length_m
        )

        npls.append(ext_source.get_n_patches(patch_length_m, "length"))

        if extension_lengths[i] == 0.0 and "seismic" in datatypes:
            patch_length = ext_source.length / npls[i] / km
            patch_widths[i] = patch_length
            patch_lengths[i] = patch_length
            logger.warning(
                "Subfault %i length was fixed! Assuring square patches, "
                "widths are changed from %f to %f!"
                % (i, patch_width_m / km, patch_length)
            )

        npws.append(ext_source.get_n_patches(patch_widths[i] * km, "width"))

    ordering = FaultOrdering(
        npls, npws, patch_sizes_strike=patch_lengths, patch_sizes_dip=patch_widths
    )

    fault = FaultGeometry(datatypes, varnames, ordering, config=config)

    for datatype in datatypes:
        logger.info("Discretizing %s source(s)" % datatype)

        for var in varnames:
            logger.info("%s slip component" % var)
            ext_sources = []
            for i, source in enumerate(sources):
                param_mod = copy.deepcopy(slip_directions[var])
                s = copy.deepcopy(source)
                param_mod["rake"] += s.rake
                s.update(**param_mod)
                patch_length_m = patch_lengths[i] * km
                patch_width_m = patch_widths[i] * km
                ext_source = s.extent_source(
                    extension_widths[i],
                    extension_lengths[i],
                    patch_width_m,
                    patch_length_m,
                )

                ext_sources.append(ext_source)
                logger.info("Extended fault(s): \n %s" % ext_source.__str__())

            fault.setup_subfaults(datatype, var, ext_sources)

    return fault


class InvalidDiscretizationError(Exception):

    context = (
        "Resolution based discretizeation" + " is available for geodetic data only! \n"
    )

    def __init__(self, errmess=""):
        self.errmess = errmess

    def __str__(self):
        return "\n%s\n%s" % (self.errmess, self.context)


def discretize_sources(
    config, sources=None, datatypes=["geodetic"], varnames=[""], tolerance=0.5
):
    """
    Create Fault Geometry and do uniform discretization

    Paraameters
    -----------
    config: :class: `config.DiscretizationConfig`
    sources : :class:`sources.RectangularSource`
        Reference plane, which is being extended and
    datatypes: list
        of strings with datatypes
    varnames : list
        of str with variable names that are being optimized for
    tolerance : float
        in [m] max difference between allowed patch length and width

    Returns
    -------
    """

    patch_widths, patch_lengths = config.get_patch_dimensions()

    fault = initialise_fault_geometry(
        config=config,
        sources=sources,
        extension_widths=config.extension_widths,
        extension_lengths=config.extension_lengths,
        patch_widths=patch_widths,
        patch_lengths=patch_lengths,
        datatypes=datatypes,
        varnames=varnames,
    )

    if fault.needs_optimization:
        if "seismic" in datatypes:
            raise InvalidDiscretizationError("Seismic dataset!")

        logger.info("Fault discretization selected to be resolution based.")
    else:
        logger.info("Discretization of Fault uniformly (initial)")
        # uniform discretization
        for component in varnames:
            for datatype in datatypes:
                for index, sf in enumerate(
                    fault.iter_subfaults(datatype=datatype, component=component)
                ):
                    npw, npl = fault.ordering.get_subfault_discretization(index)
                    patches = sf.patches(nl=npl, nw=npw, datatype=datatype)
                    fault.set_subfault_patches(index, patches, datatype, component)

                    patch = patches[0]
                    if (
                        patch.length - patch.width > tolerance
                        and config.extension_lengths[index] == 0.0
                    ):
                        logger.warning(
                            "Patch width %f and patch length %f are not equal "
                            "for subfault %i because extension in length was "
                            "fixed. Please ensure square patches if you want"
                            "to do full kinematic FFI! Static only is fine!"
                            "patch_length = fault_length / n_patches"
                            % (patch.width, patch.length, index)
                        )
                    else:
                        logger.info(
                            "Subfault %i, patch lengths %f [m] and"
                            "patch widths %f [m]." % (index, patch.length, patch.width)
                        )

    return fault


def get_division_mapping(patch_idxs, div_idxs, subfault_npatches):
    """
    Returns index mappings for fault patches to be divided from old to
    new division

    Parameters
    ----------
    patch_idxs : list
        of indexes of patches of old generation
    div_idxs : list
        of indexes of patches of old generation to be divided for new generation
    subfault_npatches : array_like
        of number of patches per subfault, length determines number of subfaults

    Returns
    -------
    old2new : dict
        mapping of old (undivided) patch indexes to new patch indexes
    div2new : dict
        mapping of divided patch indexes to new patch indexes
    new_subfault_npatches : array_like
        with number of patches in each subfault after division
    """
    count = Counter()

    old2new = OrderedDict()
    div2new = OrderedDict()
    new_subfault_npatches = num.zeros_like(subfault_npatches)
    count("sf_idx")
    count("npatches_old")
    count("npatches_new")
    for patch_idx in patch_idxs:

        if patch_idx in div_idxs:
            div2new[count("new")] = count("tot")
            div2new[count("new")] = count("tot")
            count("old")  # count old once for patch removal
            count("npatches_new", 2)
        else:
            old2new[count("old")] = count("tot")
            count("npatches_new")

        if count("npatches_old") == subfault_npatches[count["sf_idx"]]:
            new_subfault_npatches[count["sf_idx"]] = count["npatches_new"]
            count("sf_idx")
            count.reset("npatches_old")
            count.reset("npatches_new")

    return old2new, div2new, new_subfault_npatches


def euler_pole2slips(point, fault, event):
    """
    Get Euler pole rotation imposed slip component in strike-direction of fault.

    Parameters
    ----------
    point : dict
        of numpy arrays of random variables
    fault : :class:`FaultGeometry`
    event : :class:`pyrocko.model.Event`

    Returns
    -------
    ndarray : floats
        of number of slip patches with coupling between 0 and 1,
        0 - no coupling, 1 - full coupling
    """
    datatype = "geodetic"

    has_pole, pole_lat_keys = check_point_keys(point, phrase="*_pole_lat")
    has_pole, pole_lon_keys = check_point_keys(point, phrase="*_pole_lon")
    has_pole, omega_keys = check_point_keys(point, phrase="*_omega")

    npoles = len(pole_lon_keys)
    if has_pole:
        if npoles > 1:
            logger.warning(
                "Found %i poles in result point! "
                "Returning coupling only for first pole!" % npoles
            )

        plat = point[pole_lat_keys[0]]
        plon = point[pole_lon_keys[0]]
        omega = point[omega_keys[0]]
    else:
        raise ValueError("Euler Pole not in result point!")

    subfault_idxs = list(range(fault.nsubfaults))
    strikevectors_enu = fault.get_subfault_patch_attributes(
        subfault_idxs, datatype=datatype, component="uparr", attributes=["strikevector"]
    )

    strikevectors_neu = num.zeros_like(strikevectors_enu)
    strikevectors_neu[:, 0] = strikevectors_enu[:, 1]
    strikevectors_neu[:, 1] = strikevectors_enu[:, 0]

    centers = fault.get_event_relative_patch_centers(event=event)[:, 0:2] * km

    lats, lons = ne_to_latlon(
        lat0=event.lat, lon0=event.lon, north_m=centers[:, 1], east_m=centers[:, 0]
    )

    euler_velocities_neu = velocities_from_pole(
        lats=lats,
        lons=lons,
        pole_lat=plat,
        pole_lon=plon,
        omega=omega,
        earth_shape="ellipsoid",
    )

    return num.abs((euler_velocities_neu * strikevectors_neu).sum(axis=1))


def backslip2coupling(point, euler_slips):
    """
    Transform backslips and Euler pole rotation slips on fault to coupling coefficients.

    Parameters
    ----------
    point : dict
        of numpy arrays of random variables
    """
    try:
        backslips = point["uparr"]
    except KeyError:
        raise ValueError("Parallel slip component not in result point!")

    coupling = backslips / euler_slips
    coupling[coupling < 0.0] = 0.0  # negative slip values mean no coupling
    coupling[coupling > 1.0] = 1.0  # backslip higher than long term rate full coupling
    return coupling * 100  # to percent


def optimize_discretization(
    config,
    fault,
    datasets,
    varnames,
    crust_ind,
    engine,
    targets,
    event,
    force,
    nworkers,
    method="laplacian",
    debug=False,
):
    """
    Resolution based discretization of the fault surfaces

    Parameters
    ----------
    config : :class: `config.DiscretizationConfig`

    References
    ----------
    .. [Atzori2011] Atzori, S. and Antonioli, A. (2011).
        Optimal fault resolution in geodetic inversion of coseismic data
        Geophys. J. Int. (2011) 185, 529–538,
        `link <http://ascelibrary.org/doi: 10.1111/j.1365-246X.2011.04955.x>`__
    .. [Atzori2019] Atzori, S.; Antonioli, A.; Tolomei, C.; De Novellis, V.;
        De Luca, C. and Monterroso, F.
        InSAR full-resolution analysis of the 2017–2018 M > 6 earthquakes in
        Mexico
        Remote Sensing of Environment, 234, 111461,
    """
    from numpy.testing import assert_array_equal

    from beat.plotting import source_geometry

    _available_methods = ("laplacian", "svd")

    if method not in _available_methods:
        raise NotImplementedError(
            "Supported methods: %s, specified: %s"
            % (list2string(_available_methods), method)
        )

    logger.info('Using "%s" for calculation of Resolution', method)

    def sv_vec2matrix(sv_vec, ndata, nparams):
        """
        Transform vector of singular values to matrix (M, N),
        M - data length
        N - number of singular values

        Parameters
        ----------
        sv_vec:
        ndata: number of observations

        Returns
        -------
        array-like (M x N)
        """
        n_sv = sv_vec.size
        Lmat = num.zeros((ndata, nparams))
        Lmat[:n_sv, :n_sv] = num.diag(sv_vec)
        return Lmat

    logger.debug("Optimizing fault discretization based on resolution: ... \n")

    datatype = "geodetic"

    data_east_shifts = []
    data_north_shifts = []
    for dataset in datasets:
        ns, es = dataset.update_local_coords(event)
        data_north_shifts.append(ns / km)
        data_east_shifts.append(es / km)

    data_coords = num.vstack(
        [num.hstack(data_east_shifts), num.hstack(data_north_shifts)]
    ).T

    patch_widths, patch_lengths = config.get_patch_dimensions()
    gfs_comp = []
    for component in varnames:
        for index, sf in enumerate(
            fault.iter_subfaults(datatype=datatype, component=component)
        ):
            npw = sf.get_n_patches(2 * patch_widths[index] * km, "width")
            npl = sf.get_n_patches(2 * patch_lengths[index] * km, "length")
            patches = sf.patches(nl=npl, nw=npw, datatype=datatype)
            fault.set_subfault_patches(index, patches, datatype, component)

        logger.debug(
            "Calculating Green's Functions for %i "
            "initial patches for %s slip-component." % (fault.npatches, component)
        )

        gfs = geo_construct_gf_linear_patches(
            engine=engine,
            datasets=datasets,
            targets=targets,
            patches=fault.get_all_patches("geodetic", component=component),
            nworkers=nworkers,
            return_mapping=True,
        )

        gfs_comp.append(gfs)

    tobedivided = fault.npatches

    sf_div_idxs = []
    for i, sf in enumerate(fault.iter_subfaults()):
        if (
            sf.width / km <= config.patch_widths_min[i]
            or sf.length / km <= config.patch_lengths_min[i]
        ):
            pass
        else:
            sf_div_idxs.extend(
                (
                    num.arange(fault.subfault_npatches[i])
                    + fault.cum_subfault_npatches[i]
                ).tolist()
            )

    generation = 0
    R = None
    patch_data_distance_mins = None  # dummy for first plotting
    fixed_idxs = set()
    while tobedivided:
        logger.info("Discretizing %ith generation \n" % generation)
        gfs_array = []
        subfault_npatches = copy.deepcopy(fault.subfault_npatches)
        if debug:
            source_geometry(
                fault,
                list(fault.iter_subfaults()),
                event=event,
                datasets=datasets,
                values=R,
                title="Resolution",
            )
            source_geometry(
                fault,
                list(fault.iter_subfaults()),
                event=event,
                datasets=datasets,
                values=patch_data_distance_mins,
                title="min distance",
            )

        for gfs_i, component in enumerate(varnames):
            logger.debug("Component %s" % component)

            # iterate over subfaults and divide patches
            old2new, div2new, new_subfault_npatches = get_division_mapping(
                patch_idxs=range(sum(subfault_npatches)),
                div_idxs=sf_div_idxs,
                subfault_npatches=subfault_npatches,
            )

            old_patches = fault.get_all_patches(datatype=datatype, component=component)
            all_divided_patches = []
            logger.debug("Division indexes %s" % (list2string(sf_div_idxs)))
            for div_idx in sf_div_idxs:
                # pull out patch to be divided
                patch = old_patches[div_idx]
                if patch.length >= patch.width:
                    div_patches = patch.patches(nl=2, nw=1, datatype=datatype)
                else:
                    div_patches = patch.patches(nl=1, nw=2, datatype=datatype)

                # add to all patches that need recalculation
                all_divided_patches.extend(div_patches)

            logger.debug(
                "Calculating Green's Functions for %i divided "
                "patches." % len(all_divided_patches)
            )

            # calculate GFs for fault is [npatches, nobservations]
            gfs = geo_construct_gf_linear_patches(
                engine=engine,
                datasets=datasets,
                targets=targets,
                patches=all_divided_patches,
                nworkers=nworkers,
            )
            old_gfs = gfs_comp[gfs_i]

            # assemble new generation of discretization
            new_npatches_total = new_subfault_npatches.sum()
            new_gfs = num.zeros((new_npatches_total, gfs.shape[1]))

            new_patches = [None] * new_npatches_total
            logger.info("Next generation npatches %i" % new_npatches_total)
            for idx_mapping, tpatches, tgfs in [
                (old2new, old_patches, old_gfs),
                (div2new, all_divided_patches, gfs),
            ]:
                # print('gfs', tgfs[:, 0:5])
                for patch_idx, new_idx in idx_mapping.items():
                    new_patches[new_idx] = tpatches[patch_idx]
                    new_gfs[new_idx] = tgfs[patch_idx]

            if False:
                logger.debug("Cross checking gfs ...")
                check_gfs = geo_construct_gf_linear_patches(
                    engine=engine,
                    datasets=datasets,
                    targets=targets,
                    patches=new_patches,
                    nworkers=nworkers,
                )

                assert (new_gfs - check_gfs).sum() == 0

            gfs_array.append(new_gfs.T)  # G(nobs, npatches)

            # register new generation of patches with fault
            for sf_idx, sf_npatches in enumerate(new_subfault_npatches.tolist()):
                sf_patches = split_off_list(new_patches, sf_npatches)
                fault.set_subfault_patches(
                    sf_idx, sf_patches, datatype, component, replace=True
                )

            # new generation of GFs
            gfs_comp[gfs_i] = new_gfs

        # update fixed indexes
        fixed_idxs = set([old2new[idx] for idx in fixed_idxs])

        assert_array_equal(num.array(fault.subfault_npatches), new_subfault_npatches)

        if False:
            fig, axs = plt.subplots(2, 3)
            for i, gfidx in enumerate(
                num.linspace(0, fault.npatches, 6, dtype="int", endpoint=False)
            ):
                ridx, cidx = mod_i(i, 3)
                if ridx < 2:
                    ax = axs[ridx, cidx]
                    im = ax.scatter(
                        datasets[0].lons,
                        datasets[0].lats,
                        10,
                        num.vstack(gfs_array)[:, gfidx],
                        edgecolors="none",
                        cmap=plt.cm.get_cmap("jet"),
                    )
                    ax.set_title("Patch idx %i" % gfidx)

        resolution_matrices = []
        R_diags = []
        for gfs_i, component in enumerate(varnames):
            comp_gfs = gfs_array[gfs_i]

            if method == "svd":
                # Atzori & Antonioli 2011
                # U data-space, L singular values, V model space

                ndata, nparams = comp_gfs.shape
                U, l, V = svd(comp_gfs, full_matrices=True)

                # apply singular value damping
                ldamped_inv = 1.0 / (l + config.epsilon**2)
                Linv = sv_vec2matrix(ldamped_inv, ndata=ndata, nparams=nparams)
                L = sv_vec2matrix(l, ndata=ndata, nparams=nparams)

                # calculate resolution matrix and take trace
                if 0:
                    # for debugging
                    print("full_GFs", comp_gfs.shape)
                    print("V", V.shape)
                    print("l", l.shape)
                    print("L", L.shape)
                    print("Linnv", Linv.shape)
                    print("U", U.shape)

                R = num.diag(num.dot(V.dot(Linv.T).dot(U.T), U.dot(L).dot(V.T)))

            elif method == "laplacian":
                # Atzori et al. 2019 Full Resolution Analysis
                # G(nobs, npatches)
                smoothing_op = (
                    fault.get_smoothing_operator(event, correlation_function="gaussian")
                    * config.epsilon**2
                )

                # weighting makes it not work! dont weight for now
                GG = comp_gfs.T.dot(comp_gfs)
                Gdamped = num.vstack((comp_gfs, smoothing_op))
                GdampedG = Gdamped.T.dot(Gdamped)
                resolution_matrix = num.linalg.inv(GdampedG).dot(GG)
                R = num.diag(resolution_matrix)
                R_diags.append(R)
            else:
                raise NotImplementedError('Method "%s" not supported !' % method)

            resolution_matrices.append(resolution_matrix)

            R_idxs = num.argwhere(R > config.resolution_thresh).ravel().tolist()
            tmp_fixed_idxs = set(
                num.argwhere(R <= config.resolution_thresh).ravel().tolist()
            )
            logger.debug(
                "Patches fixed %s component: %s",
                component,
                list2string(list(tmp_fixed_idxs)),
            )
            fixed_idxs.update(tmp_fixed_idxs)

        # print('R > thresh', R_idxs, R[R_idxs])
        # analysis for further patch division
        sf_div_idxs = []

        width_idxs_max = []
        width_idxs_min = []
        length_idxs_max = []
        length_idxs_min = []
        for i, sf in enumerate(fault.iter_subfaults()):
            widths, lengths = fault.get_subfault_patch_attributes(
                i, datatype, attributes=["width", "length"]
            )

            # select patches that fulfill size requirements
            width_idxs_max += (
                num.argwhere(widths > config.patch_widths_max[i]).ravel()
                + fault.cum_subfault_npatches[i]
            ).tolist()
            length_idxs_max += (
                num.argwhere(lengths > config.patch_lengths_max[i]).ravel()
                + fault.cum_subfault_npatches[i]
            ).tolist()
            width_idxs_min += (
                num.argwhere(widths <= config.patch_widths_min[i]).ravel()
                + fault.cum_subfault_npatches[i]
            ).tolist()
            length_idxs_min += (
                num.argwhere(lengths <= config.patch_lengths_min[i]).ravel()
                + fault.cum_subfault_npatches[i]
            ).tolist()

        # patches that fulfill either size thresholds
        patch_size_ids = set(width_idxs_min + length_idxs_min)

        # remove patches from fixed idxs that are above max size
        above_size_thresh = set(width_idxs_max + length_idxs_max)
        fixed_idxs = fixed_idxs.difference(above_size_thresh)

        # patches above R but below size thresholds & remove fixed patches
        # put in patches that violate max size threshold
        unique_ids = (
            set(R_idxs).difference(patch_size_ids, fixed_idxs).union(above_size_thresh)
        )

        ncandidates = len(unique_ids)

        logger.debug(
            "Found %i candidate(s) for division for "
            " %i subfault(s)" % (ncandidates, fault.nsubfaults)
        )

        mean_R = num.vstack(R_diags).mean(0).ravel()
        if ncandidates:
            subfault_idxs = list(range(fault.nsubfaults))
            widths, lengths = fault.get_subfault_patch_attributes(
                subfault_idxs, datatype, attributes=["width", "length"]
            )

            # calculate division penalties
            uids = num.array(list(unique_ids))
            area_pen = widths * lengths

            # depth penalty
            c1 = []
            for i, sf in enumerate(fault.iter_subfaults()):
                # bdepths = fault.get_subfault_patch_attributes(
                #    i, datatype, attributes=['depth'])
                bdepths = fault.get_subfault_patch_attributes(
                    i, datatype, attributes=["center"]
                )[:, 2]
                c1.extend(
                    num.exp(
                        -config.depth_penalty * bdepths * km / sf.bottom_depth
                    ).tolist()
                )

            c_one_pen = num.array(c1)

            # distance penalties
            centers = fault.get_event_relative_patch_centers(event)[:, :2]
            cand_centers = centers

            patch_data_distances = distances(
                points=data_coords, ref_points=cand_centers
            )
            patch_data_distance_mins = patch_data_distances.min(axis=0)

            c_two_pen = patch_data_distance_mins.min() / patch_data_distance_mins

            # patch- patch penalty
            inter_patch_distances = distances(points=centers, ref_points=cand_centers)

            res_w = mean_R * inter_patch_distances

            c_three_pen = res_w.sum(axis=1) / inter_patch_distances.sum(0)

            rating = area_pen * c_one_pen * c_two_pen * c_three_pen
            rating_idxs = num.array(rating.argsort()[::-1])
            rated_sel = num.array([ridx for ridx in rating_idxs if ridx in uids])

            n_sel = len(rated_sel)
            idxs = rated_sel[range(int(num.ceil(config.alpha * n_sel)))]

            if debug:
                print("above size thresh", above_size_thresh)
                print("---------")
                print("min patch data distances", patch_data_distance_mins)
                print("---------")
                print("R", R)
                print("---------")
                print("depth pen C1", c_one_pen)
                print("distance C2", c_two_pen)
                print("resolution neighbor C3", c_three_pen)
                print("area A", area_pen)
                print("rating", rating)
                print("---------")
                print("rating argsorted", rating_idxs)
                print("unique patches uids", uids)
                print("R select rated", rated_sel)

            logger.debug(
                "Patches: %s of %i subfault(s) are fixed."
                % (list2string(list(fixed_idxs)), fault.nsubfaults)
            )
            logger.debug(
                "Patches: %s of %i subfault(s) are further divided."
                % (list2string(idxs.tolist()), fault.nsubfaults)
            )
            tobedivided = len(idxs)
            sf_div_idxs = copy.deepcopy(idxs)
            sf_div_idxs.sort()
            generation += 1
        else:
            tobedivided = 0

    if debug:
        patches = fault.get_all_patches(datatype, component="uparr")
        minidx, maxidx = mean_R.argmin(), mean_R.argmax()
        print(mean_R.max(), mean_R.min(), maxidx, minidx)
        print("min", patches[minidx])
        print("max", patches[maxidx])

        from matplotlib import pyplot as plt

        print("Smoothing Op", smoothing_op)
        im = plt.matshow(smoothing_op)
        plt.colorbar(im)
        plt.show()

    R_matrix = num.dstack(resolution_matrices).mean(2)
    fault.set_model_resolution(R_matrix)

    logger.info("Finished resolution based fault discretization.")
    logger.info("Quality index for this discretization: %f" % mean_R.mean())
    return fault, mean_R


class ResolutionDiscretizationResult(Object):

    epsilons = List.T(Float.T(), default=[0])
    normalized_rspreads = List.T(Float.T(), default=[1.0])
    faults_npatches = List.T(Int.T(), default=[1])
    optimum = Dict.T(default=dict(), help="Optimum fault discretization parameters")

    def plot(self):

        fig, ax = plt.subplots(1, 1, figsize=mpl_papersize("a6", "landscape"))
        ax.plot(
            num.array(self.epsilons),
            num.array(self.normalized_rspreads),
            "+b",
            markersize=6,
        )

        for epsilon, rspread, npatches in zip(
            self.epsilons, self.normalized_rspreads, self.faults_npatches
        ):
            ax.text(epsilon, rspread, npatches, fontsize=9)
        try:
            ax.plot(
                num.array(self.optimum["epsilon"]),
                num.array(self.optimum["normalized_rspread"]),
                "*r",
                markersize=8,
            )
        except KeyError:
            logger.warning("Discretization result does not contain the optimum")

        ax.set_ylabel("Normalized resolution spread")
        ax.set_xlabel("Epsilon (damping)")
        ax.set_title("Fault resolution based discretization")
        return fig, ax

    def derive_optimum_fault_geometry(self, debug=False):

        data = num.vstack(
            (num.array(self.epsilons), num.array(self.normalized_rspreads))
        ).T

        best_idx, rotated_data = find_elbow(data, rotate_left=True)

        if debug:
            fig, ax = plt.subplots(1, 1, figsize=mpl_papersize("a6", "landscape"))
            ax.plot(rotated_data[:, 0], rotated_data[:, 1], "+b", markersize=6)
            ax.plot(
                rotated_data[best_idx, 0], rotated_data[best_idx, 1], "*r", markersize=8
            )
            plt.show()

        self.optimum = {
            "epsilon": self.epsilons[best_idx],
            "normalized_rspread": self.normalized_rspreads[best_idx],
            "npatches": self.faults_npatches[best_idx],
            "idx": best_idx,
        }


def normalized_resolution_spread(resolution, nparams):
    """
    Get normalized resolution spread after Atzori et al. 2019

    Must be between 0 and 1. The closer to zero the better can the model
    parameters be resolved by the data.
    """
    return num.linalg.norm(resolution - num.eye(nparams)) / nparams


def optimize_damping(
    outdir,
    config,
    fault,
    datasets,
    varnames,
    crust_ind,
    engine,
    targets,
    event,
    force,
    nworkers,
    method="laplacian",
    plot=False,
    figuredir=None,
    debug=False,
):
    """
    Resolution based discretization of the fault surfaces epsilon optimization.

    Parameters
    ----------
    config : :class: `config.DiscretizationConfig`

    References
    ----------
    .. [Atzori2011] Atzori, S. and Antonioli, A. (2011).
        Optimal fault resolution in geodetic inversion of coseismic data
        Geophys. J. Int. (2011) 185, 529–538,
        `link <http://ascelibrary.org/doi: 10.1111/j.1365-246X.2011.04955.x>`__
    .. [Atzori2019] Atzori, S.; Antonioli, A.; Tolomei, C.; De Novellis, V.;
        De Luca, C. and Monterroso, F.
        InSAR full-resolution analysis of the 2017–2018 M > 6 earthquakes in
        Mexico
        Remote Sensing of Environment, 234, 111461,
    """

    discr_dir = os.path.join(outdir, discretization_dir_name)
    ensuredir(discr_dir)
    fault_basename, extension = os.path.splitext(fault_geometry_name)

    epsilons = (
        num.logspace(0, 2, config.epsilon_search_runs, endpoint=True) * config.epsilon
    )
    discretization_results_path = os.path.join(
        discr_dir,
        "resolution_result_%g_%i.yaml" % (config.epsilon, config.epsilon_search_runs),
    )

    logger.info("Calculating discretizations for %s", list2string(epsilons.tolist()))

    model_resolutions = []
    dfaults = []
    for epsilon in epsilons:

        logger.info("Epsilon: %g", epsilon)
        logger.info("--------------")
        fault_discr_path = os.path.join(
            discr_dir, "{}_{}{}".format(fault_basename, epsilon, extension)
        )

        if not os.path.exists(fault_discr_path) or force:
            config.epsilon = epsilon
            dfault, mean_R = optimize_discretization(
                config=config,
                fault=copy.deepcopy(fault),
                datasets=datasets,
                varnames=varnames,
                crust_ind=crust_ind,
                engine=engine,
                targets=targets,
                event=event,
                force=force,
                nworkers=nworkers,
                method="laplacian",
                debug=False,
            )

            logger.info(
                "Storing discretized fault geometry to: " "%s" % fault_discr_path
            )
            dump_objects(fault_discr_path, [dfault])

            # overwrite again with original value
            config.epsilon = epsilons[0]

        elif os.path.exists(fault_discr_path):
            logger.info(
                "Discretized fault geometry for epsilon %s exists! "
                "Use --force to overwrite!" % epsilon
            )
            logger.info("Loading existing discretized fault")
            dfault = load_objects(fault_discr_path)[0]

        dfaults.append(dfault)
        model_resolutions.append(dfault.get_model_resolution())

    logger.info("Calculating normalized resolution spreads ...")
    normalized_rspreads = []
    faults_npatches = []
    for dfault, resolution in zip(dfaults, model_resolutions):
        rspread = normalized_resolution_spread(resolution, nparams=dfault.npatches)
        normalized_rspreads.append(rspread)
        faults_npatches.append(dfault.npatches)

    result = ResolutionDiscretizationResult(
        epsilons=epsilons.tolist(),
        normalized_rspreads=normalized_rspreads,
        faults_npatches=faults_npatches,
    )
    result.derive_optimum_fault_geometry()

    logger.info("Dumping discretization result to: %s", discretization_results_path)
    dump(result, filename=discretization_results_path)
    print(result)

    best_idx = result.optimum["idx"]
    if figuredir is not None:
        logger.info("Plotting tradeoff and discretizations ...")
        from beat.plotting import source_geometry

        outformat = "png"

        # tradeoff
        fig, ax = result.plot()
        outpath = os.path.join(
            figuredir, "discretization_tradeoff_%g.%s" % (epsilons[best_idx], outformat)
        )
        logger.info("Plotting discretization_tradeoff to %s" % outpath)
        fig.savefig(outpath, format=outformat, dpi=300)

        # discretizations
        for i, dfault in enumerate(dfaults):
            fig, ax = source_geometry(
                dfault,
                list(fault.iter_subfaults()),
                event=event,
                values=num.diag(dfault.get_model_resolution()),
                cbounds=(0.5, 1),
                clabel="Resolution",
                datasets=datasets,
                show=False,
            )

            outpath = os.path.join(
                figuredir, "patch_resolutions_%i.%s" % (dfault.npatches, outformat)
            )
            logger.info("Plotting patch resolution to %s" % outpath)
            fig.savefig(outpath, format=outformat, dpi=300)

    return dfaults[best_idx]
