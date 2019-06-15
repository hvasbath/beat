from beat.utility import list2string, positions2idxs, kmtypes
from beat.fast_sweeping import fast_sweep

from beat.config import ResolutionDiscretizationConfig, \
    UniformDiscretizationConfig
from beat.models.laplacian import get_smoothing_operator_correlated, \
    get_smoothing_operator_nearest_neighbor, distances
from .base import get_backend, geo_construct_gf_linear

from pyrocko.gf.seismosizer import Cloneable
from pyrocko.guts import Float, Object, List

import copy
from logging import getLogger
from collections import namedtuple, OrderedDict

import numpy as num

from scipy.linalg import block_diag

from theano import shared


logger = getLogger('ffi.fault')


__all__ = [
    'FaultGeometry',
    'FaultOrdering',
    'discretize_sources',
    'optimize_discretization']


slip_directions = {
    'uparr': {'slip': 1., 'rake': 0.},
    'uperp': {'slip': 1., 'rake': -90.},
    'utensile': {'slip': 0., 'rake': 0., 'opening': 1.}}


PatchMap = namedtuple(
    'PatchMap', 'count, slc, shp, npatches, indexmap')


km = 1000.


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
        self.ordering = ordering
        self.config = config

    def __str__(self):
        s = '''
Complex Fault Geometry
number of subfaults: %i
total number of patches: %i ''' % (
            self.nsubfaults, self.npatches)
        return s

    def _check_datatype(self, datatype):
        if datatype not in self.datatypes:
            raise TypeError(
                'Datatype "%s" not included in FaultGeometry' % datatype)

    def _check_component(self, component):
        if component not in self.components:
            raise TypeError('Component not included in FaultGeometry')

    def _check_index(self, index):
        if index > self.nsubfaults - 1:
            raise TypeError('Subfault with index %i not defined!' % index)

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

        return datatype + '_' + component + '_' + str(index)

    def setup_subfaults(self, datatype, component, ext_sources, replace=False):

        if len(ext_sources) != self.nsubfaults:
            raise FaultGeometryError('Setup does not match fault ordering!')

        for i, source in enumerate(ext_sources):
            source_key = self.get_subfault_key(i, datatype, component)

            if source_key not in list(self._ext_sources.keys()) or replace:
                self._ext_sources[source_key] = copy.deepcopy(source)
            else:
                raise FaultGeometryError(
                    'Subfault already specified in geometry!')

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

    def iter_subfaults(self, datatype=None, component=None):
        """
        Iterator over subfaults.
        """
        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        for i in range(self.nsubfaults):
            yield self.get_subfault(
                index=i, datatype=datatype, component=component)

    def get_subfault(self, index, datatype=None, component=None):

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key in list(self._ext_sources.keys()):
            return self._ext_sources[source_key]
        else:
            raise FaultGeometryError('Requested subfault not defined!')

    def get_all_subfaults(self, datatype=None, component=None):
        """
        Return list of all reference faults
        """
        subfaults = []
        for i in range(self.nsubfaults):
            subfaults.append(
                self.get_subfault(
                    index=i, datatype=datatype, component=component))

        return subfaults

    def set_subfault_patches(
            self, index, patches, datatype, component, replace=False):

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key not in list(self._discretized_patches.keys()) or replace:
            self._discretized_patches[source_key] = copy.deepcopy(patches)
        else:
            raise FaultGeometryError(
                'Discretized Patches already specified in geometry!')

    def get_subfault_patches(
            self, index, datatype=None, component=None):
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
            raise FaultGeometryError('Requested Patches not defined!')

    def get_all_patches(self, datatype=None, component=None):
        """
        Get all RectangularSource patches for the full complex fault.

        Parameters
        ----------
        datatype : str
            'geodetic' or 'seismic'
        component : str
            slip component to return may be %s
        """ % list2string(slip_directions.keys())

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        patches = []
        for i in range(self.nsubfaults):
            patches += self.get_subfault_patches(
                i, datatype=datatype, component=component)

        return patches

    def get_subfault_patch_moments(
            self, index, slips, store=None, target=None, datatype='seismic'):
        """
        Get the seismic moments on each Patch of the complex fault

        Parameters
        ----------
        slips : list or array-like
            of slips on each fault patch
        store : string
            greens function store to use for velocity model extraction
        target : :class:`pyrocko.gf.targets.Target`
            with interpolation method to use for GF interpolation
        datatype : string
            which RectangularSOurce patches to extract
        """

        moments = []
        for i, rs in enumerate(
                self.get_subfault_patches(index=index, datatype=datatype)):
            rs.update(slip=slips[i])
            moments.append(rs.get_moment(target=target, store=store))

        return moments

    def get_subfault_patch_stfs(
            self, index, durations, starttimes,
            store=None, target=None, datatype='seismic'):
        """
        Get the seismic moments on each Patch of the complex fault

        Parameters
        ----------
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

        for i, rs in enumerate(
                self.get_subfault_patches(index=index, datatype=datatype)):

            rs.stf.duration = durations[i]
            times, amplitudes = rs.stf.discretize_t(
                store.config.deltat, starttimes[i])

            patch_times.append(times)
            patch_amplitudes.append(amplitudes)

        return patch_times, patch_amplitudes

    def get_subfault_moment_rate_function(self, index, point, target, store):

        deltat = store.config.deltat
        slips = num.sqrt(point['uparr'] ** 2 + point['uperp'] ** 2)

        starttimes = self.point2starttimes(point, index=index).ravel()
        tmax = (num.ceil(
            (starttimes.max() + point['durations'].max()) / deltat) + 1) * \
            deltat

        mrf_times = num.arange(0., tmax, deltat)
        mrf_rates = num.zeros_like(mrf_times)

        moments = self.get_subfault_patch_moments(
            index=index, slips=slips,
            store=store, target=target, datatype='seismic')

        patch_times, patch_amplitudes = self.get_subfault_patch_stfs(
            index=index, durations=point['durations'], starttimes=starttimes,
            store=store, target=target, datatype='seismic')

        for m, pt, pa in zip(moments, patch_times, patch_amplitudes):
            tmoments = pa * m
            slc = slice(int(pt.min() / deltat), int(pt.max() / deltat + 1))
            mrf_rates[slc] += tmoments

        return mrf_rates, mrf_times

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
        return slice(self.cum_subfault_npatches[index],
                     self.cum_subfault_npatches[index + 1])

    def point2starttimes(self, point, index=0):
        """
        Calculate starttimes for point in solution space.
        """

        nuc_dip = point['nucleation_dip']
        nuc_strike = point['nucleation_strike']
        velocities = point['velocities']
        # TODO make index dependent   !!!!
        nuc_dip_idx, nuc_strike_idx = self.fault_locations2idxs(
            index, positions_dip=nuc_dip,
            positions_strike=nuc_strike, backend='numpy')

        return self.get_subfault_starttimes(
            index, velocities, nuc_dip_idx, nuc_strike_idx)

    def get_subfault_starttimes(
            self, index, rupture_velocities, nuc_dip_idx, nuc_strike_idx):
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
        slownesses = 1. / rupture_velocities.reshape((npw, npl))

        start_times = fast_sweep.get_rupture_times_numpy(
            slownesses, self.ordering.patch_sizes_dip[index],
            n_patch_strike=npl, n_patch_dip=npw,
            nuc_x=nuc_strike_idx, nuc_y=nuc_dip_idx)
        return start_times

    def get_smoothing_operator(self, correlation_function='nearest_neighbor'):
        """
        Get second order Laplacian smoothing operator.

        This is beeing used to smooth the slip-distribution
        in the optimization.


        Returns
        -------
        :class:`numpy.Ndarray`
            (n_patch_strike + n_patch_dip) x (n_patch_strike + n_patch_dip)
        """

        if correlation_function == 'nearest_neighbor':
            if isinstance(self.config, UniformDiscretizationConfig):
                Ls = []
                for ns in range(self.nsubfaults):
                    self._check_index(ns)
                    npw, npl = self.ordering.get_subfault_discretization(ns)

                    # no smoothing accross sub-faults!
                    Ls.append(get_smoothing_operator_nearest_neighbor(
                        n_patch_strike=npl,
                        n_patch_dip=npw,
                        patch_size_strike=self.ordering.patch_sizes_strike[ns],
                        patch_size_dip=self.ordering.patch_sizes_dip[ns]))
                    return block_diag(Ls)
            else:
                raise InvalidDiscretizationError(
                    'Nearest neighbor correlation Laplacian is only '
                    'available for "uniform" discretization! Please change'
                    ' either correlation_function or the discretization.')
        else:
            datatype = self._assign_datatype()
            subfault_idxs = list(range(self.nsubfaults))
            centers = self.get_subfault_patch_attributes(
                subfault_idxs, datatype, attributes=['center'])[:, :-1]

            return get_smoothing_operator_correlated(
                centers, correlation_function)

    def get_subfault_patch_attributes(
            self, index, datatype='geodetic', component=None, attributes=['']):
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
            self, index, positions_dip, positions_strike, backend='numpy'):
        """
        Return patch indexes for given location on the fault.

        Parameters
        ----------
        positions_dip : :class:`numpy.NdArray` float
            of positions in dip direction of the fault [km]
        positions_strike : :class:`numpy.NdArray` float
            of positions in strike direction of the fault [km]
        backend : str
            which implementation backend to use [numpy/theano]
        """
        # TODO needs subfault index
        backend = get_backend(backend)
        dipidx = positions2idxs(
            positions=positions_dip,
            cell_size=self.ordering.patch_sizes_dip[index],
            backend=backend)
        strikeidx = positions2idxs(
            positions=positions_strike,
            cell_size=self.ordering.patch_sizes_strike[index],
            backend=backend)
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
                key = self.get_subfault_key(
                    index, datatype=None, component=None)
                patches = self._discretized_patches[key]
                npatches.append(len(patches))

            return npatches
        else:
            return [0. for _ in range(self.nsubfaults)]

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
            indexes = num.arange(npatches, dtype='int16').reshape(shp)
            self.vmap.append(PatchMap(count, slc, shp, npatches, indexes))
            self.smap.append(shared(
                indexes,
                name='patchidx_array_%i' % count,
                borrow=True).astype('int16'))
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
        config, sources=None, extension_widths=[0.1], extension_lengths=[0.1],
        patch_widths=[5.], patch_lengths=[5.], datatypes=['geodetic'],
        varnames=['']):
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
        if  na != nsources:
            raise ValueError(
                '"%s" have to be specified for each subfault! Only %i set,'
                ' but %i subfaults are configured!' % (parameter, na, nsources))

    for i, (pl, pw) in enumerate(zip(patch_lengths, patch_widths)):
        if pl != pw:
            raise ValueError(
                'Finite fault optimization does only support'
                ' square patches (yet)! Please adjust the discretization for'
                ' subfault %i: patch-length: %f != patch-width %f!' %
                (i, pl, pw))

    nsources = len(sources)
    if 'seismic' in datatypes and nsources > 1:
        logger.warning(
            'Seismic kinematic finite fault optimization does'
            ' not support rupture propagation across sub-faults yet!')

    check_subfault_consistency(patch_lengths, nsources, 'patch_lengths')
    check_subfault_consistency(patch_widths, nsources, 'patch_widths')
    check_subfault_consistency(extension_lengths, nsources, 'extension_lengths')
    check_subfault_consistency(extension_widths, nsources, 'extension_widths')

    npls = []
    npws = []
    for i, source, in enumerate(sources):
        s = copy.deepcopy(source)
        patch_length_m = patch_lengths[i] * km
        patch_width_m = patch_widths[i] * km
        ext_source = s.extent_source(
            extension_widths[i], extension_lengths[i],
            patch_width_m, patch_length_m)

        npls.append(
            ext_source.get_n_patches(patch_length_m, 'length'))
        npws.append(
            ext_source.get_n_patches(patch_width_m, 'width'))

    ordering = FaultOrdering(
        npls, npws,
        patch_sizes_strike=patch_lengths, patch_sizes_dip=patch_widths)

    fault = FaultGeometry(datatypes, varnames, ordering, config=config)

    for datatype in datatypes:
        logger.info('Discretizing %s source(s)' % datatype)

        for var in varnames:
            logger.info('%s slip component' % var)
            param_mod = copy.deepcopy(slip_directions[var])

            ext_sources = []
            for i, source in enumerate(sources):
                s = copy.deepcopy(source)
                param_mod['rake'] += s.rake
                s.update(**param_mod)

                patch_length_m = patch_lengths[i] * km
                patch_width_m = patch_widths[i] * km
                ext_source = s.extent_source(
                    extension_widths[i], extension_lengths[i],
                    patch_width_m, patch_length_m)

                ext_sources.append(ext_source)
                logger.info('Extended fault(s): \n %s' % ext_source.__str__())

            fault.setup_subfaults(datatype, var, ext_sources)

    return fault


class InvalidDiscretizationError(Exception):

    context = 'Resolution based discretizeation' + \
              ' is available for geodetic data only! \n'

    def __init__(self, errmess=''):
        self.errmess = errmess

    def __str__(self):
        return '\n%s\n%s' % (self.errmess, self.context)


def discretize_sources(
        config, sources=None, datatypes=['geodetic'], varnames=['']):
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
        varnames=varnames)

    if fault.needs_optimization:
        if 'seismic' in datatypes:
            raise InvalidDiscretizationError('Seismic dataset!')

        logger.info(
            'Fault discretization selected to be resolution based.')
    else:
        logger.info('Discretization of Fault uniformly (initial)')
        # uniform discretization
        for component in varnames:
            for datatype in datatypes:
                for index, sf in enumerate(
                        fault.iter_subfaults(datatype, component)):

                    npw, npl = fault.ordering.get_subfault_discretization(index)
                    patches = sf.patches(
                        nl=npl, nw=npw, datatype=datatype, type='pyrocko')
                    fault.set_subfault_patches(
                        index, patches, datatype, component)

    return fault


def optimize_discretization(
        config, fault, datasets, varnames, crust_ind,
        engine, targets, event, force, nworkers):
    """
    Resolution based discretization of the fault surfaces based on:
    Atzori & Antonioli 2011:
        Optimal fault resolution in geodetic inversion of coseismic data
    :return:
    """
    from beat.plotting import source_geometry

    def sv_vec2matrix(sv_vec, ndata):
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
        return num.vstack(
            [num.diag(sv_vec),
             num.zeros((ndata - sv_vec.size, sv_vec.size))])

    logger.info('Optimizing fault discretization based on resolution: ... \n')

    datatype = 'geodetic'

    east_shifts = []
    north_shifts = []
    for dataset in datasets:
        ns, es = dataset.update_local_coords(event)
        north_shifts.append(ns / km)
        east_shifts.append(es / km)

    data_coords = num.vstack(
        [num.hstack(east_shifts), num.hstack(north_shifts)]).T

    patch_widths, patch_lengths = config.get_patch_dimensions()
    for component in varnames:
        for index, sf in enumerate(
                fault.iter_subfaults(datatype, component)):
            npw = sf.get_n_patches(patch_widths[index] * km, 'width')
            npl = sf.get_n_patches(patch_lengths[index] * km, 'length')
            patches = sf.patches(
                nl=npl, nw=npw, datatype=datatype, type='beat')
            fault.set_subfault_patches(index, patches, datatype, component)

    logger.info('Initial number of patches: %i' % fault.npatches)
    tobedivided = fault.npatches

    sf_div_idxs = []
    for i, sf in enumerate(fault.iter_subfaults()):
        if sf.width /km <= config.patch_widths_min[i] or \
                sf.length / km <= config.patch_lengths_min[i]:
            sf_div_idxs = []
        else:
            sf_div_idxs.append(range(fault.subfault_npatches[i] - 1, -1, -1))

    while tobedivided:
        for component in varnames:
            logger.info('Component %s' % component)
            gfs_array = []
            # iterate over subfaults and divide patches
            for sf_idx, div_idxs in zip(range(fault.nsubfaults), sf_div_idxs):
                logger.info(
                    'Subfault %i division indexes %s' % (
                        sf_idx, list2string(div_idxs)))
                patches = fault.get_subfault_patches(sf_idx, datatype, component)
                for idx in div_idxs:

                    # pull out patch to be divided
                    patch = patches.pop(idx)
                    if patch.length >= patch.width:
                        div_patches = patch.patches(
                            nl=2, nw=1, datatype=datatype, type='beat')
                    else:
                        div_patches = patch.patches(
                            nl=1, nw=2, datatype=datatype, type='beat')

                    # insert back divided patches
                    for i, dpatch in enumerate(div_patches):
                        patches.insert(idx + i, dpatch)

                # register newly diveded patches with fault
                fault.set_subfault_patches(
                    sf_idx, patches, datatype, component, replace=True)

            #source_geometry(fault, list(fault.iter_subfaults()))

            logger.info("Calculating Green's Functions for %i "
                        "patches." % fault.npatches)

            # calculate GFs for fault is [npatches, nobservations]
            gfs = geo_construct_gf_linear(
                engine=engine, outdirectory='', crust_ind=crust_ind,
                datasets=datasets, targets=targets, fault=fault,
                varnames=[component], force=force, event=event, nworkers=nworkers)
            gfs_array.append(gfs.T)

        # U data-space, L singular values, V model space
        U, l, V = num.linalg.svd(num.vstack(gfs_array), full_matrices=True)

        # apply singular value damping
        ldamped_inv = 1. / (l + config.epsilon ** 2)
        Linv = sv_vec2matrix(ldamped_inv, ndata=U.shape[0])
        L = sv_vec2matrix(l, ndata=U.shape[0])

        # calculate resolution matrix and take trace
        R = num.diag(num.dot(
            V.dot(Linv.T).dot(U.T),
            U.dot(L.dot(V.T))))

        R_idxs = num.argwhere(R > config.resolution_thresh).ravel().tolist()

        # analysis for further patch division
        sf_div_idxs = []
        width_idxs_max = []
        width_idxs_min = []
        length_idxs_max = []
        length_idxs_min = []
        for i, sf in enumerate(fault.iter_subfaults()):
            widths, lengths = fault.get_subfault_patch_attributes(
                i, datatype, attributes=['width', 'length'])

            width_idxs_max += (num.argwhere(
                widths > config.patch_widths_max[i]).ravel()
                              + fault.cum_subfault_npatches[i]).tolist()
            length_idxs_max += (num.argwhere(
                lengths > config.patch_lengths_max[i]).ravel()
                               + fault.cum_subfault_npatches[i]).tolist()
            width_idxs_min += (num.argwhere(
                widths < config.patch_widths_min[i]).ravel()
                              + fault.cum_subfault_npatches[i]).tolist()
            length_idxs_min += (num.argwhere(
                lengths < config.patch_lengths_min[i]).ravel()
                                + fault.cum_subfault_npatches[i]).tolist()

        # patches that fulfill both size thresholds
        patch_size_ids = set(width_idxs_min).intersection(set(length_idxs_min))

        # patches above R but below size thresholds
        unique_ids = set(
            R_idxs + width_idxs_max + length_idxs_max).difference(
            patch_size_ids)

        ncandidates = len(unique_ids)

        logger.info(
            'Found %i candidate(s) for division for '
            ' %i subfault(s)' % (ncandidates, fault.nsubfaults))
        if ncandidates:
            subfault_idxs = list(range(fault.nsubfaults))
            # calculate division penalties
            uids = num.array(list(unique_ids))
            widths, lengths = fault.get_subfault_patch_attributes(
                subfault_idxs, datatype, attributes=['width', 'length'])
            area_pen = widths[uids] * lengths[uids]

            c1 = []
            for i, sf in enumerate(fault.iter_subfaults()):
                bdepths = fault.get_subfault_patch_attributes(
                    i, datatype, attributes=['bottom_depth'])
                c1.extend(num.exp(
                    -config.depth_penalty * bdepths /
                    sf.bottom_depth * km).tolist())

            c_one_pen = num.array(c1)[uids]

            centers = fault.get_subfault_patch_attributes(
                subfault_idxs, datatype, attributes=['center'])[:, :-1]
            cand_centers = centers[uids, :]

            patch_data_distance_mins = distances(
                points=data_coords, ref_points=cand_centers).min(axis=0)

            c_two_pen = patch_data_distance_mins.min() / \
                        patch_data_distance_mins

            inter_patch_distances = distances(
                points=centers, ref_points=cand_centers)

            c_three_pen = (R * inter_patch_distances.T).sum(axis=1) / \
                          inter_patch_distances.sum(axis=0)

            rating = area_pen * c_one_pen * c_two_pen * c_three_pen
            rating_idxs = rating.argsort()[::-1]

            idxs = uids[rating_idxs[range(
                int(num.ceil(config.alpha * ncandidates)))]]
            logger.info(
                'Patches: %s of %i subfault(s) are further divided.' % (
                    list2string(idxs.tolist()), fault.nsubfaults))
            tobedivided = len(idxs)

            # re-arrange indexes to subfaults
            for i in range(fault.nsubfaults):
                start = fault.cum_subfault_npatches[i]
                end = fault.cum_subfault_npatches[i + 1]
                div_idxs = idxs[(idxs >= start) & (idxs < end)] - start
                # append indexes in descending sorted order
                div_idxs[::-1].sort()
                sf_div_idxs.append(div_idxs.tolist())

        else:
            tobedivided = 0

    logger.info('Finished resolution based fault discretization.')
    logger.info('Quality index for this discretization: %f' % R.mean())
#    source_geometry(fault, list(fault.iter_subfaults()))
    return fault, R
