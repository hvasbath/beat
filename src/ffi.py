from beat import heart, utility
from beat.fast_sweeping import fast_sweep

import copy
import os
import logging
import collections

from pyrocko.trace import snuffle, Trace
from pyrocko import gf

from theano import shared
from theano import config as tconfig
import theano.tensor as tt

import numpy as num

km = 1000.

logger = logging.getLogger('ffi')


PatchMap = collections.namedtuple(
    'PatchMap', 'count, slc, shp, npatches')


gf_entries = ['durations', 'start_times', 'patches', 'targets']


slip_directions = {
    'uparr': {'slip': 1., 'rake': 0.},
    'uperp': {'slip': 1., 'rake': -90.},
    'utensile': {'slip': 0., 'rake': 0., 'opening': 1.}}


class GFLibraryError(Exception):
    pass


class GFLibrary(object):
    """
    Baseclass for linear Greens Function Libraries.
    """
    def __init__(self, component=None, event=None):

        if component is None:
            raise TypeError('Slip component is not defined')

        self.component = component
        self.event = event
        self._patchidxs = None
        self._gfmatrix = None
        self._sgfmatrix = None
        self._mode = 'numpy'
        self.datatype = ''

    def _check_setup(self):
        if self._gfmatrix is None:
            raise GFLibraryError(
                '%s Greens Function Library is not set up!' % self.datatype)

    @property
    def size(self):
        return self._gfmatrix.size

    @property
    def filesize(self):
        """
        Size of the library in MByte.
        """
        return self.size * 8 / (1024. ** 2)


class SeismicGFLibrary(GFLibrary):
    """
    Seismic Greens Funcion Library for the finite fault optimization.

    Eases inspection of Greens Functions through interface to the snuffler.

    Parameters
    ----------
    component : str
        component of slip for which the library is valid
    event : :class:`pyrocko.model.Event`
        Event information for which the library is built
    stations : list
        of station : :class:`pyrocko.model.Station`
    targets : list
        containing :class:`pyrocko.gf.seismosizer.Target` Objects
    """
    def __init__(
            self, component=None, event=None, wavemap=None,
            starttime_sampling=1., duration_sampling=1.):

        super(SeismicGFLibrary, self).__init__(
            component=component, event=event)

        self.datatype = 'seismic'
        self.wavemap = wavemap
        self.starttime_sampling = starttime_sampling
        self.duration_sampling = duration_sampling
        self._tmins = None

    def __str__(self):
        s = '''
            %s GF Library
            ------------------
            slip component: %s
            starttime_sampling[s]: %f
            duration_sampling[s]: %f
            ntargets: %i
            npatches: %i
            ndurations: %i
            nstarttimes: %i
            nsamples: %i
            size: %i
            filesize [MB]: %f
            filename: %s''' % (
            self.component, self.starttime_sampling, self.duration_sampling,
            self.ntargets, self.npatches, self.ndurations,
            self.nstarttimes, self.nsamples, self.size, self.filesize,
            self.filename)
        return s

    def setup(self, ntargets, npatches, ndurations, nstarttimes, nsamples):
        if ntargets != self.nstations:
            raise GFLibraryError(
                'Number of stations and targets is inconsistent!'
                'ntargets %i, nstations %i' % (ntargets, self.nstations))

        self._gfmatrix = num.zeros(
            [ntargets, npatches, ndurations, nstarttimes, nsamples])
        self._tmins = num.zeros([ntargets, npatches])
        self._patchidxs = num.arange(npatches, dtype='int16')

    def init_optimization(self):
        logger.info('Setting linear seismic GF Library to optimization mode.')
        self._sgfmatrix = shared(
            self._gfmatrix.astype(tconfig.floatX), borrow=True)

        self._stack_switch = {
            'numpy': self._gfmatrix,
            'theano': self._sgfmatrix}

        self.set_stack_mode(mode='theano')

    def put(
            self, entries, tmin, target, patchidx,
            durationidxs, starttimeidxs):
        """
        Fill the GF Library with synthetic traces for one target and one patch.

        Parameters
        ----------
        entries : 2d :class:`numpy.ndarray`
        """
        if len(entries.shape) < 2:
            raise ValueError('Entries have to be 2d arrays!')

        if entries.shape[1] != self.nsamples:
            raise GFLibraryError(
                'Trace length of entries is not consistent with the library'
                ' to be filled! Entries length: %i Library: %i.' % (
                    entries.shape[0], self.nsamples))

        self._check_setup()
        tidx = self.wavemap.target_index_mapping()[target]

        self._gfmatrix[
            tidx, patchidx, durationidxs, starttimeidxs, :] = entries
        self._tmins[tidx, patchidx] = tmin

    def trace_tmin(self, target, patchidx, starttimeidx):
        t2i = self.wavemap.target_index_mapping()
        return self._tmins[t2i[target], patchidx] + \
            (starttimeidx * self.starttime_sampling)

    def set_stack_mode(self, mode='numpy'):
        """
        Sets mode on witch backend the stacking is working.
        Dependend on that the input to the stack function has to be
        either of :class:`numpy.ndarray` or of :class:`theano.tensor.Tensor`

        Parameters
        ----------
        mode : str
            on witch array to stack
        """
        available_modes = self._stack_switch.keys()
        if mode not in available_modes:
            raise GFLibraryError(
                'Stacking mode %s not available! '
                'Available modes: %s' % utility.list2string(available_modes))

        self._mode = mode

    def stack(self, target, patchidxs, durationidxs, starttimeidxs, slips):
        """
        Stack selected traces from the GF Library of specified
        target, patch, durations and starttimes. Numpy or theano dependend
        on the stack_mode

        Parameters
        ----------

        Returns
        -------
        :class:`numpy.ndarray` or of :class:`theano.tensor.Tensor` dependend
        on stack mode
        """

        tidx = self.wavemap.target_index_mapping()[target]
        return self._stack_switch[self._mode][
            tidx, patchidxs, durationidxs, starttimeidxs, :].reshape(
                (slips.shape[0], self.nsamples)).T.dot(slips)

    def stack_all(self, durations, starttimes, slips):
        """
        Stack all patches for all targets at once.
        In theano for efficient optimization.

        Parameters
        ----------

        Returns
        -------
        matrix : size (ntargets, nsamples)
        option : tensor.batched_dot(sd.dimshuffle((1,0,2)), u).sum(axis=0)
        """
        if self._sgfmatrix is None:
            raise GFLibraryError(
                'To use theano stacking optimization mode'
                ' has to be initialised!')

        starttimeidxs = tt.round(
            starttimes / self.starttime_sampling).astype('int16')
        durationidxs = tt.round(
            durations / self.duration_sampling).astype('int16')

        d = self._sgfmatrix[
            :, self._patchidxs, durationidxs, starttimeidxs, :].reshape(
            (self.ntargets, self.npatches, self.nsamples))
        return tt.batched_dot(
            d.dimshuffle((1, 0, 2)), slips).sum(axis=0)

    def snuffle(
            self, targets=[], patchidxs=[0], durationidxs=[0],
            starttimeidxs=[0]):
        """
        Opens pyrocko's snuffler with the specified traces.

        Parameters
        ----------
        """
        traces = []
        display_stations = []
        t2i = self.wavemap.target_index_mapping()
        for target in targets:
            tidx = t2i[target]
            display_stations.append(self.wavemap.stations[tidx])
            for patchidx in patchidxs:
                for rtidx in durationidxs:
                    for startidx in starttimeidxs:
                        tr = Trace(
                            ydata=self._gfmatrix[
                                tidx, patchidx, rtidx, startidx, :],
                            deltat=float(target.store_id.split('_')[3][0:5]),
                            tmin=self.trace_tmin(
                                target, patchidx, starttimeidxs))
                        traces.append(tr)

        snuffle(traces, events=[self.event], stations=display_stations)

    @property
    def nstations(self):
        return len(self.stations)

    @property
    def ntargets(self):
        self._check_setup()
        return self._gfmatrix.shape[0]

    @property
    def npatches(self):
        self._check_setup()
        return self._gfmatrix.shape[1]

    @property
    def ndurations(self):
        self._check_setup()
        return self._gfmatrix.shape[2]

    @property
    def nstarttimes(self):
        self._check_setup()
        return self._gfmatrix.shape[3]

    @property
    def nsamples(self):
        self._check_setup()
        return self._gfmatrix.shape[4]

    @property
    def filename(self):
        return '%s_%s_%s.pkl' % (
            self.datatype, self.component, self.wavemap.wavename)


class FaultOrdering(object):
    """
    A mapping of source patches to the arrays of optimization results.

    Parameters
    ----------
    npls : list
        of number of patches in strike-direction
    npws : list
        of number of patches in dip-direction
    """

    def __init__(self, npls, npws):

        self.vmap = []
        dim = 0
        count = 0

        for npl, npw in zip(npls, npws):
            npatches = npl * npw
            slc = slice(dim, dim + npatches)
            shp = (npw, npl)
            self.vmap.append(PatchMap(count, slc, shp, npatches))
            dim += npatches
            count += 1

        self.npatches = dim


class FaultGeometryError(Exception):
    pass


class FaultGeometry(gf.seismosizer.Cloneable):
    """
    Object to construct complex fault-geometries with several subfaults.
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
    """

    def __init__(self, datatypes, components, ordering):
        self.datatypes = datatypes
        self.components = components
        self._ext_sources = {}
        self.ordering = ordering

    def _check_datatype(self, datatype):
        if datatype not in self.datatypes:
            raise TypeError('Datatype not included in FaultGeometry')

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

        self._check_datatype(datatype)
        self._check_component(component)

        if len(ext_sources) != self.nsubfaults:
            raise FaultGeometryError('Setup does not match fault ordering!')

        for i, source in enumerate(ext_sources):
            source_key = self.get_subfault_key(i, datatype, component)

            if source_key not in self._ext_sources.keys() or replace:
                self._ext_sources[source_key] = copy.deepcopy(source)
            else:
                raise FaultGeometryError(
                    'Subfault already specified in geometry!')

    def iter_subfaults(self, datatype=None, component=None):
        """
        Iterator over subfaults.
        """
        for i in range(self.nsubfaults):
            yield self.get_subfault(
                index=i, datatype=datatype, component=component)

    def get_subfault(self, index, datatype=None, component=None):

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key in self._ext_sources.keys():
            return self._ext_sources[source_key]
        else:
            raise FaultGeometryError('Requested subfault not defined!')

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

        subfault = self.get_subfault(
            index, datatype=datatype, component=component)
        npw, npl = self.ordering.vmap[index].shp

        return subfault.patches(nl=npl, nw=npw, datatype=datatype)

    def get_all_patches(self, datatype=None, component=None):
        """
        Get all RectangularSource patches for the full complex fault.

        Parameters
        ----------
        datatype : str
            'geodetic' or 'seismic'
        component : str
            slip component to return may be %s
        """ % utility.list2string(slip_directions.keys())

        patches = []
        for i in range(self.nsubfaults):
            patches += self.get_subfault_patches(
                i, datatype=datatype, component=component)

        return patches

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
        return self.ordering.vmap[index].slc

    def get_subfault_starttime_bound(self, index, rupture_velocities):
        """
        Get maximum bound of start times of extending rupture along
        the sub-fault.
        """

        npw, npl = self.ordering.vmap[index].shp
        slownesses = 1. / rupture_velocities.reshape((npw, npl))

        subfault = self.get_subfault(
            index=index, datatype='seismic', component=self.components[0])

        patch_size = subfault.width / npw

        start_times = fast_sweep.get_rupture_times_numpy(
            slownesses, patch_size / km,
            n_patch_strike=npl, n_patch_dip=npw,
            nuc_x=0, nuc_y=0)
        return start_times.max()

    @property
    def nsubfaults(self):
        return len(self.ordering.vmap)

    @property
    def nsubpatches(self):
        return self.ordering.npatches


def discretize_sources(
        sources=None, extension_width=0.1, extension_length=0.1,
        patch_width=5000., patch_length=5000., datatypes=['geodetic'],
        varnames=['']):
    """
    Build complex discretized fault.

    Extend sources into all directions and discretize sources into patches.
    Rounds dimensions to have no half-patches.

    Parameters
    ----------
    sources : :class:`sources.RectangularSource`
        Reference plane, which is being extended and
    extension_width : float
        factor to extend source in width (dip-direction)
    extension_length : float
        factor extend source in length (strike-direction)
    patch_width : float
        Width [m] of subpatch in dip-direction
    patch_length : float
        Length [m] of subpatch in strike-direction
    varnames : list
        of str with variable names that are being optimized for

    Returns
    -------
    :class:'FaultGeometry'
    """
    if 'seismic' in datatypes and patch_length != patch_width:
        raise ValueError(
            'Seismic kinematic fault optimization does only support'
            ' square patches (yet)! Please adjust the discretization!')

    nsources = len(sources)
    if 'seismic' in datatypes and nsources > 1:
        raise ValueError(
            'Seismic kinematic fault optimization does'
            ' only support one main fault (TODO fast'
            ' sweeping across sub-faults)!'
            ' nsources defined: %i' % nsources)

    npls = []
    npws = []
    for source in sources:
        s = copy.deepcopy(source)
        ext_source = s.extent_source(
            extension_width, extension_length,
            patch_width, patch_length)

        npls.append(int(num.ceil(ext_source.length / patch_length)))
        npws.append(int(num.ceil(ext_source.width / patch_width)))

    ordering = utility.FaultOrdering(npls, npws)

    fault = FaultGeometry(datatypes, varnames, ordering)

    for datatype in datatypes:
        logger.info('Discretizing %s source(s)' % datatype)

        for var in varnames:
            logger.info('%s slip component' % var)
            param_mod = copy.deepcopy(slip_directions[var])

            ext_sources = []
            for source in sources:
                s = copy.deepcopy(source)
                param_mod['rake'] += s.rake
                s.update(**param_mod)

                ext_source = s.extent_source(
                    extension_width, extension_length,
                    patch_width, patch_length)

                npls.append(
                    ext_source.get_n_patches(patch_length, 'length'))
                npws.append(
                    ext_source.get_n_patches(patch_width, 'width'))
                ext_sources.append(ext_source)
                logger.info('Extended fault(s): \n %s' % ext_source.__str__())

            fault.setup_subfaults(datatype, var, ext_sources)

    return fault


def geo_construct_gf_linear(
        engine, outpath, crust_ind=0, datasets=None,
        targets=None, fault=None, varnames=[''], force=False):
    """
    Create geodetic Greens Function matrix for defined source geometry.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        main path to directory containing the different Greensfunction stores
    outpath : str
        absolute path to the directory and filename where to store the
        Green's Functions
    crust_ind : int
        of index of Greens Function store to use
    datasets : list
        of :class:`heart.GeodeticDataset` for which the GFs are calculated
    targets : list
        of :class:`heart.GeodeticDataset`
    fault : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry
    varnames : list
        of str with variable names that are being optimized for
    force : bool
        Force to overwrite existing files.
    """

    if os.path.exists(outpath) and not force:
        logger.info(
            "Green's Functions exist! Use --force to"
            " overwrite!")
    else:
        out_gfs = {}
        for var in varnames:
            logger.info('For slip component: %s' % var)

            gfs = []
            for source in fault.get_all_patches('geodetic', component=var):
                disp = heart.geo_synthetics(
                    engine=engine,
                    targets=targets,
                    sources=[source],
                    outmode='stacked_arrays')

                gfs_data = []
                for d, data in zip(disp, datasets):
                    logger.debug('Target %s' % data.__str__())
                    gfs_data.append((
                        d[:, 0] * data.los_vector[:, 0] +
                        d[:, 1] * data.los_vector[:, 1] +
                        d[:, 2] * data.los_vector[:, 2]) *
                        data.odw)

                gfs.append(num.vstack(gfs_data).T)

        out_gfs[var] = gfs
        logger.info("Dumping Green's Functions to %s" % outpath)
        utility.dump_objects(outpath, [out_gfs])


def seis_construct_gf_linear(
        engine, fault, duration, varnames, wavemap, event,
        velocity, starttime_sampling, duration_sampling,
        sample_rate, outdirectory, force):
    """
    Create seismic Greens Function matrix for defined source geometry
    by convolution of the GFs with the source time function (STF).

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        main path to directory containing the different Greensfunction stores
    targets : list
        of pyrocko target objects for respective phase to compute
    wavemap : :class:`heart.WaveformMapping`
        configuration parameters for handeling seismic data around Phase
    fault : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry
    durations : :class:`heart.Parameter`
        prior of durations of the STF for each patch to convolve
    duration_sampling : float
        incremental step size for precalculation of duration GFs
    velocity : :class:`heart.Parameter`
        rupture velocity of earthquake prior
    starttime_sampling : float
        incremental step size for precalculation of startime GFs
    sample_rate : float
        sample rate of synthetic traces to produce,
        related to non-linear GF store
    outpath : str
        directory for storage
    force : boolean
        flag to overwrite existing linear GF Library
    """

    start_times = fault.get_subfault_starttime_bound(
        index=0, rupture_velocities=velocity)
    starttimeidxs = num.arange(
        int(num.ceil(start_times.max() / starttime_sampling)))
    starttimes = (starttimeidxs * starttime_sampling).tolist()

    durations = num.arange(
        duration.lower.min(), duration.upper.max(), duration_sampling)
    durationidxs = num.ceil(
        (durations - duration.lower) / duration_sampling)

    logger.info(
        'Calculating GFs for starttimes: %s \n durations: %s' %
        (utility.list2string(starttimes), utility.list2string(durations)))

    nstarttimes = len(starttimes)
    npatches = len(fault.nsubfaults)
    ntargets = len(wavemap.targets)
    ndurations = durations.size
    nsamples = int(num.ceil(wavemap.arrival_taper.duration * sample_rate))

    for var in varnames:
        logger.info('For slip component: %s' % var)
        gfs = SeismicGFLibrary(
            component=var, event=event, wavemap=wavemap,
            duration_sampling=duration_sampling,
            starttime_sampling=starttime_sampling)

        outpath = os.path.join(outdirectory, gfs.filename)
        if not os.path.exists(outpath) or force:
            gfs.setup(ntargets, npatches, ndurations, nstarttimes, nsamples)

            for patchidx, patch in enumerate(
                    fault.get_all_patches('seismic', component=var)):

                source_patches_durations = []
                logger.info('Patch Number %i', patchidx)

                for duration in durations:
                    pcopy = patch.clone()
                    pcopy.update(duration=duration)
                    source_patches_durations.append(copy)

                for j, target in enumerate(wavemap.targets):

                    traces, tmins = heart.seis_synthetics(
                        engine=engine,
                        sources=source_patches_durations,
                        targets=[target],
                        arrival_taper=None,
                        wavename=wavemap.wavename,
                        filterer=None,
                        reference_taperer=None,
                        outmode='traces')

                    arrival_time = heart.get_phase_arrival_time(
                        engine=engine,
                        source=patch,
                        target=target,
                        wavename=wavemap.wavename)

                    for starttime in starttimeidxs:

                        tmin = wavemap.arrival_taper.a + \
                            arrival_time + starttime

                        synthetics_array = heart.taper_filter_traces(
                            traces=traces,
                            arrival_taper=wavemap.arrival_taper,
                            filterer=wavemap.filterer,
                            tmins=num.ones(ndurations) * tmin,
                            outmode='array')

                        gfs.put(
                            entries=synthetics_array,
                            tmin=tmin,
                            target=target,
                            patchidx=patchidx,
                            durationidxs=durationidxs,
                            starttimeidxs=starttime)

            logger.info('Storing seismic linear GF Library under ', outpath)

            utility.dump_objects(outpath, [gfs])

        else:
            logger.info(
                'GF Library exists at path: %s. '
                'Use --force to overwrite!' % outpath)
