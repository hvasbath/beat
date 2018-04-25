from beat import heart
from beat import utility as ut
from beat.fast_sweeping import fast_sweep
from beat import parallel
from beat.config import SeismicGFLibraryConfig, GeodeticGFLibraryConfig

import copy
import os
import logging
import collections

from multiprocessing import RawArray

from pyrocko.trace import snuffle, Trace
from pyrocko import gf
from pyrocko.guts import load

from theano import shared
from theano import config as tconfig
import theano.tensor as tt

import numpy as num


gf_dtype = 'float64'

backends = {'numpy': num, 'theano': tt}


km = 1000.

logger = logging.getLogger('ffi')


PatchMap = collections.namedtuple(
    'PatchMap', 'count, slc, shp, npatches, indexmap')


gf_entries = ['durations', 'start_times', 'patches', 'targets']


slip_directions = {
    'uparr': {'slip': 1., 'rake': 0.},
    'uperp': {'slip': 1., 'rake': -90.},
    'utensile': {'slip': 0., 'rake': 0., 'opening': 1.}}


def _init_shared(gfstofill, tminstofill):
    logger.debug('Accessing shared arrays!')
    parallel.gfmatrix = gfstofill
    parallel.tmins = tminstofill


class GFLibraryError(Exception):
    pass


class GFLibrary(object):
    """
    Baseclass for linear Greens Function Libraries.
    """
    def __init__(self, config):

        self.config = config
        self._gfmatrix = None
        self._sgfmatrix = None
        self._patchidxs = None
        self._mode = 'numpy'
        self._stack_switch = {}

    def _check_mode_init(self, mode):
        if mode == 'theano':
            if self._sgfmatrix is None:
                raise GFLibraryError(
                    'To use "stack_all" theano stacking optimization mode'
                    ' has to be initialised!')

    def _check_setup(self):
        if sum(self.config.dimensions) == 0:
            raise GFLibraryError(
                '%s Greens Function Library is not set up!' % self.datatype)

    @property
    def size(self):
        return num.array(self.config.dimensions).prod()

    @property
    def filesize(self):
        """
        Size of the library in MByte.
        """
        return self.size * 8. / (1024. ** 2)

    @property
    def patchidxs(self):
        if self._patchidxs is None:
            self._patchidxs = num.arange(self.npatches, dtype='int16')
        return self._patchidxs

    @property
    def sw_patchidxs(self):
        if self._mode == 'numpy':
            return self.patchidxs
        elif self._mode == 'theano':
            return self.spatchidxs

    def save_config(self, outdir='', filename=None):

        filename = '%s.yaml' % self.filename
        outpath = os.path.join(outdir, filename)
        logger.debug('Dumping GF config to %s' % outpath)
        header = 'beat.ffi.%s YAML Config' % self.__class__.__name__
        self.config.regularize()
        self.config.validate()
        self.config.dump(filename=outpath, header=header)

    def load_config(self, filename):

        try:
            config = load(filename=filename)
        except IOError:
            raise IOError(
                'Cannot load config, file %s does not exist!' % filename)

        self.config = config

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
        available_modes = backends.keys()
        if mode not in available_modes:
            raise GFLibraryError(
                'Stacking mode %s not available! '
                'Available modes: %s' % ut.list2string(available_modes))

        self._mode = mode


def get_gf_prefix(datatype, component, wavename, crust_ind):
    return '%s_%s_%s_%i' % (
        datatype, component, wavename, crust_ind)


def load_gf_library(directory='', filename=None):
    """
    Loading GF Library config and initialise memmaps for Traces and times.
    """
    inpath = os.path.join(directory, filename)
    datatype = filename.split('_')[0]

    if datatype == 'seismic':
        gfs = SeismicGFLibrary()
        gfs.load_config(filename=inpath + '.yaml')
        gfs._gfmatrix = num.load(
            inpath + '.traces.npy',
            mmap_mode=('r'),
            allow_pickle=False)
        gfs._tmins = num.load(
            inpath + '.times.npy',
            mmap_mode=('r'),
            allow_pickle=False)

    elif datatype == 'geodetic':
        gfs = GeodeticGFLibrary()
        gfs.load_config(filename=inpath + '.yaml')
        gfs._gfmatrix = num.load(
            inpath + '.traces.npy',
            mmap_mode=('r'),
            allow_pickle=False)

    else:
        raise ValueError('datatype "%s" not supported!' % datatype)

    gfs._stack_switch['numpy'] = gfs._gfmatrix
    return gfs


class GeodeticGFLibrary(GFLibrary):
    """
    Seismic Greens Funcion Library for the finite fault optimization.

    Parameters
    ----------
    config : :class:`SeismicGFLibraryConfig`
    """
    def __init__(self, config=SeismicGFLibraryConfig()):

        super(GeodeticGFLibrary, self).__init__(config=config)

        self._sgfmatrix = None

    def __str__(self):
        s = '''
Geodetic GF Library
------------------
%s
npatches: %i
nsamples: %i
size: %i
filesize [MB]: %f
filename: %s''' % (
            self.config.dump(),
            self.npatches, self.nsamples,
            self.size, self.filesize,
            self.filename)
        return s

    def save(self, outdir='', filename=None):
        """
        Save GFLibrary data and config file.
        """
        filename = filename or '%s' % self.filename
        outpath = os.path.join(outdir, filename)
        logger.info('Dumping GF Library to %s' % outpath)
        num.save(outpath + '.traces', arr=self._gfmatrix, allow_pickle=False)
        self.save_config(outdir=outdir, filename=filename)

    def setup(
            self, npatches, nsamples, allocate=False):

        self.dimensions = (npatches, nsamples)

        if allocate:
            logger.info('Allocating GF Library')
            self._gfmatrix = num.zeros(self.dimensions)

        self.set_stack_mode(mode='numpy')

    def init_optimization(self):

        logger.info(
            'Setting %s GF Library to optimization mode.' % self.filename)
        self._sgfmatrix = shared(
            self._gfmatrix.astype(tconfig.floatX),
            name=self.filename, borrow=True)
        parallel.memshare([self.filename])

        self.spatchidxs = shared(
            self.patchidxs, name='geo_patchidx_vec', borrow=True)

        self._stack_switch = {
            'numpy': self._gfmatrix,
            'theano': self._sgfmatrix}

        self.set_stack_mode(mode='theano')

    def put(
            self, entries, patchidx):
        """
        Fill the GF Library with synthetic traces for one target and one patch.

        Parameters
        ----------
        entries : 2d :class:`numpy.NdArray`
            of synthetic trace data samples, the waveforms
        patchidx : int
            index to patch (source) that is used to produce the synthetics
        """

        if len(entries.shape) < 1:
            raise ValueError('Entries have to be 1d arrays!')

        if entries.shape[0] != self.nsamples:
            raise GFLibraryError(
                'Trace length of entries is not consistent with the library'
                ' to be filled! Entries length: %i Library: %i.' % (
                    entries.shape[0], self.nsamples))

        self._check_setup()

        if hasattr(parallel, 'gfmatrix'):
            matrix = num.frombuffer(parallel.gfmatrix).reshape(self.dimensions)

        elif self._gfmatrix is None:
            raise GFLibraryError(
                'Neither shared nor standard GFLibrary is setup!')

        else:
            matrix = self._gfmatrix

        matrix[patchidx, :] = entries

    def stack_all(self, slips):
        """
        Stack all patches for all targets at once.
        In theano for efficient optimization.

        Parameters
        ----------

        Returns
        -------
        matrix : size (nsamples)
        """
        self._check_mode_init(self._mode)
        return self.stack_switch[self._mode].T.dot(slips)

    @property
    def nsamples(self):
        return self.config.dimensions[1]

    @property
    def npatches(self):
        return self.config.dimensions[0]

    @property
    def filename(self):
        return get_gf_prefix(
            self.config.datatype, self.config.component,
            'static', self.config.crust_ind)


class SeismicGFLibrary(GFLibrary):
    """
    Seismic Greens Funcion Library for the finite fault optimization.

    Eases inspection of Greens Functions through interface to the snuffler.

    Parameters
    ----------
    config : :class:`SeismicGFLibraryConfig`
    """
    def __init__(self, config=SeismicGFLibraryConfig()):

        super(SeismicGFLibrary, self).__init__(config=config)

        self._sgfmatrix = None
        self._stmins = None

    def __str__(self):
        s = '''
Seismic GF Library
------------------
%s
ntargets: %i
npatches: %i
ndurations: %i
nstarttimes: %i
nsamples: %i
size: %i
filesize [MB]: %f
filename: %s''' % (
            self.config.dump(),
            self.ntargets, self.npatches, self.ndurations,
            self.nstarttimes, self.nsamples, self.size, self.filesize,
            self.filename)
        return s

    def save(self, outdir='', filename=None):
        """
        Save GFLibrary data and config file.
        """
        filename = filename or '%s' % self.filename
        outpath = os.path.join(outdir, filename)
        logger.info('Dumping GF Library to %s' % outpath)
        num.save(outpath + '.traces', arr=self._gfmatrix, allow_pickle=False)
        num.save(outpath + '.times', arr=self._tmins, allow_pickle=False)
        self.save_config(outdir=outdir, filename=filename)

    def setup(
            self, ntargets, npatches, ndurations,
            nstarttimes, nsamples, allocate=False):

        self.dimensions = (
            ntargets, npatches, ndurations, nstarttimes, nsamples)

        if allocate:
            logger.info('Allocating GF Library')
            self._gfmatrix = num.zeros(self.dimensions)
            self._tmins = num.zeros([ntargets, npatches + 1])

        self.set_stack_mode(mode='numpy')

    def init_optimization(self):

        logger.info(
            'Setting %s GF Library to optimization mode.' % self.filename)
        self._sgfmatrix = shared(
            self._gfmatrix.astype(tconfig.floatX),
            name=self.filename, borrow=True)
        parallel.memshare([self.filename])

        self._stmins = shared(
            self._tmins.astype(tconfig.floatX),
            name=self.filename + '_tmins',
            borrow=True)

        self.spatchidxs = shared(
            self.patchidxs, name='seis_patchidx_vec', borrow=True)

        self._stack_switch = {
            'numpy': self._gfmatrix,
            'theano': self._sgfmatrix}

        self.set_stack_mode(mode='theano')

    def set_patch_time(self, targetidx, patchidx, tmin):
        """
        Fill the GF Library with trace times for one target and one patch.

        Parameters
        ----------
        targetidx : int
            index to target
        patchidx : int
            index to patch (source) that is assumed to be hypocenter
        tmin : float
            tmin of the trace(s) if the hypocenter was in the location of this
            patch
        """

        if hasattr(parallel, 'tmins'):
            times = num.frombuffer(parallel.tmins).reshape(
                (self.ntargets, self.npatches + 1))

        elif self._tmins is None:
            raise GFLibraryError(
                'Neither shared nor standard GFLibrary is setup!')

        else:
            times = self._tmins

        times[targetidx, patchidx] = tmin

    def put(
            self, entries, targetidx, patchidx,
            durations, starttimes):
        """
        Fill the GF Library with synthetic traces for one target and one patch.

        Parameters
        ----------
        entries : 2d :class:`numpy.NdArray`
            of synthetic trace data samples, the waveforms
        targetidx : int
            index to target
        patchidx : int
            index to patch (source) that is used to produce the synthetics
        durationidxs : list or :class:`numpy.NdArray`
            of indexes to the respective duration of the STFs that have been
            used to create the synthetics
        starttimeidxs : list or :class:`numpy.NdArray`
            of indexes to the respective duration of the STFs that have been
            used to create the synthetics
        """

        if len(entries.shape) < 2:
            raise ValueError('Entries have to be 2d arrays!')

        if entries.shape[1] != self.nsamples:
            raise GFLibraryError(
                'Trace length of entries is not consistent with the library'
                ' to be filled! Entries length: %i Library: %i.' % (
                    entries.shape[0], self.nsamples))

        self._check_setup()

        durationidxs = self.durations2idxs(durations)
        starttimeidxs = self.starttimes2idxs(starttimes)

        if hasattr(parallel, 'gfmatrix'):
            matrix = num.frombuffer(parallel.gfmatrix).reshape(self.dimensions)

        elif self._gfmatrix is None:
            raise GFLibraryError(
                'Neither shared nor standard GFLibrary is setup!')

        else:
            matrix = self._gfmatrix

        matrix[targetidx, patchidx, durationidxs, starttimeidxs, :] = entries

    def trace_tmin(self, targetidx, patchidx):
        """
        Returns trace time of single target with respect to hypocentral trace.
        """
        return float(self._tmins[targetidx, patchidx])

    def get_all_tmins(self, patchidx):
        """
        Returns tmins for all targets for specified hypocentral patch.
        """
        if self._mode == 'theano':
            if self._stmins is None:
                raise GFLibraryError(
                    'To use "get_all_tmins" theano stacking optimization mode'
                    ' has to be initialised!')
            return self._stmins[:, patchidx]

        elif self._mode == 'numpy':
            return self._tmins[:, patchidx]

    def starttimes2idxs(self, starttimes, interpolation='nearest_neighbor'):
        """
        Transforms starttimes into indexes to the GFLibrary.
        Depending on the stacking mode of the GFLibrary theano or numpy
        is used.

        Parameters
        ----------
        starttimes [s]: :class:`numpy.ndarray` or :class:`theano.tensor.Tensor`
            of the rupturing of the patch, float

        Returns
        -------
        starttimeidxs : starttimes : :class:`numpy.ndarray` or
            :class:`theano.tensor.Tensor`, int16
            (output depends on interpolation scheme,
             if multilinear interpolation factors are returned as well)
        """
        if interpolation == 'nearest_neighbor':
            return backends[self._mode].round(
                (starttimes - self.starttime_min) /
                self.starttime_sampling).astype('int16'), None
        elif interpolation == 'multilinear':
            dstarttimes = (starttimes - self.starttime_min) / \
                self.starttime_sampling
            ceil_starttimes = backends[self._mode].ceil(
                dstarttimes).astype('int16')
            factors = ceil_starttimes - dstarttimes
            return ceil_starttimes, factors
        else:
            raise NotImplementedError(
                'Interpolation scheme %s not implemented!' % interpolation)

    def idxs2durations(self, idxs):
        """
        Map index to durations [s]
        """
        return idxs * self.duration_sampling + self.duration_min

    def idxs2starttimes(self, idxs):
        """
        Map index to durations [s]
        """
        return idxs * self.starttime_sampling + self.starttime_min

    def durations2idxs(self, durations, interpolation='nearest_neighbor'):
        """
        Transforms durations into indexes to the GFLibrary.
        Depending on the stacking mode of the GFLibrary theano or numpy
        is used.

        Parameters
        ----------
        durations [s] : :class:`numpy.ndarray` or :class:`theano.tensor.Tensor`
            of the rupturing of the patch, float

        Returns
        -------
        durationidxs : starttimes : :class:`numpy.ndarray` or
            :class:`theano.tensor.Tensor`, int16
        """
        if interpolation == 'nearest_neighbor':
            return backends[self._mode].round(
                (durations - self.duration_min) /
                self.duration_sampling).astype('int16'), None
        elif interpolation == 'multilinear':
            ddurations = (durations - self.duration_min) / \
                self.duration_sampling
            ceil_durations = backends[self._mode].ceil(
                ddurations).astype('int16')
            factors = ceil_durations - ddurations
            return ceil_durations, factors
        else:
            raise NotImplementedError(
                'Interpolation scheme %s not implemented!' % interpolation)

    def stack(self, targetidx, patchidxs, durationidxs, starttimeidxs, slips):
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
        return self._stack_switch[self._mode][
            targetidx, patchidxs, durationidxs, starttimeidxs, :].reshape(
                (slips.shape[0], self.nsamples)).T.dot(slips)

    def stack_all(
            self, durations, starttimes, slips,
            interpolation='nearest_neighbor'):
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

        self._check_mode_init(self._mode)

        durationidxs, rt_factors = self.durations2idxs(
            durations, interpolation=interpolation)
        starttimeidxs, st_factors = self.starttimes2idxs(
            starttimes, interpolation=interpolation)

        if interpolation == 'nearest_neighbor':

            nslips = 1
            cd = self._stack_switch[self._mode][
                :, self.sw_patchidxs,
                durationidxs, starttimeidxs, :].reshape(
                    (self.ntargets, self.npatches, self.nsamples))
            cslips = slips

        elif interpolation == 'multilinear':

            nslips = 4
            d_st_ceil_rt_ceil = self._stack_switch[self._mode][
                :, self.sw_patchidxs,
                durationidxs, starttimeidxs, :].reshape(
                (self.ntargets, self.npatches, self.nsamples))
            d_st_floor_rt_ceil = self._stack_switch[self._mode][
                :, self.sw_patchidxs,
                durationidxs, starttimeidxs - 1, :].reshape(
                (self.ntargets, self.npatches, self.nsamples))
            d_st_ceil_rt_floor = self._stack_switch[self._mode][
                :, self.sw_patchidxs,
                durationidxs - 1, starttimeidxs, :].reshape(
                (self.ntargets, self.npatches, self.nsamples))
            d_st_floor_rt_floor = self._stack_switch[self._mode][
                :, self.sw_patchidxs,
                durationidxs - 1, starttimeidxs - 1, :].reshape(
                (self.ntargets, self.npatches, self.nsamples))

            s_st_ceil_rt_ceil = (1 - st_factors) * (1 - rt_factors) * slips
            s_st_floor_rt_ceil = st_factors * (1. - rt_factors) * slips
            s_st_ceil_rt_floor = (1 - st_factors) * rt_factors * slips
            s_st_floor_rt_floor = st_factors * rt_factors * slips

            cd = backends[self._mode].concatenate(
                [d_st_ceil_rt_ceil, d_st_floor_rt_ceil,
                 d_st_ceil_rt_floor, d_st_floor_rt_floor], axis=1)
            cslips = backends[self._mode].concatenate(
                [s_st_ceil_rt_ceil, s_st_floor_rt_ceil,
                 s_st_ceil_rt_floor, s_st_floor_rt_floor])

        else:
            raise NotImplementedError(
                'Interpolation scheme %s not implemented!' % interpolation)

        if self._mode == 'theano':
            return tt.batched_dot(
                cd.dimshuffle((1, 0, 2)), cslips).sum(axis=0)

        elif self._mode == 'numpy':
            u2d = num.tile(
                cslips, self.nsamples).reshape(
                    (self.nsamples, self.npatches * nslips))
            return num.einsum('ijk->ik', cd * u2d.T)

    def get_traces(
            self, targetidxs=[0], patchidxs=[0], durationidxs=[0],
            starttimeidxs=[0]):
        """
        Return traces for specified indexes.

        Parameters
        ----------
        """
        traces = []
        for targetidx in targetidxs:
            for patchidx in patchidxs:
                for durationidx in durationidxs:
                    for starttimeidx in starttimeidxs:
                        ydata = self._gfmatrix[
                            targetidx, patchidx, durationidx, starttimeidx, :]
                        tr = Trace(
                            ydata=ydata,
                            deltat=self.deltat,
                            network='target_%i' % targetidx,
                            station='patch_%i' % patchidx,
                            channel='tau_%.2f' % self.idxs2durations(
                                durationidx),
                            location='t0_%.2f' % self.idxs2starttimes(
                                starttimeidx),
                            tmin=self.trace_tmin(targetidx, -1))
                        traces.append(tr)

        return traces

    @property
    def reference_times(self):
        return self._tmins[:, -1]

    @property
    def deltat(self):
        return self.config.wave_config.arrival_taper.duration / \
            float(self.nsamples)

    @property
    def nstations(self):
        return len(self.stations)

    @property
    def ntargets(self):
        return self.config.dimensions[0]

    @property
    def npatches(self):
        return self.config.dimensions[1]

    @property
    def ndurations(self):
        return self.config.dimensions[2]

    @property
    def nstarttimes(self):
        return self.config.dimensions[3]

    @property
    def nsamples(self):
        return self.config.dimensions[4]

    @property
    def starttime_sampling(self):
        return ut.scalar2floatX(
            self.config.starttime_sampling, tconfig.floatX)

    @property
    def duration_sampling(self):
        return ut.scalar2floatX(self.config.duration_sampling, tconfig.floatX)

    @property
    def duration_min(self):
        return ut.scalar2floatX(self.config.duration_min, tconfig.floatX)

    @property
    def starttime_min(self):
        return ut.scalar2floatX(self.config.starttime_min, tconfig.floatX)

    @property
    def filename(self):
        return get_gf_prefix(
            self.config.datatype, self.config.component,
            self.config.wave_config.name, self.config.crust_ind)


class FaultOrdering(object):
    """
    A mapping of source patches to the arrays of optimization results.

    Parameters
    ----------
    npls : list
        of number of patches in strike-direction
    npws : list
        of number of patches in dip-direction
    patch_size_strike : float
        patch size in strike-direction [km]
    patch_size_dip : float
        patch size in dip-direction [km]
    """

    def __init__(self, npls, npws, patch_size_strike, patch_size_dip):

        self.patch_size_dip = patch_size_dip
        self.patch_size_strike = patch_size_strike
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


class FaultGeometryError(Exception):
    pass


def positions2idxs(positions, cell_size, backend='numpy'):
    """
    Return index to a grid with a given cell size.npatches

    Parameters
    ----------
    positions : :class:`numpy.NdArray` float
        of positions [km]
    cell_size : float
        size of grid cells
    backend : str
    """
    available_backends = backends.keys()
    if backend not in available_backends:
        raise NotImplementedError(
            'Backend not supported! Options: %s' %
            ut.list2string(available_backends))

    return backends[backend].round((positions - (
        cell_size / 2.)) / cell_size).astype('int16')


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

    def __str__(self):
        s = '''
Complex Fault Geometry
number of subfaults: %i
number of patches: %i ''' % (
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

    def _assign_datatype(self, datatype):
        if datatype is None:
            return self.datatypes[0]
        else:
            return datatype

    def _assign_component(self, component):
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

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

        subfault = self.get_subfault(
            index, datatype=datatype, component=component)
        npw, npl = self.get_subfault_discretization(index)

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
        """ % ut.list2string(slip_directions.keys())

        datatype = self._assign_datatype(datatype)
        component = self._assign_component(component)

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
        self._check_index(index)
        return self.ordering.vmap[index].shp

    def point2starttimes(self, point, index=0):
        """
        Calculate starttimes for point in solution space.
        """

        nuc_dip = point['nucleation_dip']
        nuc_strike = point['nucleation_strike']
        velocities = point['velocities']

        nuc_dip_idx, nuc_strike_idx = self.fault_locations2idxs(
            nuc_dip, nuc_strike, backend='numpy')
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

        npw, npl = self.get_subfault_discretization(index)
        slownesses = 1. / rupture_velocities.reshape((npw, npl))

        start_times = fast_sweep.get_rupture_times_numpy(
            slownesses, self.ordering.patch_size_dip,
            n_patch_strike=npl, n_patch_dip=npw,
            nuc_x=nuc_strike_idx, nuc_y=nuc_dip_idx)
        return start_times

    def get_subfault_smoothing_operator(self, index):
        """
        Get second order Laplacian smoothing operator.

        This is beeing used to smooth the slip-distribution
        in the optimization.


        Returns
        -------
        :class:`numpy.Ndarray`
            (n_patch_strike + n_patch_dip) x (n_patch_strike + n_patch_dip)
        """

        npw, npl = self.get_subfault_discretization(index)
        return get_smoothing_operator(
            n_patch_strike=npl,
            n_patch_dip=npw,
            patch_size_strike=self.ordering.patch_size_strike * km,
            patch_size_dip=self.ordering.patch_size_dip * km)

    def fault_locations2idxs(
            self, positions_dip, positions_strike, backend='numpy'):
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

        dipidx = positions2idxs(
            positions=positions_dip,
            cell_size=self.ordering.patch_size_dip,
            backend=backend)
        strikeidx = positions2idxs(
            positions=positions_strike,
            cell_size=self.ordering.patch_size_strike,
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
    def npatches(self):
        return self.ordering.npatches


def discretize_sources(
        sources=None, extension_width=0.1, extension_length=0.1,
        patch_width=5., patch_length=5., datatypes=['geodetic'],
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
        Width [km] of subpatch in dip-direction
    patch_length : float
        Length [km] of subpatch in strike-direction
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

    patch_length_m = patch_length * km
    patch_width_m = patch_width * km

    npls = []
    npws = []
    for source in sources:
        s = copy.deepcopy(source)
        ext_source = s.extent_source(
            extension_width, extension_length,
            patch_width_m, patch_length_m)

        npls.append(int(num.ceil(ext_source.length / patch_length_m)))
        npws.append(int(num.ceil(ext_source.width / patch_width_m)))

    ordering = FaultOrdering(
        npls, npws, patch_size_strike=patch_length, patch_size_dip=patch_width)

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
                    patch_width_m, patch_length_m)

                npls.append(
                    ext_source.get_n_patches(patch_length_m, 'length'))
                npws.append(
                    ext_source.get_n_patches(patch_width_m, 'width'))
                ext_sources.append(ext_source)
                logger.info('Extended fault(s): \n %s' % ext_source.__str__())

            fault.setup_subfaults(datatype, var, ext_sources)

    return fault


def _process_patch_geodetic(
        engine, gfs, targets, patch, patchidx, los_vectors, odws):

    logger.info('Patch Number %i', patchidx)
    logger.debug('Calculating synthetics ...')
    disp = heart.geo_synthetics(
        engine=engine,
        targets=targets,
        sources=[patch],
        outmode='stacked_array')

    logger.debug('Applying LOS vector ...')
    los_disp = (disp * los_vectors).sum(axis=1) * odws

    gfs.put(entries=los_disp, patchidx=patchidx)


def geo_construct_gf_linear(
        engine, outdirectory, crust_ind=0, datasets=None,
        targets=None, fault=None, varnames=[''], force=False,
        event=None, nworkers=1):
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

    _, los_vectors, odws, _ = heart.concatenate_datasets(datasets)

    nsamples = odws.size
    npatches = fault.npatches
    logger.info('Using %i workers ...' % nworkers)

    for var in varnames:
        logger.info('For slip component: %s' % var)

        gfl_config = GeodeticGFLibraryConfig(
            component=var,
            dimensions=(npatches, nsamples),
            event=event,
            crust_ind=crust_ind,
            datatype='geodetic')
        gfs = GeodeticGFLibrary(config=gfl_config)

        outpath = os.path.join(outdirectory, gfs.filename + '.npz')
        if not os.path.exists(outpath) or force:
            if nworkers < 2:
                allocate = True
            else:
                allocate = False

            gfs.setup(npatches, nsamples, allocate=allocate)

            shared_gflibrary = RawArray('d', gfs.size)

            work = [
                (engine, gfs, targets, patch, patchidx, los_vectors, odws)
                for patchidx, patch in enumerate(
                    fault.get_all_patches('geodetic', component=var))]

            p = parallel.paripool(
                _process_patch_geodetic, work,
                initializer=_init_shared,
                initargs=(shared_gflibrary, None), nprocs=nworkers)

            for res in p:
                pass

            if nworkers > 1:
                # collect and store away
                gfs._gfmatrix = num.frombuffer(
                    shared_gflibrary).reshape(gfs.dimensions)

            logger.info('Storing geodetic linear GF Library ...')

            gfs.save(outdir=outdirectory)

        else:
            logger.info(
                'GF Library exists at path: %s. '
                'Use --force to overwrite!' % outpath)


def _process_patch_seismic(
        engine, gfs, targets, patch, patchidx, durations, starttimes):

    patch.time += gfs.config.event.time
    source_patches_durations = []
    logger.info('Patch Number %i', patchidx)

    for duration in durations:
        pcopy = patch.clone()
        pcopy.stf.duration = duration
        source_patches_durations.append(pcopy)

    for j, target in enumerate(targets):

        traces, _ = heart.seis_synthetics(
            engine=engine,
            sources=source_patches_durations,
            targets=[target],
            arrival_taper=None,
            wavename=gfs.config.wave_config.name,
            filterer=None,
            reference_taperer=None,
            outmode='data')

        # getting patch related arrival time for hypocenter
        arrival_time = heart.get_phase_arrival_time(
            engine=engine,
            source=patch,
            target=target,
            wavename=gfs.config.wave_config.name)

        ptmin = gfs.config.wave_config.arrival_taper.a + arrival_time
        gfs.set_patch_time(targetidx=j, patchidx=patchidx, tmin=ptmin)

        # getting event related arrival time valid for all patches
        # as common reference
        arrival_time = heart.get_phase_arrival_time(
            engine=engine,
            source=gfs.config.event,
            target=target,
            wavename=gfs.config.wave_config.name)

        ref_tmin = gfs.config.wave_config.arrival_taper.a + arrival_time
        gfs.set_patch_time(targetidx=j, patchidx=-1, tmin=ref_tmin)

        for starttime in starttimes:
            tmin = ref_tmin - starttime

            synthetics_array = heart.taper_filter_traces(
                traces=traces,
                arrival_taper=gfs.config.wave_config.arrival_taper,
                filterer=gfs.config.wave_config.filterer,
                tmins=num.ones(durations.size) * tmin,
                outmode='array')

            gfs.put(
                entries=synthetics_array,
                targetidx=j,
                patchidx=patchidx,
                durations=durations,
                starttimes=starttime)


def seis_construct_gf_linear(
        engine, fault, durations_prior, velocities_prior,
        varnames, wavemap, event, nworkers=1,
        starttime_sampling=1., duration_sampling=1.,
        sample_rate=1., outdirectory='./', force=False):
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
    durations_prior : :class:`heart.Parameter`
        prior of durations of the STF for each patch to convolve
    duration_sampling : float
        incremental step size for precalculation of duration GFs
    velocities_prior : :class:`heart.Parameter`
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

    # get starttimes for hypocenter at corner of fault
    start_times = fault.get_subfault_starttimes(
        index=0, rupture_velocities=velocities_prior.lower,
        nuc_dip_idx=0, nuc_strike_idx=0)

    starttimeidxs = num.arange(
        int(num.ceil(start_times.max() / starttime_sampling)))
    starttimes = starttimeidxs * starttime_sampling

    ndurations = ut.error_not_whole((
        (durations_prior.upper.max() -
         durations_prior.lower.min()) / duration_sampling),
        errstr='ndurations') + 1

    durations = num.linspace(
        durations_prior.lower.min(),
        durations_prior.upper.max(),
        ndurations)

    logger.info(
        'Calculating GFs for starttimes: %s \n durations: %s' %
        (ut.list2string(starttimes), ut.list2string(durations)))
    logger.info('Using %i workers ...' % nworkers)

    nstarttimes = len(starttimes)
    npatches = fault.npatches
    ntargets = len(wavemap.targets)
    nsamples = wavemap.config.arrival_taper.nsamples(sample_rate)

    for var in varnames:
        logger.info('For slip component: %s' % var)

        gfl_config = SeismicGFLibraryConfig(
            component=var,
            datatype='seismic',
            event=event,
            duration_sampling=duration_sampling,
            starttime_sampling=starttime_sampling,
            wave_config=wavemap.config,
            dimensions=(ntargets, npatches, ndurations, nstarttimes, nsamples),
            starttime_min=float(starttimes.min()),
            duration_min=float(durations.min()))

        gfs = SeismicGFLibrary(config=gfl_config)

        outpath = os.path.join(outdirectory, gfs.filename + '.npz')
        if not os.path.exists(outpath) or force:
            if nworkers < 2:
                allocate = True
            else:
                allocate = False

            gfs.setup(
                ntargets, npatches, ndurations,
                nstarttimes, nsamples, allocate=allocate)

            shared_gflibrary = RawArray('d', gfs.size)
            shared_times = RawArray('d', gfs.ntargets * (gfs.npatches + 1))

            work = [
                (engine, gfs, wavemap.targets,
                    patch, patchidx, durations, starttimes)
                for patchidx, patch in enumerate(
                    fault.get_all_patches('seismic', component=var))]

            p = parallel.paripool(
                _process_patch_seismic, work,
                initializer=_init_shared,
                initargs=(shared_gflibrary, shared_times), nprocs=nworkers)

            for res in p:
                pass

            if nworkers > 1:
                # collect and store away
                gfs._gfmatrix = num.frombuffer(
                    shared_gflibrary).reshape(gfs.dimensions)
                gfs._tmins = num.frombuffer(shared_times).reshape(
                    (gfs.ntargets, gfs.npatches + 1))

            logger.info('Storing seismic linear GF Library ...')

            gfs.save(outdir=outdirectory)
            del gfs

        else:
            logger.info(
                'GF Library exists at path: %s. '
                'Use --force to overwrite!' % outpath)


def _patch_locations(n_patch_strike, n_patch_dip):
    """
    Determines from patch locations the neighboring patches

    Parameters
    ----------
    n_patch_strike : int
        number of patches in strike direction
    n_patch_dip : int
        number of patches in dip direction

    Returns
    -------
    :class:`numpy.Ndarray`
        (n_patch_strike + n_patch_dip) x 4
    """
    n_patches = n_patch_dip * n_patch_strike

    zeros_strike = num.zeros(n_patch_strike)
    zeros_dip = num.zeros(n_patch_dip)

    dmat = num.ones((n_patches, 4))
    dmat[0:n_patch_strike, 0] = zeros_strike
    dmat[-n_patch_strike:, 1] = zeros_strike
    dmat[0::n_patch_strike, 2] = zeros_dip
    dmat[n_patch_strike - 1::n_patch_strike, 3] = zeros_dip
    return dmat


def get_smoothing_operator(
        n_patch_strike, n_patch_dip, patch_size_strike, patch_size_dip):
    """
    Get second order Laplacian smoothing operator.

    This is beeing used to smooth the slip-distribution in the optimization.

    Parameters
    ----------
    n_patch_strike : int
        number of patches in strike direction
    n_patch_dip : int
        number of patches in dip direction
    patch_size_strike : float
        size of patches along strike-direction [km]
    patch_size_dip : float
        size of patches along dip-direction [km]

    Returns
    -------
    :class:`numpy.Ndarray`
        (n_patch_strike + n_patch_dip) x (n_patch_strike + n_patch_dip)
    """
    n_patches = n_patch_dip * n_patch_strike

    dmat = _patch_locations(
        n_patch_strike=n_patch_strike, n_patch_dip=n_patch_dip)

    smooth_op = num.zeros((n_patches, n_patches))

    delta_l_dip = 1. / (patch_size_dip ** 2)
    delta_l_strike = 1. / (patch_size_strike ** 2)
    deltas = num.array(
        [delta_l_dip, delta_l_dip, delta_l_strike, delta_l_strike])

    for i in range(n_patches):
        flags = dmat[i, :]

        smooth_op[i, i] = -1 * flags.dot(deltas)

        if flags[0] == 1:
            smooth_op[i, i - n_patch_strike] = delta_l_dip
        if flags[1] == 1:
            smooth_op[i, i + n_patch_strike] = delta_l_dip
        if flags[2] == 1:
            smooth_op[i, i - 1] = delta_l_strike
        if flags[3] == 1:
            smooth_op[i, i + 1] = delta_l_strike

    return smooth_op
