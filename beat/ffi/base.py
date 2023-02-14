import logging
import os
from multiprocessing import RawArray

import numpy as num
import theano.tensor as tt
from pyrocko.guts import load
from pyrocko.trace import Trace
from theano import config as tconfig
from theano import shared

from beat import heart, parallel
from beat.config import GeodeticGFLibraryConfig, SeismicGFLibraryConfig
from beat.utility import error_not_whole, list2string, scalar2floatX

logger = logging.getLogger("ffi")

gf_dtype = "float64"

backends = {"numpy": num, "theano": tt}


def get_backend(backend):
    available_backends = backends.keys()
    if backend not in available_backends:
        raise NotImplementedError(
            "Backend not supported! Options: %s" % list2string(available_backends)
        )

    return backends[backend]


km = 1000.0


gf_entries = ["durations", "start_times", "patches", "targets"]


__all__ = [
    "load_gf_library",
    "get_gf_prefix",
    "GFLibrary",
    "GeodeticGFLibrary",
    "geo_construct_gf_linear",
    "geo_construct_gf_linear_patches",
    "SeismicGFLibrary",
    "seis_construct_gf_linear",
    "backends",
]


def _init_shared(gfstofill, tminstofill):
    logger.debug("Accessing shared arrays!")
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
        self._mode = "numpy"
        self._stack_switch = {}

    def _check_mode_init(self, mode):
        if mode == "theano":
            if self._sgfmatrix is None:
                raise GFLibraryError(
                    'To use "stack_all" theano stacking optimization mode'
                    " has to be initialised!"
                )

    def _check_setup(self):
        if sum(self.config.dimensions) == 0:
            raise GFLibraryError(
                "%s Greens Function Library is not set up!" % self.datatype
            )

    @property
    def size(self):
        return int(num.array(self.config.dimensions).prod())

    @property
    def filesize(self):
        """
        Size of the library in MByte.
        """
        return self.size * 8.0 / (1024.0**2)

    @property
    def patchidxs(self):
        if self._patchidxs is None:
            self._patchidxs = num.arange(self.npatches, dtype="int16")
        return self._patchidxs

    @property
    def sw_patchidxs(self):
        if self._mode == "numpy":
            return self.patchidxs
        elif self._mode == "theano":
            return self.spatchidxs

    def save_config(self, outdir="", filename=None):

        filename = filename or "%s" % self.filename
        outpath = os.path.join(outdir, filename + ".yaml")
        logger.debug("Dumping GF config to %s" % outpath)
        header = "beat.ffi.%s YAML Config" % self.__class__.__name__
        self.config.regularize()
        self.config.validate()
        self.config.dump(filename=outpath, header=header)

    def load_config(self, filename):

        try:
            config = load(filename=filename)
        except IOError:
            raise IOError("Cannot load config, file %s does not exist!" % filename)

        self.config = config

    def set_stack_mode(self, mode="numpy"):
        """
        Sets mode on witch backend the stacking is working.
        Dependent on that the input to the stack function has to be
        either of :class:`numpy.ndarray` or of :class:`theano.tensor.Tensor`

        Parameters
        ----------
        mode : str
            on witch array to stack
        """
        available_modes = backends.keys()
        if mode not in available_modes:
            raise GFLibraryError(
                "Stacking mode %s not available! "
                "Available modes: %s" % list2string(available_modes)
            )

        self._mode = mode

    def get_stack_mode(self):
        """
        Returns string of stack mode either "numpy" or "theano"
        """
        return self._mode


def get_gf_prefix(datatype, component, wavename, crust_ind):
    return "%s_%s_%s_%i" % (datatype, component, wavename, crust_ind)


def load_gf_library(directory="", filename=None):
    """
    Loading GF Library config and initialise memmaps for Traces and times.
    """
    inpath = os.path.join(directory, filename)
    datatype = filename.split("_")[0]

    if datatype == "seismic":
        gfs = SeismicGFLibrary()
        gfs.load_config(filename=inpath + ".yaml")
        gfs._gfmatrix = num.load(
            inpath + ".traces.npy", mmap_mode=("r"), allow_pickle=False
        )
        gfs._tmins = num.load(
            inpath + ".times.npy", mmap_mode=("r"), allow_pickle=False
        )

    elif datatype == "geodetic":
        gfs = GeodeticGFLibrary()
        gfs.load_config(filename=inpath + ".yaml")
        gfs._gfmatrix = num.load(
            inpath + ".traces.npy", mmap_mode=("r"), allow_pickle=False
        )

    else:
        raise ValueError('datatype "%s" not supported!' % datatype)

    gfs._stack_switch["numpy"] = gfs._gfmatrix
    return gfs


class GeodeticGFLibrary(GFLibrary):
    """
    Seismic Greens Function Library for the finite fault optimization.

    Parameters
    ----------
    config : :class:`GeodeticGFLibraryConfig`
    """

    def __init__(self, config=GeodeticGFLibraryConfig()):

        super(GeodeticGFLibrary, self).__init__(config=config)

        self._sgfmatrix = None

    def __str__(self):
        s = """
Geodetic GF Library
------------------
%s
npatches: %i
nsamples: %i
size: %i
filesize [MB]: %f
filename: %s""" % (
            self.config.dump(),
            self.npatches,
            self.nsamples,
            self.size,
            self.filesize,
            self.filename,
        )
        return s

    def save(self, outdir="", filename=None):
        """
        Save GFLibrary data and config file.
        """
        filename = filename or "%s" % self.filename
        outpath = os.path.join(outdir, filename)
        logger.info("Dumping GF Library to %s" % outpath)
        num.save(outpath + ".traces", arr=self._gfmatrix, allow_pickle=False)
        self.save_config(outdir=outdir, filename=filename)

    def setup(self, npatches, nsamples, allocate=False):

        self.dimensions = (npatches, nsamples)

        if allocate:
            logger.info("Allocating GF Library")
            self._gfmatrix = num.zeros(self.dimensions)

        self.set_stack_mode(mode="numpy")

    def init_optimization(self):

        logger.info("Setting %s GF Library to optimization mode." % self.filename)
        self._sgfmatrix = shared(
            self._gfmatrix.astype(tconfig.floatX), name=self.filename, borrow=True
        )
        parallel.memshare([self.filename])

        self.spatchidxs = shared(self.patchidxs, name="geo_patchidx_vec", borrow=True)

        self._stack_switch = {"numpy": self._gfmatrix, "theano": self._sgfmatrix}

        self.set_stack_mode(mode="theano")

    def put(self, entries, patchidx):
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
            raise ValueError("Entries have to be 1d arrays!")

        if entries.shape[0] != self.nsamples:
            raise GFLibraryError(
                "Trace length of entries is not consistent with the library"
                " to be filled! Entries length: %i Library: %i."
                % (entries.shape[0], self.nsamples)
            )

        self._check_setup()

        if hasattr(parallel, "gfmatrix"):
            matrix = num.frombuffer(parallel.gfmatrix).reshape(self.dimensions)

        elif self._gfmatrix is None:
            raise GFLibraryError("Neither shared nor standard GFLibrary is setup!")

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
        return self._stack_switch[self._mode].T.dot(slips)

    @property
    def nsamples(self):
        return self.config.dimensions[1]

    @property
    def npatches(self):
        return self.config.dimensions[0]

    @property
    def filename(self):
        return get_gf_prefix(
            self.config.datatype, self.config.component, "static", self.config.crust_ind
        )


class SeismicGFLibrary(GFLibrary):
    """
    Seismic Greens Function Library for the finite fault optimization.

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
        s = """
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
filename: %s""" % (
            self.config.dump(),
            self.ntargets,
            self.npatches,
            self.ndurations,
            self.nstarttimes,
            self.nsamples,
            self.size,
            self.filesize,
            self.filename,
        )
        return s

    def save(self, outdir="", filename=None):
        """
        Save GFLibrary data and config file.
        """
        filename = filename or "%s" % self.filename
        outpath = os.path.join(outdir, filename)
        logger.info("Dumping GF Library to %s" % outpath)
        num.save(outpath + ".traces", arr=self._gfmatrix, allow_pickle=False)
        num.save(outpath + ".times", arr=self._tmins, allow_pickle=False)
        self.save_config(outdir=outdir, filename=filename)

    def setup(
        self, ntargets, npatches, ndurations, nstarttimes, nsamples, allocate=False
    ):

        self.dimensions = (ntargets, npatches, ndurations, nstarttimes, nsamples)

        if allocate:
            logger.info("Allocating GF Library")
            self._gfmatrix = num.zeros(self.dimensions)
            self._tmins = num.zeros([ntargets])

        self.set_stack_mode(mode="numpy")

    def init_optimization(self):

        logger.info("Setting %s GF Library to optimization mode." % self.filename)
        self._sgfmatrix = shared(
            self._gfmatrix.astype(tconfig.floatX), name=self.filename, borrow=True
        )
        parallel.memshare([self.filename])

        self._stmins = shared(
            self._tmins.astype(tconfig.floatX),
            name=self.filename + "_tmins",
            borrow=True,
        )

        self.spatchidxs = shared(self.patchidxs, name="seis_patchidx_vec", borrow=True)

        self._stack_switch = {"numpy": self._gfmatrix, "theano": self._sgfmatrix}

        self.set_stack_mode(mode="theano")

    def set_patch_time(self, targetidx, tmin):
        """
        Fill the GF Library with trace times for one target and one patch.

        Parameters
        ----------
        targetidx : int
            index to target
        patchidx : int
            index to patch (source) that is assumed to be hypocenter
            patch_idx refers to reference time wrt event
        tmin : float
            tmin of the trace(s) if the hypocenter was in the location of this
            patch
        """

        if hasattr(parallel, "tmins"):
            times = num.frombuffer(parallel.tmins).reshape((self.ntargets))

        elif self._tmins is None:
            raise GFLibraryError("Neither shared nor standard GFLibrary is setup!")

        else:
            times = self._tmins

        times[targetidx] = tmin

    def put(self, entries, targetidx, patchidx, durations, starttimes):
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
        durations : list or :class:`numpy.NdArray`
            of the STFs that have been used to create the synthetics
        starttimes : list or :class:`numpy.NdArray`
            of the STFs that have been used to create the synthetics
        """

        if len(entries.shape) < 2:
            raise ValueError("Entries have to be 2d arrays!")

        if entries.shape[1] != self.nsamples:
            raise GFLibraryError(
                "Trace length of entries is not consistent with the library"
                " to be filled! Entries length: %i Library: %i."
                % (entries.shape[0], self.nsamples)
            )

        self._check_setup()

        durationidxs, _ = self.durations2idxs(durations)
        starttimeidxs, _ = self.starttimes2idxs(starttimes)

        if hasattr(parallel, "gfmatrix"):
            matrix = num.frombuffer(parallel.gfmatrix).reshape(self.dimensions)

        elif self._gfmatrix is None:
            raise GFLibraryError("Neither shared nor standard GFLibrary is setup!")

        else:
            matrix = self._gfmatrix

        logger.debug("targetidx %i, patchidx %i" % (targetidx, patchidx))

        matrix[targetidx, patchidx, durationidxs, starttimeidxs, :] = entries

    def trace_tmin(self, targetidx):
        """
        Returns trace time of single target with respect to reference event.
        """
        return float(self.reference_times[targetidx])

    def starttimes2idxs(self, starttimes, interpolation="nearest_neighbor"):
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
        starttimeidxs, starttimes : :class:`numpy.ndarray` or
            :class:`theano.tensor.Tensor`, int16
            (output depends on interpolation scheme,
            if multilinear interpolation factors are returned as well)
        """
        backend = get_backend(self._mode)

        if interpolation == "nearest_neighbor":
            return (
                backend.round(
                    (starttimes - self.starttime_min) / self.starttime_sampling
                ).astype("int16"),
                None,
            )
        elif interpolation == "multilinear":
            dstarttimes = (starttimes - self.starttime_min) / self.starttime_sampling
            ceil_starttimes = backend.ceil(dstarttimes).astype("int16")
            factors = ceil_starttimes - dstarttimes
            return ceil_starttimes, factors
        else:
            raise NotImplementedError(
                "Interpolation scheme %s not implemented!" % interpolation
            )

    def idxs2durations(self, idxs):
        """
        Map index to durations [s]
        """
        return idxs * self.duration_sampling + self.duration_min

    def idxs2starttimes(self, idxs):
        """
        Map index to starttimes [s]
        """
        return idxs * self.starttime_sampling + self.starttime_min

    def durations2idxs(self, durations, interpolation="nearest_neighbor"):
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
        durationidxs, starttimes : :class:`numpy.ndarray` or
            :class:`theano.tensor.Tensor`, int16
        """
        backend = get_backend(self._mode)

        if interpolation == "nearest_neighbor":
            return (
                backend.round(
                    (durations - self.duration_min) / self.duration_sampling
                ).astype("int16"),
                None,
            )
        elif interpolation == "multilinear":
            ddurations = (durations - self.duration_min) / self.duration_sampling
            ceil_durations = backend.ceil(ddurations).astype("int16")
            factors = ceil_durations - ddurations
            return ceil_durations, factors
        else:
            raise NotImplementedError(
                "Interpolation scheme %s not implemented!" % interpolation
            )

    def stack(
        self,
        targetidx,
        patchidxs,
        durations,
        starttimes,
        slips,
        interpolation="nearest_neighbor",
    ):
        """
        Stack selected traces from the GF Library of specified
        target, patch, durations and starttimes. Numpy or theano dependent
        on the stack_mode

        Parameters
        ----------

        Returns
        -------
        :class:`numpy.ndarray` or of :class:`theano.tensor.Tensor` dependent
        on stack mode
        """
        durationidxs, rt_factors = self.durations2idxs(
            durations, interpolation=interpolation
        )
        starttimeidxs, st_factors = self.starttimes2idxs(
            starttimes, interpolation=interpolation
        )

        return (
            self._stack_switch[self._mode][
                targetidx, patchidxs, durationidxs, starttimeidxs, :
            ]
            .reshape((slips.shape[0], self.nsamples))
            .T.dot(slips)
        )

    def stack_all(
        self,
        durations,
        starttimes,
        slips,
        targetidxs=None,
        patchidxs=None,
        interpolation="nearest_neighbor",
    ):
        """
        Stack all patches for all targets at once.
        In theano for efficient optimization.

        Parameters
        ----------
        starttimes: numpy or theano tensor
            size (ntargets, npatches) to be able to account for time-shifts!

        Returns
        -------
        matrix : size (ntargets, nsamples)
        option : tensor.batched_dot(sd.dimshuffle((1,0,2)), u).sum(axis=0)
        """

        if targetidxs is None:
            raise ValueError("Target indexes have to be defined!")

        if patchidxs is None:
            patchidxs = self.sw_patchidxs
            npatches = self.npatches
        else:
            npatches = len(patchidxs)

        self._check_mode_init(self._mode)
        backend = get_backend(self._mode)

        durationidxs, rt_factors = self.durations2idxs(
            durations, interpolation=interpolation
        )
        starttimeidxs, st_factors = self.starttimes2idxs(
            starttimes, interpolation=interpolation
        )

        if interpolation == "nearest_neighbor":

            cd = (
                self._stack_switch[self._mode][
                    targetidxs, patchidxs, durationidxs, starttimeidxs, :
                ]
                .reshape((self.ntargets, npatches, self.nsamples))
                .T
            )

            cslips = backend.tile(slips, self.ntargets).reshape(
                (self.ntargets, npatches)
            )

        elif interpolation == "multilinear":

            d_st_ceil_rt_ceil = self._stack_switch[self._mode][
                targetidxs, patchidxs, durationidxs, starttimeidxs, :
            ].reshape((self.ntargets, npatches, self.nsamples))
            d_st_floor_rt_ceil = self._stack_switch[self._mode][
                targetidxs, patchidxs, durationidxs, starttimeidxs - 1, :
            ].reshape((self.ntargets, npatches, self.nsamples))
            d_st_ceil_rt_floor = self._stack_switch[self._mode][
                targetidxs, patchidxs, durationidxs - 1, starttimeidxs, :
            ].reshape((self.ntargets, npatches, self.nsamples))
            d_st_floor_rt_floor = self._stack_switch[self._mode][
                targetidxs, patchidxs, durationidxs - 1, starttimeidxs - 1, :
            ].reshape((self.ntargets, npatches, self.nsamples))

            s_st_ceil_rt_ceil = (1 - st_factors) * (1 - rt_factors) * slips
            s_st_floor_rt_ceil = st_factors * (1.0 - rt_factors) * slips
            s_st_ceil_rt_floor = (1 - st_factors) * rt_factors * slips
            s_st_floor_rt_floor = st_factors * rt_factors * slips

            cd = backend.concatenate(
                [
                    d_st_ceil_rt_ceil,
                    d_st_floor_rt_ceil,
                    d_st_ceil_rt_floor,
                    d_st_floor_rt_floor,
                ],
                axis=1,
            ).T  # T
            cslips = backend.concatenate(
                [
                    s_st_ceil_rt_ceil,
                    s_st_floor_rt_ceil,
                    s_st_ceil_rt_floor,
                    s_st_floor_rt_floor,
                ],
                axis=1,
            )  #

        else:
            raise NotImplementedError(
                "Interpolation scheme %s not implemented!" % interpolation
            )

        if self._mode == "theano":
            return tt.batched_dot(cd.dimshuffle((2, 0, 1)), cslips)

        elif self._mode == "numpy":
            return num.einsum("ijk->ik", cd * cslips.T).T

    def get_traces(
        self, targetidxs=[0], patchidxs=[0], durationidxs=[0], starttimeidxs=[0]
    ):
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
                            targetidx, patchidx, durationidx, starttimeidx, :
                        ]
                        tr = Trace(
                            ydata=ydata,
                            deltat=self.deltat,
                            network="target_%i" % targetidx,
                            station="patch_%i" % patchidx,
                            channel="tau_%.2f" % self.idxs2durations(durationidx),
                            location="t0_%.2f" % self.idxs2starttimes(starttimeidx),
                            tmin=self.trace_tmin(targetidx),
                        )
                        traces.append(tr)

        return traces

    @property
    def reference_times(self):
        """
        Returns tmins for all targets for specified hypocentral patch.
        """
        return self._tmins + self.config.wave_config.arrival_taper.b

    @property
    def deltat(self):
        return self.config.wave_config.arrival_taper.duration(["b", "c"]) / float(
            self.nsamples
        )

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
        return scalar2floatX(self.config.starttime_sampling, tconfig.floatX)

    @property
    def duration_sampling(self):
        return scalar2floatX(self.config.duration_sampling, tconfig.floatX)

    @property
    def duration_min(self):
        return scalar2floatX(self.config.duration_min, tconfig.floatX)

    @property
    def starttime_min(self):
        return scalar2floatX(self.config.starttime_min, tconfig.floatX)

    @property
    def filename(self):
        return get_gf_prefix(
            self.config.datatype,
            self.config.component,
            self.config._mapid,
            self.config.crust_ind,
        )


def _process_patch_geodetic(engine, gfs, targets, patch, patchidx, los_vectors, odws):

    logger.debug("Patch Number %i", patchidx)
    logger.debug("Calculating synthetics ...")

    disp = heart.geo_synthetics(
        engine=engine, targets=targets, sources=[patch], outmode="stacked_array"
    )

    logger.debug("Applying LOS vector ...")
    los_disp = (disp * los_vectors).sum(axis=1) * odws
    if isinstance(gfs, GeodeticGFLibrary):
        gfs.put(entries=los_disp, patchidx=patchidx)
    elif gfs is None and hasattr(parallel, "gfmatrix"):
        npatches = len(parallel.gfmatrix) // los_disp.size
        matrix = num.frombuffer(parallel.gfmatrix).reshape((npatches, los_disp.size))
        matrix[patchidx, :] = los_disp
    else:
        raise ValueError("GF Library not allocated!")


def geo_construct_gf_linear(
    engine,
    outdirectory,
    crust_ind=0,
    datasets=None,
    targets=None,
    fault=None,
    varnames=[""],
    force=False,
    event=None,
    nworkers=1,
):
    """
    Create geodetic Greens Function Library for defined source geometry.

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

    logger.info("Using %i workers ..." % nworkers)

    for var in varnames:
        logger.info("For slip component: %s" % var)

        gfl_config = GeodeticGFLibraryConfig(
            component=var,
            dimensions=(npatches, nsamples),
            event=event,
            crust_ind=crust_ind,
            datatype="geodetic",
            reference_sources=fault.get_all_subfaults(
                datatype="geodetic", component=var
            ),
        )
        gfs = GeodeticGFLibrary(config=gfl_config)

        outpath = os.path.join(outdirectory, gfs.filename + ".npz")

        if os.path.exists(outpath) and not force:
            logger.info(
                "Library exists: %s. " "Please use --force to override!" % outpath
            )

        else:
            if nworkers < 2:
                allocate = True
            else:
                allocate = False

            gfs.setup(npatches, nsamples, allocate=allocate)

            if outdirectory:
                logger.info(
                    "Setting up Green's Function Library: %s \n ", gfs.__str__()
                )

            parallel.check_available_memory(gfs.filesize)

            shared_gflibrary = RawArray("d", gfs.size)

            patches = fault.get_all_patches("geodetic", component=var)
            work = [
                (engine, gfs, targets, patch, patchidx, los_vectors, odws)
                for patchidx, patch in enumerate(patches)
            ]

            p = parallel.paripool(
                _process_patch_geodetic,
                work,
                initializer=_init_shared,
                initargs=(shared_gflibrary, None),
                nprocs=nworkers,
            )

            for res in p:
                pass

            # collect and store away
            gfs._gfmatrix = num.frombuffer(shared_gflibrary).reshape(gfs.dimensions)

            if outdirectory:
                logger.info("Storing geodetic linear GF Library ...")
                gfs.save(outdir=outdirectory)
            else:
                return gfs._gfmatrix


def geo_construct_gf_linear_patches(
    engine,
    datasets=None,
    targets=None,
    patches=None,
    nworkers=1,
    apply_weight=False,
    return_mapping=False,
):
    """
    Create geodetic Greens Function matrix for given patches.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        main path to directory containing the different Greensfunction stores
    datasets : list
        of :class:`heart.GeodeticDataset` for which the GFs are calculated
    targets : list
        of :class:`heart.GeodeticDataset`
    patches : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry
    nworkers : int
        number of CPUs to use for processing
    return_mapping : bool
        return array mapping as well
    """

    _, los_vectors, odws, Bij = heart.concatenate_datasets(datasets)

    nsamples = odws.size
    npatches = len(patches)

    logger.debug("Using %i workers ..." % nworkers)

    shared_gflibrary = RawArray("d", npatches * nsamples)

    work = [
        (engine, None, targets, patch, patchidx, los_vectors, odws)
        for patchidx, patch in enumerate(patches)
    ]

    p = parallel.paripool(
        _process_patch_geodetic,
        work,
        initializer=_init_shared,
        initargs=(shared_gflibrary, None),
        nprocs=nworkers,
    )

    for res in p:
        pass

    # collect and store away
    gfmatrix = num.frombuffer(shared_gflibrary).reshape((npatches, nsamples))

    if apply_weight:
        list_gfs = Bij.a_nd2l(gfmatrix)
        w_gfs = []
        for dataset, gfs in zip(datasets, list_gfs):
            w_gfs.append(gfs.dot(dataset.covariance.chol_inverse))

        gfmatrix = num.hstack(w_gfs)

    if not return_mapping:
        return gfmatrix
    else:
        return gfmatrix, Bij


def _process_patch_seismic(
    engine, gfs, targets, patch, patchidx, durations, starttimes
):

    # ensur event reference time
    logger.debug("Using reference event source time ...")
    patch.time = gfs.config.event.time

    # ensure stf anchor point at -1
    patch.stf.anchor = -1
    source_patches_durations = []
    logger.info("Patch Number %i", patchidx)

    for duration in durations:
        pcopy = patch.clone()
        pcopy.stf.duration = float(duration)
        source_patches_durations.append(pcopy)

    for j, target in enumerate(targets):

        traces, _ = heart.seis_synthetics(
            engine=engine,
            sources=source_patches_durations,
            targets=[target],
            arrival_taper=None,
            arrival_times=num.array(None),
            wavename=gfs.config.wave_config.name,
            filterer=None,
            reference_taperer=None,
            outmode="data",
        )

        # getting event related arrival time valid for all patches
        # as common reference
        event_arrival_time = heart.get_phase_arrival_time(
            engine=engine,
            source=gfs.config.event,
            target=target,
            wavename=gfs.config.wave_config.name,
        )

        gfs.set_patch_time(targetidx=j, tmin=event_arrival_time)

        for starttime in starttimes:
            shifted_arrival_time = event_arrival_time - starttime

            synthetics_array = heart.taper_filter_traces(
                traces=traces,
                arrival_taper=gfs.config.wave_config.arrival_taper,
                filterer=gfs.config.wave_config.filterer,
                arrival_times=num.ones(durations.size) * shifted_arrival_time,
                outmode="array",
                chop_bounds=["b", "c"],
            )

            gfs.put(
                entries=synthetics_array,
                targetidx=j,
                patchidx=patchidx,
                durations=durations,
                starttimes=starttime,
            )


def seis_construct_gf_linear(
    engine,
    fault,
    durations_prior,
    velocities_prior,
    nucleation_time_prior,
    varnames,
    wavemap,
    event,
    nworkers=1,
    time_shift=None,
    starttime_sampling=1.0,
    duration_sampling=1.0,
    sample_rate=1.0,
    outdirectory="./",
    force=False,
):
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
        configuration parameters for handling seismic data around Phase
    fault : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry
    durations_prior : :class:`heart.Parameter`
        prior of durations of the STF for each patch to convolve
    velocities_prior : :class:`heart.Parameter`
        rupture velocity of earthquake prior
    nucleation_time_prior : :class:`heart.Parameter`
        prior of nucleation time of the event
    starttime_sampling : float
        incremental step size for precalculation of startime GFs
    duration_sampling : float
        incremental step size for precalculation of duration GFs
    sample_rate : float
        sample rate of synthetic traces to produce,
        related to non-linear GF store
    time_shift : hierarchical parameter or None
    outpath : str
        directory for storage
    force : boolean
        flag to overwrite existing linear GF Library
    """

    if wavemap.config.domain == "spectrum":
        raise TypeError("FFI is currently only supported for time-domain!")

    # get starttimes for hypocenter at corner of fault
    st_mins = []
    st_maxs = []
    for idx, sf in enumerate(fault.iter_subfaults()):
        try:
            rupture_velocities = fault.vector2subfault(
                idx, velocities_prior.get_lower(fault.subfault_npatches)
            )
        except (IndexError):
            raise ValueError(
                "Velocities need to be of size either"
                " npatches or number of fault segments"
            )

        start_times = fault.get_subfault_starttimes(
            index=idx,
            rupture_velocities=rupture_velocities,
            nuc_dip_idx=0,
            nuc_strike_idx=0,
        )
        if time_shift is not None:
            shift_times_min = time_shift.lower.min()
            shift_times_max = time_shift.upper.max()
        else:
            shift_times_min = shift_times_max = 0.0

        st_mins.append(start_times.min() + shift_times_min)
        st_maxs.append(start_times.max() + shift_times_max)

    starttimeidxs = num.arange(
        int(
            num.floor(min(st_mins) + nucleation_time_prior.lower.min())
            / starttime_sampling
        ),
        int(
            num.ceil(max(st_maxs) + nucleation_time_prior.upper.max())
            / starttime_sampling
        )
        + 1,
    )
    starttimes = starttimeidxs * starttime_sampling

    durations_max = durations_prior.get_upper(fault.subfault_npatches).max()
    durations_min = durations_prior.get_lower(fault.subfault_npatches).min()
    ndurations = (
        error_not_whole(
            ((durations_max - durations_min) / duration_sampling), errstr="ndurations"
        )
        + 1
    )

    durations = num.linspace(durations_min, durations_max, ndurations)

    logger.info(
        "Calculating GFs for starttimes: %s \n durations: %s"
        % (list2string(starttimes), list2string(durations))
    )
    logger.info("Using %i workers ..." % nworkers)

    nstarttimes = len(starttimes)
    npatches = fault.npatches
    ntargets = len(wavemap.targets)
    nsamples = wavemap.config.arrival_taper.nsamples(sample_rate)

    for var in varnames:
        logger.info("For slip component: %s" % var)
        gfl_config = SeismicGFLibraryConfig(
            component=var,
            datatype="seismic",
            event=event,
            reference_sources=fault.get_all_subfaults(
                datatype="seismic", component=var
            ),
            duration_sampling=duration_sampling,
            starttime_sampling=starttime_sampling,
            wave_config=wavemap.config,
            dimensions=(ntargets, npatches, ndurations, nstarttimes, nsamples),
            starttime_min=float(starttimes.min()),
            duration_min=float(durations.min()),
            mapnumber=wavemap.mapnumber,
        )

        gfs = SeismicGFLibrary(config=gfl_config)

        outpath = os.path.join(outdirectory, gfs.filename + ".npz")

        if os.path.exists(outpath) and not force:
            logger.info(
                "Library exists: %s. " "Please use --force to override!" % outpath
            )
        else:
            if nworkers < 2:
                allocate = True
            else:
                allocate = False

            gfs.setup(
                ntargets, npatches, ndurations, nstarttimes, nsamples, allocate=allocate
            )

            logger.info("Setting up Green's Function Library: %s \n ", gfs.__str__())

            parallel.check_available_memory(gfs.filesize)

            shared_gflibrary = RawArray("d", gfs.size)
            shared_times = RawArray("d", gfs.ntargets)

            work = [
                (engine, gfs, wavemap.targets, patch, patchidx, durations, starttimes)
                for patchidx, patch in enumerate(
                    fault.get_all_patches("seismic", component=var)
                )
            ]

            p = parallel.paripool(
                _process_patch_seismic,
                work,
                initializer=_init_shared,
                initargs=(shared_gflibrary, shared_times),
                nprocs=nworkers,
            )

            for res in p:
                pass

            # collect and store away
            gfs._gfmatrix = num.frombuffer(shared_gflibrary).reshape(gfs.dimensions)
            gfs._tmins = num.frombuffer(shared_times).reshape((gfs.ntargets))

            logger.info("Storing seismic linear GF Library ...")

            gfs.save(outdir=outdirectory)
            del gfs
