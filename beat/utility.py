"""
This module provides a namespace for various functions:
coordinate transformations,
loading and storing objects,
book-keeping of indexes in arrays that relate to defined variable names,
manipulation of various pyrocko objects
and many more ...
"""

import collections
import copy
import logging
import os
import pickle
import re
from functools import wraps
from timeit import Timer

import numpy as num
from pyrocko import catalog, orthodrome, util
from pyrocko.cake import LayeredModel, m2d, read_nd_model_str
from pyrocko.gf.seismosizer import RectangularSource
from pyrocko.guts import Float, Int, Object
from theano import config as tconfig

logger = logging.getLogger("utility")

DataMap = collections.namedtuple("DataMap", "list_ind, slc, shp, dtype, name")

locationtypes = {"east_shift", "north_shift", "depth", "distance", "delta_depth"}
dimensiontypes = {"length", "width", "diameter"}
mttypes = {"mnn", "mee", "mdd", "mne", "mnd", "med"}
degtypes = {"strike", "dip", "rake"}
nucleationtypes = {"nucleation_x", "nucleation_y"}
patch_anchor_points = {"center", "bottom_depth", "bottom_left"}

kmtypes = set.union(locationtypes, dimensiontypes, patch_anchor_points)
grouped_vars = set.union(kmtypes, mttypes, degtypes, nucleationtypes)

unit_sets = {
    "locationtypes": locationtypes,
    "dimensiontypes": dimensiontypes,
    "mttypes": mttypes,
    "degtypes": degtypes,
    "nucleationtypes": nucleationtypes,
}

seconds_str = "00:00:00"

sphr = 3600.0
hrpd = 24.0

d2r = num.pi / 180.0
km = 1000.0


def argsorted(seq, reverse=False):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    # by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


class Counter(object):
    """
    Counts calls of types with string_ids. Repeated calls with the same
    string id increase the count.
    """

    def __init__(self):
        self.d = dict()

    def __call__(self, string, multiplier=1):

        if string not in self.d:
            self.d[string] = 0
        else:
            self.d[string] += 1 * multiplier
        return self.d[string]

    def __getitem__(self, key):
        try:
            return self.d[key]
        except ValueError:
            raise KeyError(
                'type "%s" is not listed in the counter!'
                " Counted types are: %s" % (key, list2string(list(self.d.keys())))
            )

    def reset(self, string=None):
        if string is None:
            self.d = dict()
        else:
            self.d[string] = 0


class ListArrayOrdering(object):
    """
    An ordering for a list to an array space. Takes also non theano.tensors.
    Modified from pymc3 blocking.

    Parameters
    ----------
    list_arrays : list
        :class:`numpy.ndarray` or :class:`theano.tensor.Tensor`
    intype : str
        defining the input type 'tensor' or 'numpy'
    """

    def __init__(self, list_arrays, intype="numpy"):
        self.vmap = []
        dim = 0

        count = 0
        for array in list_arrays:
            if intype == "tensor":
                name = array.name
                array = array.tag.test_value
            elif intype == "numpy":
                name = "numpy"

            slc = slice(dim, dim + array.size)
            vm = DataMap(count, slc, array.shape, array.dtype, name)
            self.vmap.append(vm)
            dim += array.size
            count += 1

        self.size = dim
        self._keys = None

    def __getitem__(self, key):
        try:
            return self.vmap[self.variables.index(key)]
        except ValueError:
            raise KeyError(
                'Variable "%s" is not in the mapping!'
                " Mapped Variables: %s" % (key, list2string(self.variables))
            )

    def __iter__(self):
        return iter(self.variables)

    @property
    def variables(self):
        if self._keys is None:
            self._keys = [vmap.name for vmap in self.vmap]

        return self._keys


class ListToArrayBijection(object):
    """
    A mapping between a List of arrays and an array space

    Parameters
    ----------
    ordering : :class:`ListArrayOrdering`
    list_arrays : list
        of :class:`numpy.ndarray`
    """

    def __init__(self, ordering, list_arrays, blacklist=[]):
        self.ordering = ordering
        self.list_arrays = list_arrays
        self.dummy = -9.0e40
        self.blacklist = blacklist

    def d2l(self, dpt):
        """
        Maps values from dict space to List space
        If variable expected from ordering is not in point
        it is filled with a low dummy value -999999.

        Parameters
        ----------
        dpt : list
            of :class:`numpy.ndarray`

        Returns
        -------
        lpoint
        """

        a_list = copy.copy(self.list_arrays)

        for list_ind, _, shp, _, var in self.ordering.vmap:
            try:
                a_list[list_ind] = dpt[var].ravel()
            except KeyError:
                # Needed for initialisation of chain_l_point in Metropolis
                a_list[list_ind] = num.atleast_1d(num.ones(shp) * self.dummy).ravel()

        return a_list

    def l2d(self, a_list):
        """
        Maps values from List space to dict space

        Parameters
        ----------
        list_arrays : list
            of :class:`numpy.ndarray`

        Returns
        -------
        :class:`pymc3.model.Point`
        """
        point = {}

        for list_ind, _, _, _, var in self.ordering.vmap:
            if var not in self.blacklist:
                point[var] = a_list[list_ind].ravel()

        return point

    def l2a(self, list_arrays):
        """
        Maps values from List space to array space

        Parameters
        ----------
        list_arrays : list
            of :class:`numpy.ndarray`

        Returns
        -------
        array : :class:`numpy.ndarray`
            single array comprising all the input arrays
        """

        array = num.empty(self.ordering.size)
        for list_ind, slc, _, _, _ in self.ordering.vmap:
            array[slc] = list_arrays[list_ind].ravel()
        return array

    def f3map(self, list_arrays):
        """
        Maps values from List space to array space with 3 columns

        Parameters
        ----------
        list_arrays : list
            of :class:`numpy.ndarray` with size: n x 3

        Returns
        -------
        array : :class:`numpy.ndarray`
            single array comprising all the input arrays
        """

        array = num.empty((self.ordering.size, 3))
        for list_ind, slc, _, _, _ in self.ordering.vmap:
            array[slc, :] = list_arrays[list_ind]
        return array

    def a2l(self, array):
        """
        Maps value from array space to List space
        Inverse operation of fmap.

        Parameters
        ----------
        array : :class:`numpy.ndarray`

        Returns
        -------
        a_list : list
            of :class:`numpy.ndarray`
        """

        a_list = copy.copy(self.list_arrays)

        for list_ind, slc, shp, dtype, _ in self.ordering.vmap:
            try:
                a_list[list_ind] = num.atleast_1d(array)[slc].reshape(shp).astype(dtype)
            except ValueError:  # variable does not exist in array use dummy
                a_list[list_ind] = num.atleast_1d(num.ones(shp) * self.dummy).ravel()

        return a_list

    def a_nd2l(self, array):
        """
        Maps value from ndarray space (ndims, data) to List space
        Inverse operation of fmap. Nd

        Parameters
        ----------
        array : :class:`numpy.ndarray`

        Returns
        -------
        a_list : list
            of :class:`numpy.ndarray`
        """

        a_list = copy.copy(self.list_arrays)
        nd = array.ndim
        if nd != 2:
            raise ValueError(
                "Input array has wrong dimensions! Needed 2d array! Got %i" % nd
            )

        for list_ind, slc, shp, dtype, _ in self.ordering.vmap:
            shpnd = (array.shape[0],) + shp
            try:
                a_list[list_ind] = (
                    num.atleast_2d(array)[:, slc].reshape(shpnd).astype(dtype)
                )
            except ValueError:  # variable does not exist in array use dummy
                a_list[list_ind] = num.atleast_2d(num.ones(shpnd) * self.dummy)

        return a_list

    def srmap(self, tarray):
        """
        Maps value from symbolic variable array space to List space

        Parameters
        ----------
        tarray : :class:`theano.tensor.Tensor`

        Returns
        -------
        a_list : list
            of :class:`theano.tensor.Tensor`
        """

        a_list = copy.copy(self.list_arrays)

        for list_ind, slc, shp, dtype, _ in self.ordering.vmap:
            a_list[list_ind] = tarray[slc].reshape(shp).astype(dtype.name)

        return a_list


def weed_input_rvs(input_rvs, mode, datatype):
    """
    Throw out random variables (RV)s from input list that are not included by
    the respective synthetics generating functions.

    Parameters
    ----------
    input_rvs : dict
        of :class:`pymc3.Distribution` or set of variable names
    mode : str
        'geometry', 'static, 'kinematic', 'interseismic' determining the
        discarded RVs
    datatype : str
        'seismic' or 'geodetic' determining the discarded RVs

    Returns
    -------
    weeded_input_rvs : dict
        of :class:`pymc3.Distribution`
    """

    weeded_input_rvs = copy.copy(input_rvs)

    burian = """
        lat lon name stf stf1 stf2 stf_mode moment anchor
        velocity interpolation decimation_factor npointsources
        elevation exponent aggressive_oversampling
        """.split()

    if mode == "geometry":
        if datatype == "geodetic":
            tobeweeded = [
                "time",
                "duration",
                "delta_time",
                "nucleation_x",
                "nucleation_y",
                "peak_ratio",
            ] + burian
        elif datatype == "seismic":
            tobeweeded = ["opening"] + burian
        elif datatype == "polarity":
            tobeweeded = [
                "time",
                "duration",
                "magnitude",
                "peak_ratio",
                "slip",
                "opening_fraction",
                "nucleation_x",
                "nucleation_y",
                "length",
                "width",
            ] + burian

    elif mode == "interseismic":
        if datatype == "geodetic":
            tobeweeded = burian

    else:
        tobeweeded = []

    for weed in tobeweeded:
        if isinstance(weeded_input_rvs, dict):
            if weed in weeded_input_rvs.keys():
                weeded_input_rvs.pop(weed)

        elif isinstance(weeded_input_rvs, set):
            weeded_input_rvs.discard(weed)

        else:
            raise TypeError(
                "Variables are not of proper format: %s !" % weeded_input_rvs.__class__
            )

    return weeded_input_rvs


def apply_station_blacklist(stations, blacklist):
    """
    Weed stations listed in the blacklist.

    Parameters
    ----------
    stations : list
        :class:`pyrocko.model.Station`
    blacklist : list
        strings of station names

    Returns
    -------
    stations : list of :class:`pyrocko.model.Station`
    """

    outstations = []
    for st in stations:
        station_name = get_ns_id((st.network, st.station))
        if station_name not in blacklist:
            outstations.append(st)
    return outstations


def weed_data_traces(data_traces, stations):
    """
    Throw out data traces belonging to stations that are not in the
    stations list. Keeps list orders!

    Parameters
    ----------
    data_traces : list
        of :class:`pyrocko.trace.Trace`
    stations : list
        of :class:`pyrocko.model.Station`

    Returns
    -------
    weeded_data_traces : list
        of :class:`pyrocko.trace.Trace`
    """

    station_names = [
        get_ns_id((station.network, station.station)) for station in stations
    ]

    weeded_data_traces = []

    for tr in data_traces:
        trace_name = get_ns_id(tr.nslc_id)
        if trace_name in station_names:
            weeded_data_traces.append(tr)

    return weeded_data_traces


def weed_targets(targets, stations, discard_targets=[]):
    """
    Throw out targets belonging to stations that are not in the
    stations list. Keeps list orders and returns new list!

    Parameters
    ----------
    targets : list
        of :class:`pyrocko.gf.targets.Target`
    stations : list
        of :class:`pyrocko.model.Station`

    Returns
    -------
    weeded_targets : list
        of :class:`pyrocko.gf.targets.Target`
    """
    station_names = [
        get_ns_id((station.network, station.station)) for station in stations
    ]

    weeded_targets = []
    for target in targets:
        target_name = get_ns_id((target.codes[0], target.codes[1]))
        if target_name in station_names:
            if target in discard_targets:
                pass
            else:
                weeded_targets.append(target)

    return weeded_targets


def downsample_trace(data_trace, deltat=None, snap=False):
    """
    Downsample data_trace to given sampling interval 'deltat'.

    Parameters
    ----------
    data_trace : :class:`pyrocko.trace.Trace`
    deltat : sampling interval [s] to which trace should be downsampled

    Returns
    -------
    :class:`pyrocko.trace.Trace`
        new instance
    """
    tr = data_trace.copy()
    if deltat is not None:
        if num.abs(tr.deltat - deltat) > 1.0e-6:
            try:
                tr.downsample_to(deltat, snap=snap, allow_upsample_max=5, demean=False)
                tr.deltat = deltat
                if snap:
                    tr.snap()

            except util.UnavailableDecimation as e:
                logger.error("Cannot downsample %s.%s.%s.%s: %s" % (tr.nslc_id + (e,)))
        elif snap:
            if tr.tmin / tr.deltat > 1e-6 or tr.tmax / tr.deltat > 1e-6:
                tr.snap()
    else:
        raise ValueError("Need to provide target sample rate!")

    return tr


def weed_stations(stations, event, distances=(30.0, 90.0), remove_duplicate=False):
    """
    Weed stations, that are not within the given distance range(min, max) to
    a reference event.

    Parameters
    ----------
    stations : list
        of :class:`pyrocko.model.Station`
    event
        :class:`pyrocko.model.Event`
    distances : tuple
        of minimum and maximum distance [deg] for station-event pairs

    Returns
    -------
    weeded_stations : list
        of :class:`pyrocko.model.Station`
    """

    weeded_stations = []
    logger.debug("Valid distance range: [%f, %f]!" % (distances[0], distances[1]))
    check_duplicate = []
    for station in stations:
        distance = orthodrome.distance_accurate50m(event, station) * m2d
        logger.debug("Distance of station %s: %f [deg]" % (station.station, distance))
        if distance >= distances[0] and distance <= distances[1]:
            logger.debug("Inside defined distance range!")
            ns_str = get_ns_id((station.network, station.station))
            if ns_str in check_duplicate and remove_duplicate:
                logger.warning(
                    "Station %s already in wavemap! Multiple "
                    "locations not supported yet! "
                    "Discarding duplicate ..." % ns_str
                )
            else:
                weeded_stations.append(station)
                check_duplicate.append(ns_str)
        else:
            logger.debug("Outside defined distance range!")

    return weeded_stations


def transform_sources(sources, datatypes, decimation_factors=None):
    """
    Transforms a list of :py:class:`heart.RectangularSource` to a dictionary of
    sources :py:class:`pscmp.PsCmpRectangularSource` for geodetic data and
    :py:class:`pyrocko.gf.seismosizer.RectangularSource` for seismic data.

    Parameters
    ----------
    sources : list
        :class:`heart.RectangularSource`
    datatypes : list
        of strings with the datatypes to be included 'geodetic' or 'seismic'
    decimation_factors : dict
        of datatypes and their respective decimation factor

    Returns
    -------
    d : dict
        of transformed sources with datatypes as keys
    """

    d = dict()

    for datatype in datatypes:
        transformed_sources = []

        for source in sources:
            transformed_source = copy.deepcopy(source)

            if decimation_factors is not None:
                transformed_source.update(
                    decimation_factor=decimation_factors[datatype], anchor="top"
                )

            if datatype == "geodetic" or datatype == "polarity":
                transformed_source.stf = None

            transformed_sources.append(transformed_source)

        d[datatype] = transformed_sources

    return d


def adjust_point_units(point):
    """
    Transform variables with [km] units to [m]

    Parameters
    ----------
    point : dict
        :func:`pymc3.model.Point` of model parameter units as keys

    Returns
    -------
    mpoint : dict
        :func:`pymc3.model.Point`
    """

    mpoint = {}
    for key, value in point.items():
        if key in kmtypes:
            mpoint[key] = value * km
        else:
            mpoint[key] = value

    return mpoint


def split_point(point):
    """
    Split point in solution space into List of dictionaries with source
    parameters for each source.

    Parameters
    ----------
    point : dict
        :func:`pymc3.model.Point`

    Returns
    -------
    source_points : list
        of :func:`pymc3.model.Point`
    """
    params = point.keys()
    if len(params) > 0:
        n_sources = point[next(iter(params))].shape[0]
    else:
        n_sources = 0

    source_points = []
    for i in range(n_sources):
        source_param_dict = dict()
        for param, value in point.items():
            source_param_dict[param] = float(value[i])

        source_points.append(source_param_dict)

    return source_points


def join_points(ldicts):
    """
    Join list of dicts into one dict with concatenating
    values of keys that are present in multiple dicts.
    """

    keys = set([k for d in ldicts for k in d.keys()])

    jpoint = {}
    for k in keys:
        jvar = []
        for d in ldicts:
            jvar.append(d[k])

        jpoint[k] = num.array(jvar)

    return jpoint


def check_point_keys(point, phrase):
    """
    Searches point keys for a phrase, returns list of keys with the phrase.
    """
    from fnmatch import fnmatch

    keys = list(point.keys())

    contains = False
    contained_keys = []
    for k in keys:
        if fnmatch(k, phrase):
            contains = True
            contained_keys.append(k)

    return contains, contained_keys


def update_source(source, **point):
    """
    Update source keeping stf and source params separate.
    Modifies input source Object!

    Parameters
    ----------
    source : :class:`pyrocko.gf.seismosizer.Source`
    point : dict
        :func:`pymc3.model.Point`
    """

    for (k, v) in point.items():
        if k not in source.keys():
            if source.stf is not None:
                try:
                    source.stf[k] = float(v)
                except (KeyError, TypeError):
                    logger.warning("Not updating source with %s" % k)
            else:
                raise AttributeError(
                    "Please set a STF before updating its" " parameters."
                )
        else:
            source[k] = float(v)


def setup_logging(project_dir, levelname, logfilename="BEAT_log.txt"):
    """
    Setup function for handling BEAT logging. The logfile 'BEAT_log.txt' is
    saved in the 'project_dir'.

    Parameters
    ----------
    project_dir : str
        absolute path to the output directory for the Log file
    levelname : str
        defining the level of logging
    """

    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    filename = os.path.join(project_dir, logfilename)

    logger = logging.getLogger()
    # remove existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # setup file handler
    fhandler = logging.FileHandler(filename=filename, mode="a")
    fformatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fhandler.setFormatter(fformatter)
    fhandler.setLevel(levels[levelname])
    logger.addHandler(fhandler)

    # setup screen handler
    console = logging.StreamHandler()
    console.setLevel(levels[levelname])
    cformatter = logging.Formatter("%(name)-12s - %(levelname)-8s %(message)s")
    console.setFormatter(cformatter)
    logger.addHandler(console)
    logger.setLevel(levels[levelname])


def search_catalog(date, min_magnitude, dayrange=1.0):
    """
    Search the gcmt catalog for the specified date (+- 1 day), filtering the
    events with given magnitude threshold.

    Parameters
    ----------
    date : str
        'YYYY-MM-DD', date of the event
    min_magnitude : float
        approximate minimum Mw of the event
    dayrange : float
        temporal search interval [days] around date

    Returns
    -------
    event : :class:`pyrocko.model.Event`
    """

    gcmt = catalog.GlobalCMT()

    time_s = util.stt(date + " " + seconds_str)
    d1 = time_s - (dayrange * (sphr * hrpd))
    d2 = time_s + (dayrange * (sphr * hrpd))

    logger.info(
        "Getting relevant events from the gCMT catalog for the dates:"
        "%s - %s \n" % (util.tts(d1), util.tts(d2))
    )

    events = gcmt.get_events((d1, d2), magmin=min_magnitude)

    if len(events) < 1:
        logger.warn("Found no event information in the gCMT catalog.")
        event = None

    if len(events) > 1:
        logger.info(
            "More than one event from that date with specified magnitude "
            "found! Please copy the relevant event information to the "
            "configuration file!"
        )
        for event in events:
            print(event)

        event = events[0]

    elif len(events) == 1:
        event = events[0]

    return event


def RS_dipvector(source):
    """
    Get 3 dimensional dip-vector of a planar fault.

    Parameters
    ----------
    source : RectangularSource

    Returns
    -------
    :class:`numpy.ndarray`
    """

    return num.array(
        [
            num.cos(source.dip * d2r) * num.cos(source.strike * d2r),
            -num.cos(source.dip * d2r) * num.sin(source.strike * d2r),
            num.sin(source.dip * d2r),
        ]
    )


def strike_vector(strike, order="ENZ"):
    if order == "ENZ":
        return num.array([num.sin(strike * d2r), num.cos(strike * d2r), 0.0])
    elif order == "NEZ":
        return num.array([num.cos(strike * d2r), num.sin(strike * d2r), 0.0])
    else:
        raise Exception("Order %s not implemented!" % order)


def RS_strikevector(source):
    """
    Get 3 dimensional strike-vector of a planar fault.

    Parameters
    ----------
    source : RedctangularSource

    Returns
    -------
    :class:`numpy.ndarray`
    """

    return strike_vector(source.strike)


def RS_center(source):
    """
    Get 3d fault center coordinates. Depth attribute is top depth!

    Parameters
    ----------
    source : RedctangularSource

    Returns
    -------
    :class:`numpy.ndarray` with x, y, z coordinates of the center of the
    fault
    """

    return num.array(
        [source.east_shift, source.north_shift, source.depth]
    ) + 0.5 * source.width * RS_dipvector(source)


def adjust_fault_reference(source, input_depth="top"):
    """
    Adjusts source depth and east/north-shifts variables of fault according to
    input_depth mode 'top/center'.

    Parameters
    ----------
    source : :class:`RectangularSource` or :class:`pscmp.RectangularSource` or
        :class:`pyrocko.gf.seismosizer.RectangularSource`
    input_depth : string
        if 'top' the depth in the source is interpreted as top depth
        if 'center' the depth in the source is interpreted as center depth

    Returns
    -------
    Updated input source object
    """

    if input_depth == "top":
        center = RS_center(source)
    elif input_depth == "center":
        center = num.array([source.east_shift, source.north_shift, source.depth])
    else:
        raise Exception("input_depth %s not supported!" % input_depth)

    source.east_shift = float(center[0])
    source.north_shift = float(center[1])
    source.depth = float(center[2])


def dump_objects(outpath, outlist):
    """
    Dump objects in outlist into pickle file.

    Parameters
    ----------
    outpath : str
        absolute path and file name for the file to be stored
    outlist : list
        of objects to save pickle
    """

    with open(outpath, "wb") as f:
        pickle.dump(outlist, f, protocol=4)


def load_objects(loadpath):
    """
    Load (unpickle) saved (pickled) objects from specified loadpath.

    Parameters
    ----------
    loadpath : absolute path and file name to the file to be loaded

    Returns
    -------
    objects : list
        of saved objects
    """

    try:
        objects = pickle.load(open(loadpath, "rb"))
    except UnicodeDecodeError:
        objects = pickle.load(open(loadpath, "rb"), encoding="latin1")
    except IOError:
        raise Exception("File %s does not exist!" % loadpath)
    return objects


def ensure_cov_psd(cov):
    """
    Ensure that the input covariance matrix is positive definite.
    If not, find the nearest positive semi-definite matrix.

    Parameters
    ----------
    cov : :class:`numpy.ndarray`
        symmetric covariance matrix

    Returns
    -------
    cov : :class:`numpy.ndarray`
        positive definite covariance matrix
    """

    try:
        num.linalg.cholesky(cov)
    except num.linalg.LinAlgError:
        logger.debug("Cov_pv not positive definite!" " Finding nearest psd matrix...")
        cov = repair_covariance(cov)

    return cov


def near_psd(x, epsilon=num.finfo(num.float64).eps):
    """
    Calculates the nearest positive semi-definite matrix for a correlation/
    covariance matrix

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Covariance/correlation matrix
    epsilon : float
        Eigenvalue limit
        here set to accuracy of numbers in numpy, otherwise the resulting
        matrix, likely is still not going to be positive definite

    Returns
    -------
    near_cov : :class:`numpy.ndarray`
        closest positive definite covariance/correlation matrix

    Notes
    -----
    Numpy number precision not high enough to resolve this for low valued
    covariance matrixes! The result will have very small negative eigvals!!!

    See repair_covariance below for a simpler implementation that can resolve
    the numbers!

    Algorithm after Rebonato & Jaekel 1999
    """

    if min(num.linalg.eigvals(x)) > epsilon:
        return x

    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    scaling = num.sqrt(num.diag(x))
    a, b = num.meshgrid(scaling, scaling)
    y = x / (a * b)

    # getting the nearest correlation matrix
    eigval, eigvec = num.linalg.eigh(y)
    val = num.maximum(eigval, epsilon)
    vec = num.matrix(eigvec)
    T = 1.0 / (num.multiply(vec, vec) * val.T)
    T = num.matrix(num.sqrt(num.diag(num.array(T).reshape((n)))))
    B = T * vec * num.diag(num.array(num.sqrt(val)).reshape((n)))
    near_corr = num.array(B * B.T)

    # returning the scaling factors
    return near_corr * a * b


def repair_covariance(x, epsilon=num.finfo(num.float64).eps):
    """
    Make covariance input matrix A positive definite.
    Setting eigenvalues that are lower than the  of numpy floats to
    at least that precision and backtransform.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Covariance/correlation matrix
    epsilon : float
        Eigenvalue limit
        here set to accuracy of numbers in numpy, otherwise the resulting
        matrix, likely is still not going to be positive definite

    Returns
    -------
    near_cov : :class:`numpy.ndarray`
        closest positive definite covariance/correlation matrix

    Notes
    -----
    Algorithm after Gilbert Strange, 'Introduction to linear Algebra'
    """

    eigval, eigvec = num.linalg.eigh(x)
    val = num.maximum(eigval, epsilon)
    return eigvec.dot(num.diag(val)).dot(eigvec.T)


def running_window_rms(data, window_size, mode="valid"):
    """
    Calculate the standard deviations of a running window over data.

    Parameters
    ----------
    data : :class:`numpy.ndarray` 1-d
        containing data to calculate stds from
    window_size : int
        sample size of running window
    mode : str
        see numpy.convolve for modes

    Returns
    -------
    :class:`numpy.ndarray` 1-d
        with stds, size data.size - window_size + 1
    """
    data2 = num.power(data, 2)
    window = num.ones(window_size) / float(window_size)
    return num.sqrt(num.convolve(data2, window, mode))


def slice2string(slice_obj):
    """
    Wrapper for better formatted string method for slices.

    Returns
    -------
    str
    """
    if isinstance(slice_obj, slice):
        if slice_obj.step:
            return "{}:{}:{}".format(slice_obj.start, slice_obj.stop, slice_obj.step)
        else:
            return "{}:{}".format(slice_obj.start, slice_obj.stop)
    else:
        return slice_obj


def list2string(l, fill=", "):
    """
    Convert list of string to single string.

    Parameters
    ----------
    l: list
        of strings
    """
    return fill.join("%s" % slice2string(listentry) for listentry in l)


def string2slice(slice_string):
    """
    Convert string of slice form to python slice object.

    Parameters
    ----------
    slice_string: str
        of form "0:2" i.e. two integer numbers separated by colon
    """

    return slice(*[int(idx) for idx in slice_string.split(":")])


def unique_list(l):
    """
    Find unique entries in list and return them in a list.
    Keeps variable order.

    Parameters
    ----------
    l : list

    Returns
    -------
    list with only unique elements
    """
    used = []
    return [x for x in l if x not in used and (used.append(x) or True)]


def join_models(global_model, crustal_model):
    """
    Replace the part of the 'global model' that is covered by 'crustal_model'.

    Parameters
    ----------
    global_model : :class:`pyrocko.cake.LayeredModel`
    crustal_model : :class:`pyrocko.cake.LayeredModel`

    Returns
    -------
    joined_model : cake.LayeredModel
    """

    max_depth = crustal_model.max("z")

    cut_model = global_model.extract(depth_min=max_depth)
    joined_model = copy.deepcopy(crustal_model)

    for element in cut_model.elements():
        joined_model.append(element)

    return joined_model


def split_off_list(l, off_length):
    """
    Split a list with length 'off_length' from the beginning of an input
    list l.
    Modifies input list!

    Parameters
    ----------
    l : list
        of objects to be separated
    off_length : int
        number of elements from l to be split off

    Returns
    -------
    list
    """

    return [l.pop(0) for i in range(off_length)]


def mod_i(i, cycle):
    """
    Calculates modulus of a function and returns number of full cycles and the
    rest.

    Parameters
    ----------
    i : int or float
        Number to be cycled over
    cycle : int o float
        Cycle length

    Returns
    -------
    fullc : int or float depending on input
    rest : int or float depending on input
    """
    fullc = i // cycle
    rest = i % cycle
    return fullc, rest


def biggest_common_divisor(a, b):
    """
    Find the biggest common divisor of two float numbers a and b.

    Parameters
    ----------
    a, b: float

    Returns
    -------
    int
    """

    while b > 0:
        rest = a % b
        a = b
        b = rest

    return int(a)


def gather(l, key, sort=None, filter=None):
    """
    Return dictionary of input l grouped by key.
    """
    d = {}
    for x in l:
        if filter is not None and not filter(x):
            continue

        k = key(x)
        if k not in d:
            d[k] = []

        d[k].append(x)

    if sort is not None:
        for v in d.values():
            v.sort(key=sort)

    return d


def get_fit_indexes(llk):
    """
    Find indexes of various likelihoods in a likelihood distribution.

    Parameters
    ----------
    llk : :class:`numpy.ndarray`

    Returns
    -------
    dict with array indexes
    """
    mean_idx = (num.abs(llk - llk.mean())).argmin()
    min_idx = (num.abs(llk - llk.min())).argmin()
    max_idx = (num.abs(llk - llk.max())).argmin()

    posterior_idxs = {"mean": mean_idx, "min": min_idx, "max": max_idx}

    return posterior_idxs


def check_hyper_flag(problem):
    """
    Check problem setup for type of model standard/hyperparameters.

    Parameters
    ----------
    :class:`models.Problem`

    Returns
    -------
    flag : boolean
    """

    if os.path.basename(problem.outfolder) == "hypers":
        return True
    else:
        return False


def error_not_whole(f, errstr=""):
    """
    Test if float is a whole number, if not raise Error.
    """
    if f.is_integer():
        return int(f)
    else:
        raise ValueError("%s : %f is not a whole number!" % (errstr, f))


def scalar2floatX(a, floatX=tconfig.floatX):
    if floatX == "float32":
        return num.float32(a)
    elif floatX == "float64":
        return num.float64(a)


def scalar2int(a, floatX=tconfig.floatX):
    if floatX == "float32":
        return num.int16(a)
    elif floatX == "float64":
        return num.int64(a)


def PsGrnArray2LayeredModel(psgrn_input_path):
    """
    Read PsGrn Input file and return velocity model.

    Parameters
    ----------
    psgrn_input_path : str
        Absolute path to the psgrn input file.

    Returns
    -------
    :class:`LayeredModel`
    """
    a = num.loadtxt(psgrn_input_path, skiprows=136)
    b = a[:, 1:-1]
    b[:, 3] /= 1000.0
    return LayeredModel.from_scanlines(
        read_nd_model_str(
            re.sub(
                "[\[\]]",
                "",
                num.array2string(
                    b, precision=4, formatter={"float_kind": lambda x: "%.3f" % x}
                ),
            )
        )
    )


def swap_columns(array, index1, index2):
    """
    Swaps the column of the input array based on the given indexes.
    """
    array[:, index1], array[:, index2] = array[:, index2], array[:, index1].copy()
    return array


def line_intersect(e1, e2, n1, n2):
    """
    Get intersection point of n-lines.

    Parameters
    ----------
    end points of each line in (n x 2) arrays
    e1 : :class:`numpy.array` (n x 2)
        east coordinates of first line
    e2 : :class:`numpy.array` (n x 2)
        east coordinates of second line
    n1 : :class:`numpy.array` (n x 2)
        north coordinates of first line
    n2 : :class:`numpy.array` (n x 2)
        east coordinates of second line

    Returns
    -------
    :class:`numpy.array` (n x 2) of intersection points (easts, norths)
    """
    perp = num.array([[0, -1], [1, 0]])
    de = num.atleast_2d(e2 - e1)
    dn = num.atleast_2d(n2 - n1)
    dp = num.atleast_2d(e1 - n1)
    dep = num.dot(de, perp)
    denom = num.sum(dep * dn, axis=1)

    if denom == 0:
        logger.warn("Lines are parallel! No intersection point!")
        return None

    tmp = num.sum(dep * dp, axis=1)
    return num.atleast_2d(tmp / denom).T * dn + n1


def get_rotation_matrix(axes=["x", "y", "z"]):
    """
    Return a function for 3-d rotation matrix for a specified axis.

    Parameters
    ----------
    axes : str or list of str
        x, y or z for the axis

    Returns
    -------
    func that takes an angle [rad]
    """
    ax_avail = ["x", "y", "z"]
    for ax in axes:
        if ax not in ax_avail:
            raise TypeError(
                "Rotation axis %s not supported!"
                " Available axes: %s" % (ax, list2string(ax_avail))
            )

    def rotx(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[1, 0, 0], [0, cos_angle, -sin_angle], [0, sin_angle, cos_angle]],
            dtype="float64",
        )

    def roty(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]],
            dtype="float64",
        )

    def rotz(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]],
            dtype="float64",
        )

    R = {"x": rotx, "y": roty, "z": rotz}

    if isinstance(axes, list):
        return R
    elif isinstance(axes, str):
        return R[axes]
    else:
        raise Exception("axis has to be either string or list of strings!")


def get_random_uniform(lower, upper, dimension=1):
    """
    Get uniform random values between given bounds

    Parameters
    ==========
    lower : float
    upper : float
    dimension : size of result vector
    """
    values = (upper - lower) * num.random.rand(dimension) + lower
    if dimension == 1:
        return float(values)
    else:
        return values


def positions2idxs(positions, cell_size, min_pos=0.0, backend=num, dtype="int16"):
    """
    Return index to a grid with a given cell size.npatches

    Parameters
    ----------
    positions : :class:`numpy.NdArray` float
        of positions [km]
    cell_size : float
        size of grid cells
    backend : str
    dtype : str
        data type of returned array, default: int16
    """
    return backend.round((positions - min_pos - (cell_size / 2.0)) / cell_size).astype(
        dtype
    )


def rotate_coords_plane_normal(coords, sf):

    coords -= sf.bottom_left / km

    rots = get_rotation_matrix()
    rotz = coords.dot(rots["z"](d2r * -sf.strike))
    roty = rotz.dot(rots["y"](d2r * -sf.dip))

    roty[:, 0] *= -1.0
    return roty


def get_ns_id(nslc_id):
    return "{}.{}".format(nslc_id[0], nslc_id[1])


def time_method(loop=10000):
    def timer_decorator(func):
        @wraps(func)
        def wrap_func(*args, **kwargs):
            total_time = Timer(lambda: func(*args, **kwargs)).timeit(number=loop)
            print(
                "Method {name} run {loop} times".format(name=func.__name__, loop=loop)
            )
            print(
                "It took: {time} s, Mean: {mean_time} s".format(
                    mean_time=total_time / loop, time=total_time
                )
            )
            # return func(*args, **kwargs)

        return wrap_func

    return timer_decorator


def is_odd(value):
    return (value & 1) == 1


def is_even(value):
    return (value & 1) == 0


def get_valid_spectrum_data(deltaf, taper_frequencies=[0, 1.0]):
    """extract valid frequency range of spectrum"""
    lower_f, upper_f = taper_frequencies

    lower_idx = int(num.floor(lower_f / deltaf))
    upper_idx = int(num.ceil(upper_f / deltaf))
    return lower_idx, upper_idx


def get_data_radiant(data):
    """
    Data needs to be [n, 2]
    """
    return num.arctan2(
        data[:, 1].max() - data[:, 1].min(), data[:, 0].max() - data[:, 0].min()
    )


def find_elbow(data, theta=None, rotate_left=False):
    """
    Get point closest to turning point in data by rotating it by theta.

    Adapted from:
    https://datascience.stackexchange.com/questions/57122/in-elbow-curve-
    how-to-find-the-point-from-where-the-curve-starts-to-rise

    Parameters
    ----------
    data : array like,
        [n, 2]
    theta : rotation angle

    Returns
    -------
    Index : int
        closest to elbow.
    rotated_data : array-like [n, 2]
    """
    if theta is None:
        theta = get_data_radiant(data)

    if rotate_left:
        theta = 2 * num.pi - theta

    # make rotation matrix
    co = num.cos(theta)
    si = num.sin(theta)
    rotation_matrix = num.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_data = data.dot(rotation_matrix)
    return rotated_data[:, 1].argmin(), rotated_data


class StencilOperator(Object):

    h = Float.T(default=0.1, help="step size left and right of the reference value")
    order = Int.T(default=3, help="number of points of central differences")

    def __init__(self, **kwargs):

        stencil_order = kwargs["order"]
        if stencil_order not in [3, 5]:
            raise ValueError(
                "Only stencil orders 3 and 5 implemented."
                " Requested: %i" % stencil_order
            )

        self._coeffs = {3: num.array([1.0, -1.0]), 5: num.array([1.0, 8.0, -8.0, -1.0])}

        self._denominator = {3: 2.0, 5: 12.0}

        self._hsteps = {3: num.array([-1, 1]), 5: num.array([-2, -1, 1, 2])}

        Object.__init__(self, **kwargs)

    @property
    def coefficients(self):
        coeffs = self._coeffs[self.order]
        return coeffs.reshape((coeffs.size, 1, 1))

    def __len__(self):
        return self.coefficients.size

    @property
    def denominator(self):
        return self._denominator[self.order] * self.h

    @property
    def hsteps(self):
        return self._hsteps[self.order] * self.h


def distances(points, ref_points):
    """
    Calculate distances in Cartesian coordinates between points and reference
    points in N-D.

    Parameters
    ----------
    points: :class:`numpy.Ndarray` (n points x n spatial dimensions)
    ref_points: :class:`numpy.Ndarray` (m points x n spatial dimensions)

    Returns
    -------
    ndarray (n_points x n_ref_points)
    """
    nref_points = ref_points.shape[0]
    ndim = points.shape[1]
    ndim_ref = ref_points.shape[1]
    if ndim != ndim_ref:
        raise TypeError(
            "Coordinates to calculate differences must have the same number "
            "of dimensions! Given dimensions are {} and {}".format(ndim, ndim_ref)
        )

    points_rep = num.tile(points, nref_points).reshape(
        points.shape[0], nref_points, ndim
    )

    distances = num.sqrt(num.power(points_rep - ref_points, 2).sum(axis=2))
    return distances
