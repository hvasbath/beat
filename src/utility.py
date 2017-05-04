import logging
import os
import re
import collections
import copy
import cPickle as pickle

from pyrocko import util, orthodrome, catalog
from pyrocko.cake import m2d, LayeredModel, read_nd_model_str

from pyrocko.gf.seismosizer import RectangularSource

import numpy as num
from theano import config as tconfig

from pyproj import Proj


logger = logging.getLogger('utility')

DataMap = collections.namedtuple('DataMap', 'list_ind, slc, shp, dtype')
PatchMap = collections.namedtuple(
    'PatchMap', 'count, slc, shp, npatches')

kmtypes = set(['east_shift', 'north_shift', 'length', 'width', 'depth',
               'distance', 'delta_depth'])

seconds_str = '00:00:00'

sphr = 3600.
hrpd = 24.

d2r = num.pi / 180.
km = 1000.


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

    def __init__(self, list_arrays, intype='numpy'):
        self.vmap = []
        dim = 0

        count = 0
        for array in list_arrays:
            if intype == 'tensor':
                array = array.tag.test_value
            elif intype == 'numpy':
                pass

            slc = slice(dim, dim + array.size)
            self.vmap.append(DataMap(
                count, slc, array.shape, array.dtype))
            dim += array.size
            count += 1

        self.dimensions = dim


class ListToArrayBijection(object):
    """
    A mapping between a List of arrays and an array space

    Parameters
    ----------
    ordering : :class:`ListArrayOrdering`
    list_arrays : list
        of :class:`numpy.ndarray`
    """

    def __init__(self, ordering, list_arrays):
        self.ordering = ordering
        self.list_arrays = list_arrays

    def fmap(self, list_arrays):
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

        array = num.empty(self.ordering.dimensions)
        for list_ind, slc, _, _ in self.ordering.vmap:
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

        array = num.empty((self.ordering.dimensions, 3))
        for list_ind, slc, _, _ in self.ordering.vmap:
            array[slc, :] = list_arrays[list_ind]
        return array

    def rmap(self, array):
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

        for list_ind, slc, shp, dtype in self.ordering.vmap:
            a_list[list_ind] = num.atleast_1d(
                                        array)[slc].reshape(shp).astype(dtype)

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

        for list_ind, slc, shp, dtype in self.ordering.vmap:
            a_list[list_ind] = tarray[slc].reshape(shp).astype(dtype.name)

        return a_list


def weed_input_rvs(input_rvs, mode, datatype):
    """
    Throw out random variables (RV)s from input list that are not included by
    the respective synthetics generating functions.

    Parameters
    ----------
    input_rvs : dict
        of :class:`pymc3.Distribution`
        or set
            of variable names
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

    burian = '''
        lat lon name stf stf1 stf2 stf_mode moment anchor nucleation_x sign
        nucleation_y velocity interpolation decimation_factor npointsources
        '''.split()

    if mode == 'geometry':
        if datatype == 'geodetic':
            tobeweeded = ['time', 'duration', 'delta_time'] + burian
        elif datatype == 'seismic':
            tobeweeded = ['opening'] + burian

    elif mode == 'interseismic':
        if datatype == 'geodetic':
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
            raise TypeError('Variables are not of proper format: %s !' % \
                weeded_input_rvs.__class__)

    return weeded_input_rvs


def apply_station_blacklist(stations, blacklist):
    """
    Weed stations listed in the blacklist.
    Modifies input list!

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

    station_names = [station.station for station in stations]

    indexes = []
    for burian in blacklist:
        try:
            indexes.append(station_names.index(burian))
        except ValueError:
            logger.info('Station %s in blacklist is not in stations.' % burian)

    if len(indexes) > 0:
        indexes.sort(reverse=True)

        for ind in indexes:
            stations.pop(ind)

    return stations


def weed_data_traces(data_traces, stations):
    """
    Throw out data traces belonging to stations that are not in the
    stations list.

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

    station_names = [station.station for station in stations]

    weeded_data_traces = []

    for tr in data_traces:
        if tr.station in station_names:
            weeded_data_traces.append(tr.copy())

    return weeded_data_traces


def downsample_traces(data_traces, deltat=None):
    """
    Downsample data_traces to given sampling interval 'deltat'.
    Modifies input :class:`pyrocko.trace.Trace` Objects!

    Parameters
    ----------
    data_traces : list
        of :class:`pyrocko.trace.Trace`
    deltat : sampling interval [s] to which traces should be downsampled
    """

    for tr in data_traces:
        if deltat is not None:
            try:
                tr.downsample_to(deltat, snap=True, allow_upsample_max=5)
            except util.UnavailableDecimation, e:
                print('Cannot downsample %s.%s.%s.%s: %s' % (
                                                            tr.nslc_id + (e,)))
                continue


def weed_stations(stations, event, distances=(30., 90.)):
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
    for station in stations:
        distance = orthodrome.distance_accurate50m(event, station) * m2d

        if distance >= distances[0] and distance <= distances[1]:
            weeded_stations.append(station)

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
                    decimation_factor=decimation_factors[datatype])

            if datatype == 'geodetic':
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
    for key, value in point.iteritems():
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
        n_sources = point[params[0]].shape[0]
    else:
        n_sources = 0

    source_points = []
    for i in range(n_sources):
        source_param_dict = dict()
        for param, value in point.iteritems():
            source_param_dict[param] = float(value[i])

        source_points.append(source_param_dict)

    return source_points


def update_source(source, **kwargs):
    """
    Update source keeping stf and source params seperate.
    Modifies input source Object!

    Parameters
    ----------
    source : :class:`pyrocko.gf.seismosizer.Source`
    point : dict
        :func:`pymc3.model.Point`
    """

    for (k, v) in kwargs.iteritems():
        if k not in source.keys():
            if source.stf is not None:
                source.stf[k] = v
            else:
                raise Exception('Please set a STF before updating its'
                                    ' parameters.')
        else:
            source[k] = v

    if isinstance(source, RectangularSource):
        adjust_fault_reference(source, input_depth='top')


def utm_to_loc(utmx, utmy, zone, event):
    """
    Convert UTM[m] to local coordinates with reference to the
    :class:`pyrocko.model.Event`

    Parameters
    ----------
    utmx : :class:`numpy.ndarray`
        with UTM easting
    utmy : :class:`numpy.ndarray`
        with UTM northing
    zone : int
        number with utm zone
    event : :class:`pyrocko.model.Event`

    Returns
    -------
    locx : :class:`numpy.ndarray`
        Local coordinates [m] for x direction (East)
    locy : :class:`numpy.ndarray`
        Local coordinates [m] for y direction (North)
    """

    p = Proj(proj='utm', zone=zone, ellps='WGS84')
    ref_x, ref_y = p(event.lon, event.lat)
    locx = utmx - ref_x
    locy = utmy - ref_y
    return locx, locy


def lonlat_to_utm(lon, lat, zone):
    """
    Convert UTM[m] to local coordinates with reference to the
    :class:`pyrocko.model.Event`

    Parameters
    ----------
    utmx : :class:`numpy.ndarray`
        with UTM easting
    utmy : :class:`numpy.ndarray`
        with UTM northing
    zone : int
        number with utm zone

    Returns
    -------
    utme : :class:`numpy.ndarray`
        Local coordinates [m] for x direction (East)
    utmn : :class:`numpy.ndarray`
        Local coordinates [m] for y direction (North)
    """

    p = Proj(proj='utm', zone=zone, ellps='WGS84')
    utme, utmn = p(lon, lat)
    return utme, utmn


def utm_to_lonlat(utmx, utmy, zone):
    """
    Convert UTM[m] to Latitude and Longitude coordinates.

    Parameters
    ----------
    utmx : :class:`numpy.ndarray`
        with UTM easting
    utmy : :class:`numpy.ndarray`
        with UTM northing
    zone : int
        number with utm zone

    Returns
    -------
    lon : :class:`numpy.ndarray` Longitude [decimal deg]
    lat : :class:`numpy.ndarray` Latitude [decimal deg]
    """

    p = Proj(proj='utm', zone=zone, ellps='WGS84')
    lon, lat = p(utmx, utmy, inverse=True)
    return lon, lat


def setup_logging(project_dir, levelname):
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

    levels = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}

    logging.basicConfig(
        level=levels[levelname],
        format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
        filename=os.path.join(project_dir, 'BEAT_log.txt'),
        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(levels[levelname])

    formatter = logging.Formatter('%(name)-12s - %(levelname)-8s %(message)s')

    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)


def search_catalog(date, min_magnitude, dayrange=1.):
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

    time_s = util.stt(date + ' ' + seconds_str)
    d1 = time_s - (dayrange * (sphr * hrpd))
    d2 = time_s + (dayrange * (sphr * hrpd))

    logger.info('Getting relevant events from the gCMT catalog for the dates:'
                '%s - %s \n' % (util.tts(d1), util.tts(d2)))

    events = gcmt.get_events((d1, d2), magmin=min_magnitude)

    if len(events) < 1:
        logger.warn('Found no event information in the gCMT catalog.')
        event = None

    if len(events) > 1:
        logger.info(
            'More than one event from that date with specified magnitude'
            'found! Please copy the relevant event information to the'
            'configuration file file!')
        for event in events:
            print event

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
        [num.cos(source.dip * d2r) * num.cos(source.strike * d2r),
         -num.cos(source.dip * d2r) * num.sin(source.strike * d2r),
          num.sin(source.dip * d2r)])


def strike_vector(strike, order='ENZ'):
    if order == 'ENZ':
        return num.array(
            [num.sin(strike * d2r),
             num.cos(strike * d2r),
             0.])
    elif order == 'NEZ':
        return num.array(
            [num.cos(strike * d2r),
             num.sin(strike * d2r),
             0.])
    else:
        raise Exception('Order %s not implemented!' % order)


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

    return num.array([source.east_shift, source.north_shift, source.depth]) + \
        0.5 * source.width * RS_dipvector(source)


def adjust_fault_reference(source, input_depth='top'):
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

    if input_depth == 'top':
        center = RS_center(source)
    elif input_depth == 'center':
        center = num.array(
            [source.east_shift, source.north_shift, source.depth])
    else:
        raise Exception('input_depth %s not supported!' % input_depth)

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

    with open(outpath, 'w') as f:
        pickle.dump(outlist, f)


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
        objects = pickle.load(open(loadpath, 'rb'))
    except IOError:
        raise Exception(
            'File %s does not exist!' % loadpath)
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
        logger.debug('Cov_pv not positive definite!'
                    ' Finding nearest psd matrix...')
        cov = repair_covariance(cov)

    return cov


def near_psd(x, epsilon=num.finfo(num.float64).eps):
    """
    Calculates the nearest postive semi-definite matrix for a correlation/
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
    Numpy number precission not high enough to resolve this for low valued
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
    T = 1. / (num.multiply(vec, vec) * val.T)
    T = num.matrix(num.sqrt(num.diag(num.array(T).reshape((n)))))
    B = T * vec * num.diag(num.array(num.sqrt(val)).reshape((n)))
    near_corr = num.array(B * B.T)

    # returning the scaling factors
    return near_corr * a * b


def repair_covariance(x, epsilon=num.finfo(num.float64).eps):
    """
    Make covariance input matrix A positive definite.
    Setting eigenvalues that are lower than the precission of numpy floats to
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
    vec = num.matrix(eigvec)
    return vec * num.diag(val) * vec.T


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

    max_depth = crustal_model.max('z')

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
        of objects to be seperated
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
    fullc = i / cycle
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
        for v in d.tervalues():
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

    posterior_idxs = {
        'mean': mean_idx,
        'min': min_idx,
        'max': max_idx}

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

    if os.path.basename(problem.outfolder) == 'hypers':
        return True
    else:
        return False


def scalar2floatX(a, floatX=tconfig.floatX):
    if floatX == 'float32':
        return num.float32(a)
    elif floatX == 'float64':
        return num.float64(a)


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
    b = a[:, 1: -1]
    b[:, 3] /= 1000.
    return LayeredModel.from_scanlines(
        read_nd_model_str(
            re.sub('[\[\]]', '', num.array2string(
                b, precision=4,
                    formatter={'float_kind': lambda x: "%.3f" % x}))))


def list_to_str(l):
    """
    Transform entries of a list or 1-d array to one single string.
    """
    return ''.join('%f ' % entry for entry in l)


def swap_columns(array, index1, index2):
    """
    Swaps the column of the input array based on the given indexes.
    """
    array[:, index1], array[:, index2] = \
        array[:, index2], array[:, index1].copy()
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
        logger.warn('Lines are parallel! No intersection point!')
        return None

    tmp = num.sum(dep * dp, axis=1)
    return num.atleast_2d(tmp / denom).T * dn + n1


def get_rotation_matrix(axis):
    """
    Return a function for 3-d rotation matrix for a specified axis.

    Parameters
    ----------
    axis : str or list of str
        x, y or z for the axis

    Returns
    -------
    func that takes an angle
    """

    def rotx(angle):
        angle *= num.pi / 180.
        return num.array(
            [[1, 0, 0],
             [0, num.cos(angle), -num.sin(angle)],
             [0, num.sin(angle), -num.cos(angle)]])

    def roty(angle):
        angle *= num.pi / 180.
        return num.array(
            [[num.cos(angle), 0, num.sin(angle)],
             [0, 1, 0],
             [-num.sin(angle), 0, num.cos(angle)]])

    def rotz(angle):
        angle *= num.pi / 180.
        return num.array(
            [[num.cos(angle), -num.sin(angle), 0],
             [num.sin(angle), num.cos(angle), 0],
             [0, 0, 1]])

    R = {'x': rotx,
         'y': roty,
         'z': rotz}

    if isinstance(axis, list):
        return [R[a] for a in axis]
    elif isinstance(axis, str):
        return R[axis]
    else:
        raise Exception('axis has to be either string or list of strings!')
