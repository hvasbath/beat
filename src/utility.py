import logging
import os
import collections
import copy
import cPickle as pickle

from pyrocko import util, orthodrome, catalog
from pyrocko.cake import m2d

import numpy as num

from pyproj import Proj


logger = logging.getLogger('utility')

DataMap = collections.namedtuple('DataMap', 'list_ind, slc, shp, dtype')

kmtypes = set(['east_shift', 'north_shift', 'length', 'width', 'depth'])

seconds_str = '00:00:00'

sphr = 3600.
hrpd = 24.

km = 1000.


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


def weed_input_rvs(input_rvs, mode, dataset):
    """
    Throw out random variables (RV)s from input list that are not included by
    the respective synthetics generating functions.

    Parameters
    ----------
    input_rvs : list
        of :class:`pymc3.Distribution`
    mode : str
        'geometry', 'static, 'kinematic' determining the discarded RVs
    dataset : str
        'seismic' or 'geodetic' determining the discarded RVs

    Returns
    -------
    weeded_input_rvs : list
        of :class:`pymc3.Distribution`
    """

    name_order = [param.name for param in input_rvs]
    weeded_input_rvs = copy.copy(input_rvs)

    if mode == 'geometry':
        if dataset == 'geodetic':
            tobeweeded = ['time', 'duration']
        elif dataset == 'seismic':
            tobeweeded = ['opening']

    indexes = []
    for burian in tobeweeded:
        if burian in name_order:
            indexes.append(name_order.index(burian))

    indexes.sort(reverse=True)

    for ind in indexes:
        weeded_input_rvs.pop(ind)

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


def transform_sources(sources, datasets):
    """
    Transforms a list of :py:class:`heart.RectangularSource` to a dictionary of
    sources :py:class:`pscmp.PsCmpRectangularSource` for geodetic data and
    :py:class:`pyrocko.gf.seismosizer.RectangularSource` for seismic data.

    Parameters
    ----------
    sources : list
        :class:`heart.RectangularSource`
    datasets : list
        of strings with the datasets to be included 'geodetic' or 'seismic'

    Returns
    -------
    d : dict
        of transformed sources with datasets as keys
    """

    d = dict()

    for dataset in datasets:
        sub_sources = []

        for source in sources:
            sub_sources.append(source.patches(1, 1, dataset))

        # concatenate list of lists to single list
        transformed_sources = []
        map(transformed_sources.extend, sub_sources)

        d[dataset] = transformed_sources

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

    n_sources = point[point.keys()[0]].shape[0]

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
            'File %s does not exist! Data already imported?' % loadpath)
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
