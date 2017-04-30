"""
Core module with functions to calculate Greens Functions and synthetics.
Also contains main classes for setup specific parameters.
"""

import os
import logging
import shutil
import copy
from time import time

from beat import psgrn, pscmp, utility, qseis2d

from theano import config as tconfig
import numpy as num

from pyrocko.guts import Object, String, Float, Int, Tuple, List
from pyrocko.guts_array import Array

from pyrocko import crust2x2, gf, cake, orthodrome, trace, util
from pyrocko.cake import GradientLayer
from pyrocko.fomosto import qseis, qssp
from pyrocko.model import Station

from pyrocko.gf.seismosizer import outline_rect_source, Cloneable
from pyrocko.orthodrome import ne_to_latlon
#from pyrocko.fomosto import qseis2d


logger = logging.getLogger('heart')

c = 299792458.  # [m/s]
km = 1000.
d2r = num.pi / 180.
r2d = 180. / num.pi

lambda_sensors = {
    'Envisat': 0.056,       # needs updating- no ressource file
    'ERS1': 0.05656461471698113,
    'ERS2': 0.056,          # needs updating
    'JERS': 0.23513133960784313,
    'RadarSat2': 0.055465772433
    }


class RectangularSource(gf.DCSource, Cloneable):
    """
    Source for rectangular fault that unifies the necessary different source
    objects for teleseismic and geodetic computations.
    Reference point of the depth attribute is the top-center of the fault.

    Many of the methods of the RectangularSource have been modified from
    the HalfspaceTool from GertjanVanZwieten.
    """

    width = Float.T(help='width of the fault [m]',
                    default=1. * km)
    length = Float.T(help='length of the fault [m]',
                    default=1. * km)
    slip = Float.T(help='slip of the fault [m]',
                    default=1.)
    opening = Float.T(help='opening of the fault [m]',
                    default=0.)

    @property
    def dipvector(self):
        """
        Get 3 dimensional dip-vector of the planar fault.

        Parameters
        ----------
        dip : scalar, float
            dip-angle [deg] of the fault
        strike : scalar, float
            strike-abgle [deg] of the fault

        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.array(
            [num.cos(self.dip * d2r) * num.cos(self.strike * d2r),
             -num.cos(self.dip * d2r) * num.sin(self.strike * d2r),
              num.sin(self.dip * d2r)])

    @property
    def strikevector(self):
        """
        Get 3 dimensional strike-vector of the planar fault.

        Parameters
        ----------
        strike : scalar, float
            strike-abgle [deg] of the fault

        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.array(
            [num.sin(self.strike * d2r),
             num.cos(self.strike * d2r),
             0.])

    def center(self, width):
        """
        Get 3d fault center coordinates. Depth attribute is top depth!

        Parameters
        ----------
        width : scalar, float
            width [m] of the fault (dip-direction)

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the center of the
        fault
        """

        return num.array([self.east_shift, self.north_shift, self.depth]) + \
            0.5 * width * self.dipvector

    def center2top_depth(self, center):
        """
        Get top depth of the fault [m] given a potential center point.
        (Patches method needs input depth to
        be top_depth.)

        Parameters
        ----------
        center : scalar, float
            coordinates [m] of the center of the fault

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        upper edge of the fault
        """

        return num.array([center[0], center[1], center[2]]) - \
            0.5 * self.width * self.dipvector

    def bottom_depth(self, depth):
        """
        Get bottom depth of the fault [m].
        (Patches method needs input depth to be top_depth.)

        Parameters
        ----------
        depth : scalar, float
            depth [m] of the center of the fault

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        lower edge of the fault
        """

        return num.array([self.east_shift, self.north_shift, depth]) + \
            0.5 * self.width * self.dipvector

    def trace_center(self, depth):
        """
        Get trace central coordinates of the fault [m] at the surface of the
        halfspace.

        Parameters
        ----------
        depth : scalar, float
            depth [m] of the center of the fault

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        lower edge of the fault
        """

        bd = self.bottom_depth(depth)
        xtrace = bd[0] - \
            (bd[2] * num.cos(d2r * self.strike) / num.tan(d2r * self.dip))
        ytrace = bd[1] + \
            (bd[2] * num.sin(d2r * self.strike) / num.tan(d2r * self.dip))
        return num.array([xtrace, ytrace, 0.])

    def patches(self, nl, nw, dataset):
        """
        Cut source into n by m sub-faults and return n times m
        :class:`RectangularSource` Objects.
        Discretization starts at shallow depth going row-wise deeper.
        REQUIRES: self.depth to be TOP DEPTH!!!

        Parameters
        ----------
        nl : int
            number of patches in length direction (strike)
        nw : int
            number of patches in width direction (dip)
        dataset : string
            'geodetic' or 'seismic' determines the source to be returned

        Returns
        -------
        :class:`pscmp.PsCmpRectangularSource` or
        :class:`pyrocko.gf.seismosizer.RectangularSource` depending on
        datatype. Depth is being updated from top_depth to center_depth.
        """

        length = self.length / float(nl)
        width = self.width / float(nw)

        patches = []
        for j in range(nw):
            for i in range(nl):
                sub_center = self.center(self.width) + \
                    self.strikevector * ((i + 0.5 - 0.5 * nl) * length) + \
                    self.dipvector * ((j + 0.5 - 0.5 * nw) * width)

                if dataset == 'seismic':
                    patch = gf.RectangularSource(
                        lat=float(self.lat),
                        lon=float(self.lon),
                        east_shift=float(sub_center[0]),
                        north_shift=float(sub_center[1]),
                        depth=float(sub_center[2]),
                        strike=self.strike, dip=self.dip, rake=self.rake,
                        length=length, width=width, stf=self.stf,
                        time=self.time, slip=self.slip)
                elif dataset == 'geodetic':
                    patch = pscmp.PsCmpRectangularSource(
                        lat=self.lat,
                        lon=self.lon,
                        east_shift=float(sub_center[0]),
                        north_shift=float(sub_center[1]),
                        depth=float(sub_center[2]),
                        strike=self.strike, dip=self.dip, rake=self.rake,
                        length=length, width=width, slip=self.slip,
                        opening=self.opening)
                else:
                    raise Exception(
                        "Datatype not supported either: 'seismic/geodetic'")

                patches.append(patch)

        return patches

    def outline(self, cs='xyz'):
        points = outline_rect_source(self.strike, self.dip, self.length,
                                     self.width)
        center = self.center(self.width)
        points[:, 0] += center[0]
        points[:, 1] += center[1]
        points[:, 2] += center[2]
        if cs == 'xyz':
            return points
        elif cs == 'xy':
            return points[:, :2]
        elif cs in ('latlon', 'lonlat'):
            latlon = ne_to_latlon(
                self.lat, self.lon, points[:, 0], points[:, 1])

            latlon = num.array(latlon).T
            if cs == 'latlon':
                return latlon
            else:
                return latlon[:, ::-1]

    def extent_source(self, extension_width, extension_length,
                     patch_width, patch_length):
        """
        Extend fault into all directions. Rounds dimensions to have no
        half-patches.

        Parameters
        ----------
        extension_width : float
            factor to extend source in width (dip-direction)
        extension_length : float
            factor extend source in length (strike-direction)
        patch_width : float
            Width [m] of subpatch in dip-direction
        patch_length : float
            Length [m] of subpatch in strike-direction

        Returns
        -------
        dict with list of :class:`pscmp.PsCmpRectangularSource` or
            :class:`pyrocko.gf.seismosizer.RectangularSource`
        """
        s = copy.deepcopy(self)

        l = self.length
        w = self.width

        new_length = num.ceil((l + (2. * l * extension_length)) / km) * km
        new_width = num.ceil((w + (2. * w * extension_length)) / km) * km

        npl = int(num.ceil(new_length / patch_length))
        npw = int(num.ceil(new_width / patch_width))

        new_length = float(npl * patch_length)
        new_width = float(npw * patch_width)
        logger.info(
            'Fault extended to length=%f, width=%f!' % (new_length, new_width))

        orig_center = s.center(s.width)
        s.update(length=new_length, width=new_width)

        top_center = s.center2top_depth(orig_center)

        if top_center[2] < 0.:
            logger.info('Fault would intersect surface!'
                        ' Setting top center to 0.!')
            trace_center = s.trace_center(s.depth)
            s.update(east_shift=float(trace_center[0]),
                     north_shift=float(trace_center[1]),
                     depth=float(trace_center[2]))
        else:
            s.update(east_shift=float(top_center[0]),
                     north_shift=float(top_center[1]),
                     depth=float(top_center[2]))

        return s


def log_determinant(A, inverse=False):
    """
    Calculates the natural logarithm of a determinant of the given matrix '
    according to the properties of a triangular matrix.

    Parameters
    ----------
    A : n x n :class:`numpy.ndarray`
    inverse : boolean
        If true calculates the log determinant of the inverse of the colesky
        decomposition, which is equvalent to taking the determinant of the
        inverse of the matrix.

        L.T* L = R           inverse=False
        L-1*(L-1)T = R-1     inverse=True

    Returns
    -------
    float logarithm of the determinant of the input Matrix A
    """

    cholesky = num.linalg.cholesky(A)
    if inverse:
        cholesky = num.linalg.inv(cholesky)
    return num.log(num.diag(cholesky)).sum()


class ReferenceLocation(gf.Location):
    """
    Reference Location for Green's Function store calculations!
    """
    station = String.T(
        default='Store_Name',
        help='This mimics the station.station attribute which determines the'
             ' store name!')


class Covariance(Object):
    """
    Covariance of an observation. Holds data and model prediction uncertainties
    for one observation object.
    """

    data = Array.T(shape=(None, None),
                    dtype=tconfig.floatX,
                    help='Data covariance matrix',
                    optional=True)
    pred_g = Array.T(shape=(None, None),
                    dtype=tconfig.floatX,
                    help='Model prediction covariance matrix, fault geometry',
                    optional=True)
    pred_v = Array.T(shape=(None, None),
                    dtype=tconfig.floatX,
                    help='Model prediction covariance matrix, velocity model',
                    optional=True)

    @property
    def p_total(self):
        if self.pred_g is None:
            self.pred_g = num.zeros_like(self.data, dtype=tconfig.floatX)

        if self.pred_v is None:
            self.pred_v = num.zeros_like(self.data, dtype=tconfig.floatX)

        return self.pred_g + self.pred_v

    @property
    def inverse(self):
        """
        Add and invert ALL uncertainty covariance Matrices.
        """
        Cx = self.p_total + self.data
        if Cx.sum() == 0:
            logger.debug('No covariances given, using I matrix!')
            return num.eye(Cx.shape[0]).astype(tconfig.floatX)
        else:
            return num.linalg.inv(Cx).astype(tconfig.floatX)

    @property
    def inverse_p(self):
        """
        Add and invert different MODEL uncertainty covariance Matrices.
        """
        if self.p_total.sum() == 0:
            raise Exception('No model covariance defined!')
        return num.linalg.inv(self.p_total).astype(tconfig.floatX)

    @property
    def inverse_d(self):
        """
        Invert DATA covariance Matrix.
        """
        if self.data is None:
            raise Exception('No data covariance matrix defined!')
        return num.linalg.inv(self.data).astype(tconfig.floatX)

    @property
    def log_norm_factor(self):
        """
        Calculate the normalisation factor of the posterior pdf.
        Following Duputel et al. 2014
        """

        N = self.data.shape[0]

        if self.p_total.any():
            ldet_x = log_determinant(self.data + self.p_total)
        elif self.data.any():
            ldet_x = log_determinant(self.data)
        else:
            logger.debug('No covariance defined, using I matrix!')
            ldet_x = 1.

        return utility.scalar2floatX((N * num.log(2 * num.pi)) + ldet_x)


class TeleseismicTarget(gf.Target):
    """
    Extension to :class:`pyrocko.gf.seismpsizer.Target` to have
    :class:`Covariance` as an attribute.
    """

    covariance = Covariance.T(
        default=Covariance.D(),
        optional=True,
        help=':py:class:`Covariance` that holds data'
             'and model prediction covariance matrixes')

    @property
    def typ(self):
        return self.codes[3]

    @property
    def samples(self):
        return self.covariance.data.shape[0]

    def update_target_times(self, source=None, taperer=None):
        """
        Update the target attributes tmin and tmax to do the stacking
        only in this interval. Adds twice taper fade in time to each taper
        side.

        Parameters
        ----------
        source : list
            containing :class:`pyrocko.gf.seismosizer.Target` Objects
        taperer : :class:`pyrocko.trace.CosTaper`
        """
        if source is None or taperer is None:
            self.tmin = None
            self.tmax = None
        else:
            tolerance = 4 * (taperer.b - taperer.a)
            self.tmin = taperer.a - tolerance - source.time
            self.tmax = taperer.d + tolerance - source.time


class ArrivalTaper(trace.Taper):
    """
    Cosine arrival Taper.
    """

    a = Float.T(default=-15.,
                help='start of fading in; [s] w.r.t. phase arrival')
    b = Float.T(default=-10.,
                help='end of fading in; [s] w.r.t. phase arrival')
    c = Float.T(default=50.,
                help='start of fading out; [s] w.r.t. phase arrival')
    d = Float.T(default=55.,
                help='end of fading out; [s] w.r.t phase arrival')

    @property
    def duration(self):
        return num.abs(self.a) + self.d

    @property
    def fade(self):
        return num.abs(self.a - self.b)


class Trace(Object):
    pass


class Filter(Object):
    """
    Filter object defining frequency range of traces after filtering
    """

    lower_corner = Float.T(
        default=0.001,
        help='Lower corner frequency')
    upper_corner = Float.T(
        default=0.1,
        help='Upper corner frequency')
    order = Int.T(
        default=4,
        help='order of filter, the higher the steeper')


class SeismicResult(Object):
    """
    Result object assembling different traces of misfit.
    """
    processed_obs = Trace.T(optional=True)
    filtered_obs = Trace.T(optional=True)
    processed_syn = Trace.T(optional=True)
    filtered_syn = Trace.T(optional=True)
    processed_res = Trace.T(optional=True)
    arrival_taper = trace.Taper.T(optional=True)
    llk = Float.T(default=0., optional=True)
    taper = trace.Taper.T(optional=True)


physical_bounds = dict(
    east_shift=(-500., 500.),
    north_shift=(-500., 500.),
    depth=(0., 1000.),
    strike=(0, 360.),
    dip=(0., 90.),
    rake=(-180., 180.),
    length=(0., 7000.),
    width=(0., 500.),
    slip=(0., 150.),
    magnitude=(-10., 10.),
    time=(-300., 300.),
    duration=(0., 600.),
    Uparr=(-0.3, 150.),
    Uperp=(-150., 150.),
    nuc_x=(0., num.inf),
    nuc_y=(0., num.inf),
    velocity=(0.5, 7.0),
    azimuth=(0, 360),
    amplitude=(0., 0.2),
    locking_depth=(0.1, 100.),
    seis_Z=(-20., 20.),
    seis_T=(-20., 20.),
    geo_S=(-20., 20.),
    geo_G=(-20., 20.),
    ramp=(-0.005, 0.005))


class Parameter(Object):
    """
    Optimization parameter determines the bounds of the search space.
    """

    name = String.T(default='depth')
    form = String.T(default='Uniform',
                    help='Type of prior distribution to use. Options:'
                         ' "Uniform", ...')
    lower = Array.T(shape=(None,),
                    dtype=tconfig.floatX,
                    serialize_as='list',
                    default=num.array([0., 0.], dtype=tconfig.floatX))
    upper = Array.T(shape=(None,),
                    dtype=tconfig.floatX,
                    serialize_as='list',
                    default=num.array([1., 1.], dtype=tconfig.floatX))
    testvalue = Array.T(shape=(None,),
                        dtype=tconfig.floatX,
                        serialize_as='list',
                        default=num.array([0.5, 0.5], dtype=tconfig.floatX))

    def validate_bounds(self):
        if self.name not in physical_bounds.keys():
            raise TypeError('The parameter "%s" cannot'
                ' be optimized for!' % self.name)

        if self.lower is not None:
            for i in range(self.dimension):
                if self.upper[i] < self.lower[i]:
                    raise ValueError('The upper parameter bound for'
                        ' parameter "%s" must be higher than the lower'
                        ' bound' % self.name)

                if self.testvalue[i] > self.upper[i] or \
                    self.testvalue[i] < self.lower[i]:
                    raise ValueError('The testvalue of parameter "%s" has to'
                        ' be within the upper and lower bounds' % self.name)

                phys_b = physical_bounds[self.name]

                if self.upper[i] > phys_b[1] or \
                    self.lower[i] < phys_b[0]:
                    raise ValueError(
                        'The parameter bounds (%f, %f) for "%s" are outside of'
                        ' physically meaningful values (%f, %f)!' % (
                            self.lower[i], self.upper[i], self.name,
                            phys_b[0], phys_b[1]))
        else:
            raise ValueError(
                'Parameter bounds for "%s" have to be defined!' % self.name)

    def random(self):
        """
        Create random samples within the parameter bounds.

        Returns
        -------
        :class:`numpy.ndarray` of size (n, m)
        """
        return (self.upper - self.lower) * num.random.rand(
            self.dimension) + self.lower

    @property
    def dimension(self):
        return self.lower.size

    def bound_to_array(self):
        return num.array([self.lower, self.testval, self.upper],
                         dtype=num.float)


class GeodeticTarget(gf.meta.MultiLocation):
    """
    Overall geodetic data set class
    """

    typ = String.T(
        default='SAR',
        help='Type of geodetic data, e.g. SAR, GPS, ...')
    name = String.T(
        default='A',
        help='e.g. GPS station name or InSAR satellite track ')

    def update_local_coords(self, loc):
        """
        Calculate local coordinates with respect to given Location.

        Parameters
        ----------
        loc : :class:`pyrocko.gf.meta.Location`

        Returns
        -------
        :class:`numpy.ndarray` (n_points, 3)
        """

        self.north_shifts, self.east_shifts = orthodrome.latlon_to_ne_numpy(
            loc.lat, loc.lon, self.lats, self.lons)
        return self.north_shifts, self.east_shifts

    @property
    def samples(self):
        if self.lats is not None:
            n = self.lats.size
        elif self.utmn is not None:
            n = self.utmn.size
        elif self.north_shifts is not None:
            n = self.north_shifts.size
        else:
            raise Exception('No coordinates defined!')
        return n


class GPSComponent(Object):
    """
    Object holding the GPS data for a single station.
    """
    name = String.T(default='E', help='direction of measurement, E/N/U')
    v = Float.T(default=0.1, help='Average velocity in [m/yr]')
    sigma = Float.T(default=0.01, help='sigma measurement error (std) [m/yr]')
    unit = String.T(default='m/yr', help='Unit of velocity v')


class GPSStation(Station):
    """
    GPS station object, holds the displacment components and has all pyrocko
    station functionality.
    """

    components = List.T(GPSComponent.T())

    def set_components(self, components):
        self.components = []
        for c in components:
            self.add_component(c)

    def get_components(self):
        return list(self.components)

    def get_component_names(self):
        return set(c.name for c in self.components)

    def remove_component_by_name(self, name):
        todel = [c for c in self.components if c.name == name]
        for c in todel:
            self.components.remove(c)

    def add_component(self, component):
        self.remove_component_by_name(component.name)
        self.components.append(component)
        self.components.sort(key=lambda c: c.name)

    def get_component(self, name):
        for c in self.components:
            if c.name == name:
                return c


class GPSCompoundComponent(GeodeticTarget):
    """
    Collecting many GPS components and merging them into arrays.
    Make synthetics generation more efficient.
    """
    los_vector = Array.T(shape=(None, 3), dtype=num.float, optional=True)
    displacement = Array.T(shape=(None,), dtype=num.float, optional=True)
    name = String.T(default='E', help='direction of measurement, E/N/U')
    station_names = List.T(String.T(optional=True))
    covariance = Covariance.T(
        optional=True,
        help=':py:class:`Covariance` that holds data'
             'and model prediction covariance matrixes')
    odw = Array.T(
        shape=(None,),
        dtype=num.float,
        help='Overlapping data weights, additional weight factor to the'
             'dataset for overlaps with other datasets',
        optional=True)

    def update_los_vector(self):
        if self.name == 'E':
            c = num.array([0, 1, 0])
        elif self.name == 'N':
            c = num.array([1, 0, 0])
        elif self.name == 'U':
            c = num.array([0, 0, 1])
        else:
            raise Exception('Component %s not supported' % self.component)

        self.los_vector = num.tile(c, self.samples).reshape(self.samples, 3)
        return self.los_vector

    def __str__(self):
        s = 'GPS\n compound: \n'
        s += '  component: %s\n' % self.name
        if self.lats is not None:
            s += '  number of stations: %i\n' % self.samples
        return s


class GPSDataset(object):
    """
    Collecting many GPS stations into one object. Easy managing and assessing
    single stations and also merging all the stations components into compound
    components for fast and easy modeling.
    """

    def __init__(self, name=None, stations=None):
        self.stations = {}
        self.name = name

        if stations is not None:
            for station in stations:
                self.stations[station.name] = station

    def add_station(self, station, force=False):
        if not isinstance(station, GPSStation):
            raise Exception(
                'Input object is not a valid station of'
                ' class: %s' % GPSStation)

        if station.name not in self.stations.keys() or force:
            self.stations[station.name] = station
        else:
            raise Exception(
                'Station %s already exists in dataset!' % station.name)

    def get_station(self, name):
        return self.stations[name]

    def remove_stations(self, stations):
        for st in stations:
            self.stations.pop(st)

    def get_station_names(self):
        return list(self.stations.keys())

    def get_component_names(self):
        return self.stations.values()[0].get_component_names()

    def get_compound(self, name):
        stations = self.stations.values()

        comps = self.get_component_names()

        if name in comps:
            stations_comps = [st.get_component(name) for st in stations]
            lats = num.array([st.lat for st in stations])
            lons = num.array([st.lon for st in stations])

            vs = num.array([c.v for c in stations_comps])
            variances = num.power(
                num.array([c.sigma for c in stations_comps]), 2)
        else:
            raise Exception(
                'Requested component %s does not exist in the dataset' % name)

        return GPSCompoundComponent(
            typ='GPS',
            station_names=self.get_station_names(),
            displacement=vs,
            covariance=Covariance(data=num.eye(lats.size) * variances),
            lats=lats,
            lons=lons,
            east_shifts=num.zeros_like(lats),
            north_shifts=num.zeros_like(lats),
            name=name,
            odw=num.ones_like(lats.size))

    def iter_stations(self):
        return self.stations.iteritems()


class IFG(GeodeticTarget):
    """
    Interferogram class as a dataset in the optimization.
    """

    master = String.T(optional=True,
                      help='Acquisition time of master image YYYY-MM-DD')
    slave = String.T(optional=True,
                      help='Acquisition time of slave image YYYY-MM-DD')
    amplitude = Array.T(shape=(None,), dtype=num.float, optional=True)
    wrapped_phase = Array.T(shape=(None,), dtype=num.float, optional=True)
    incidence = Array.T(shape=(None,), dtype=num.float, optional=True)
    heading = Array.T(shape=(None,), dtype=num.float, optional=True)
    los_vector = Array.T(shape=(None, 3), dtype=num.float, optional=True)
    utmn = Array.T(shape=(None,), dtype=num.float, optional=True)
    utme = Array.T(shape=(None,), dtype=num.float, optional=True)
    satellite = String.T(default='Envisat')

    def __str__(self):
        s = 'IFG\n Acquisition Track: %s\n' % self.name
        s += '  timerange: %s - %s\n' % (self.master, self.slave)
        if self.lats is not None:
            s += '  number of pixels: %i\n' % self.samples
        return s

    @property
    def wavelength(self):
        return lambda_sensors[self.satellite]

    def update_los_vector(self):
        """
        Calculate LOS vector for given attributes incidence and heading angles.

        Returns
        -------
        :class:`numpy.ndarray` (n_points, 3)
        """

        if self.incidence.all() and self.heading.all() is None:
            raise Exception('Incidence and Heading need to be provided!')

        Su = num.cos(num.deg2rad(self.incidence))
        Sn = - num.sin(num.deg2rad(self.incidence)) * \
             num.cos(num.deg2rad(self.heading - 270))
        Se = - num.sin(num.deg2rad(self.incidence)) * \
             num.sin(num.deg2rad(self.heading - 270))
        self.los_vector = num.array([Sn, Se, Su], dtype=num.float).T
        return self.los_vector


class DiffIFG(IFG):
    """
    Differential Interferogram class as geodetic target for the calculation
    of synthetics and container for SAR data.
    """

    unwrapped_phase = Array.T(shape=(None,), dtype=num.float, optional=True)
    coherence = Array.T(shape=(None,), dtype=num.float, optional=True)
    reference_point = Tuple.T(2, Float.T(), optional=True)
    reference_value = Float.T(optional=True, default=0.0)
    displacement = Array.T(shape=(None,), dtype=num.float, optional=True)
    covariance = Covariance.T(
        optional=True,
        help=':py:class:`Covariance` that holds data'
             'and model prediction covariance matrixes')
    odw = Array.T(
        shape=(None,),
        dtype=num.float,
        help='Overlapping data weights, additional weight factor to the'
             'dataset for overlaps with other datasets',
        optional=True)


class GeodeticResult(Object):
    """
    Result object assembling different geodetic data.
    """
    processed_obs = GeodeticTarget.T(optional=True)
    processed_syn = GeodeticTarget.T(optional=True)
    processed_res = GeodeticTarget.T(optional=True)
    llk = Float.T(default=0., optional=True)


def init_targets(stations, earth_model='ak135-f-average.m',
                 channels=['T', 'Z'], sample_rate=1.0,
                 crust_inds=[0], interpolation='multilinear',
                 reference_location=None):
    """
    Initiate a list of target objects given a list of indexes to the
    respective GF store velocity model variation index (crust_inds).

    Parameters
    ----------
    stations : List of :class:`pyrocko.model.Station`
        List of station objects for which the targets are being initialised
    earth_model = str
        Name of the earth model that has been used for GF calculation.
    channels : List of str
        Components of the traces to be optimized for if rotated:
        T - transversal, Z - vertical, R - radial
        If not rotated:
        E - East, N- North, U - Up (Vertical)
    sample_rate : scalar, float
        sample rate [Hz] of the Greens Functions to use
    crust_inds : List of int
        Indexes of different velocity model realisations, 0 - reference model
    interpolation : str
        Method of interpolation for the Greens Functions, can be 'multilinear'
        or 'nearest_neighbor'
    reference_location : :class:`ReferenceLocation` or
        :class:`pyrocko.model.Station`
        if given, targets are initialised with this reference location

    Returns
    -------
    List of :class:`pyrocko.gf.seismosizer.Target`
    """

    if reference_location is None:
        store_prefixes = [copy.deepcopy(station.station) \
             for station in stations]
    else:
        store_prefixes = [copy.deepcopy(reference_location.station) \
             for station in stations]

    em_name = earth_model.split('-')[0].split('.')[0]

    targets = [TeleseismicTarget(
        quantity='displacement',
        codes=(stations[sta_num].network,
                 stations[sta_num].station,
                 '%i' % crust_ind, channel),  # n, s, l, c
        lat=stations[sta_num].lat,
        lon=stations[sta_num].lon,
        azimuth=stations[sta_num].get_channel(channel).azimuth,
        dip=stations[sta_num].get_channel(channel).dip,
        interpolation=interpolation,
        store_id='%s_%s_%.3fHz_%s' % (
            store_prefixes[sta_num], em_name, sample_rate, crust_ind))

        for channel in channels
            for crust_ind in crust_inds
                for sta_num in range(len(stations))]

    return targets


def vary_model(earthmod, error_depth=0.1, error_velocities=0.1,
        depth_limit_variation=600 * km):
    """
    Vary depths and velocities in the given source model by Gaussians with
    given 2-sigma errors [percent]. Ensures increasing velocity with depth.
    Stops variating the input model at the given depth_limit_variation [m].
    Mantle discontinuity uncertainties are hardcoded based on
    Mooney et al. 1981 and Woodward et al.1991

    Parameters
    ----------
    earthmod : :class:`pyrocko.cake.LayeredModel`
        Earthmodel defining layers, depth, velocities, densities
    error_depth : scalar, float
        2 sigma error in percent of the depth for the respective layers
    error_velocities : scalar, float
        2 sigma error in percent of the velocities for the respective layers
    depth_limit_variations : scalar, float
        depth threshold [m], layers with depth > than this are not varied

    Returns
    -------
    Varied Earthmodel : :class:`pyrocko.cake.LayeredModel`
    Cost : int
        Counts repetitions of cycles to ensure increasing layer velocity,
        unlikely velocities have high Cost
        Cost of up to 20 are ok for crustal profiles.
    """

    new_earthmod = copy.deepcopy(earthmod)
    layers = new_earthmod.layers()

    last_l = None
    cost = 0
    deltaz = 0

    # uncertainties in discontinuity depth after Shearer 1991
    discont_unc = {
        '410': 3 * km,
        '520': 4 * km,
        '660': 8 * km}

    # uncertainties in velocity for upper and lower mantle from Woodward 1991
    # and Mooney 1989
    mantle_vel_unc = {
        '100': 0.05,     # above 100
        '200': 0.03,     # above 200
        '400': 0.01}     # above 400

    for layer in layers:
        # stop if depth_limit_variation is reached
        if depth_limit_variation:
            if layer.ztop >= depth_limit_variation:
                layer.ztop = last_l.zbot
                # assign large cost if previous layer has higher velocity
                if layer.mtop.vp < last_l.mtop.vp or \
                   layer.mtop.vp > layer.mbot.vp:
                    cost = 1000
                # assign large cost if layer bottom depth smaller than top
                if layer.zbot < layer.ztop:
                    cost = 1000
                break
        repeat = 1
        count = 0
        while repeat:
            if count > 1000:
                break

            # vary layer velocity
            # check for layer depth and use hardcoded uncertainties
            for l_depth, vel_unc in mantle_vel_unc.items():
                if float(l_depth) * km < layer.ztop:
                    error_velocities = vel_unc
                    logger.debug('Velocity error: %f ', error_velocities)

            deltavp = float(num.random.normal(
                        0, layer.mtop.vp * error_velocities / 3., 1))

            if layer.ztop == 0:
                layer.mtop.vp += deltavp
                layer.mbot.vs += (deltavp / layer.mbot.vp_vs_ratio())

            # ensure increasing velocity with depth
            if last_l:
                # gradient layer without interface
                if layer.mtop.vp == last_l.mbot.vp:
                    if layer.mbot.vp + deltavp < layer.mtop.vp:
                        count += 1
                    else:
                        layer.mbot.vp += deltavp
                        layer.mbot.vs += (deltavp /
                                                layer.mbot.vp_vs_ratio())
                        repeat = 0
                        cost += count
                elif layer.mtop.vp + deltavp < last_l.mbot.vp:
                    count += 1
                else:
                    layer.mtop.vp += deltavp
                    layer.mtop.vs += (deltavp / layer.mtop.vp_vs_ratio())

                    if isinstance(layer, GradientLayer):
                        layer.mbot.vp += deltavp
                        layer.mbot.vs += (deltavp / layer.mbot.vp_vs_ratio())
                    repeat = 0
                    cost += count
            else:
                repeat = 0

        # vary layer depth
        layer.ztop += deltaz
        repeat = 1

        # use hard coded uncertainties for mantle discontinuities
        if '%i' % (layer.zbot / km) in discont_unc:
            factor_d = discont_unc['%i' % (layer.zbot / km)] / layer.zbot
        else:
            factor_d = error_depth

        while repeat:
            # ensure that bottom of layer is not shallower than the top
            deltaz = float(num.random.normal(
                       0, layer.zbot * factor_d / 3., 1))  # 3 sigma
            layer.zbot += deltaz
            if layer.zbot < layer.ztop:
                layer.zbot -= deltaz
                count += 1
            else:
                repeat = 0
                cost += count

        last_l = copy.deepcopy(layer)

    return new_earthmod, cost


def ensemble_earthmodel(ref_earthmod, num_vary=10, error_depth=0.1,
                        error_velocities=0.1, depth_limit_variation=600 * km):
    """
    Create ensemble of earthmodels that vary around a given input earth model
    by a Gaussian of 2 sigma (in Percent 0.1 = 10%) for the depth layers
    and for the p and s wave velocities. Vp / Vs is kept unchanged

    Parameters
    ----------
    ref_earthmod : :class:`pyrocko.cake.LayeredModel`
        Reference earthmodel defining layers, depth, velocities, densities
    num_vary : scalar, int
        Number of variation realisations
    error_depth : scalar, float
        3 sigma error in percent of the depth for the respective layers
    error_velocities : scalar, float
        3 sigma error in percent of the velocities for the respective layers
    depth_limit_variation : scalar, float
        depth threshold [m], layers with depth > than this are not varied

    Returns
    -------
    List of Varied Earthmodels :class:`pyrocko.cake.LayeredModel`
    """

    earthmods = []
    i = 0
    while i < num_vary:
        new_model, cost = vary_model(
            ref_earthmod,
            error_depth,
            error_velocities,
            depth_limit_variation)

        if cost > 20:
            logger.debug('Skipped unlikely model %f' % cost)
        else:
            i += 1
            earthmods.append(new_model)

    return earthmods


def get_velocity_model(
    location, earth_model_name, crust_ind=0, gf_config=None,
    custom_velocity_model=None):
    """
    Get velocity model at the specified location, combines given or crustal
    models with the global model.

    Parameters
    ----------
    location : :class:`pyrocko.meta.Location`
    earth_model_name : str
        Name of the base earth model to be used, check
        :func:`pyrocko.cake.builtin_models` for alternatives,
        default ak135 with medium resolution
    crust_ind : int
        Index to set to the Greens Function store, 0 is reference store
        indexes > 0 use reference model and vary its parameters by a Gaussian
    gf_config : :class:`beat.config.GFConfig`
    custom_velocity_model : :class:`pyrocko.cake.LayeredModel`

    Returns
    -------
    :class:`pyrocko.cake.LayeredModel`
    """
    gfc = gf_config

    if custom_velocity_model is not None:
        logger.info('Using custom model from config file')
        global_model = cake.load_model(earth_model_name)
        source_model = utility.join_models(
            global_model, custom_velocity_model)

    elif gfc.use_crust2:
        # load velocity profile from CRUST2x2 and check for water layer
        profile = crust2x2.get_profile(location.lat, location.lon)

        if gfc.replace_water:
            thickness_lwater = profile.get_layer(crust2x2.LWATER)[0]
            if thickness_lwater > 0.0:
                logger.info('Water layer %f in CRUST model!'
                    ' Remove and add to lower crust' % thickness_lwater)
                thickness_llowercrust = profile.get_layer(
                    crust2x2.LLOWERCRUST)[0]
                thickness_lsoftsed = profile.get_layer(
                    crust2x2.LSOFTSED)[0]

                profile.set_layer_thickness(crust2x2.LWATER, 0.0)
                profile.set_layer_thickness(crust2x2.LSOFTSED,
                        num.ceil(thickness_lsoftsed / 3))
                profile.set_layer_thickness(
                    crust2x2.LLOWERCRUST,
                    thickness_llowercrust + \
                    thickness_lwater + \
                    (thickness_lsoftsed - num.ceil(thickness_lsoftsed / 3))
                                            )
                profile._elevation = 0.0
                logger.info('New Lower crust layer thickness %f' % \
                    profile.get_layer(crust2x2.LLOWERCRUST)[0])
        source_model = cake.load_model(
            earth_model_name, crust2_profile=profile)

    else:
        source_model = cake.load_model(earth_model_name)

    if crust_ind > 0:
        source_model = ensemble_earthmodel(
            source_model,
            num_vary=1,
            error_depth=gfc.error_depth,
            error_velocities=gfc.error_velocities,
            depth_limit_variation=gfc.depth_limit_variation * km)[0]

    return source_model


def get_slowness_taper(fomosto_config, velocity_model, distances):
    """
    Calculate slowness taper for backends that determine wavefield based
    on the velociy model.

    Parameters
    ----------
    fomosto_config : :class:`pyrocko.meta.Config`
    velocity_model : :class:`pyrocko.cake.LayeredModel`
    distances : tuple
        minimum and maximum distance [deg]

    Returns
    -------
    tuple of slownesses
    """

    fc = fomosto_config

    phases = [fc.tabulated_phases[i].phases
        for i in range(len(fc.tabulated_phases))]

    all_phases = []
    map(all_phases.extend, phases)

    mean_source_depth = num.mean(
        (fc.source_depth_min, fc.source_depth_max)) / km

    dists = num.linspace(distances[0], distances[1], 100)

    arrivals = velocity_model.arrivals(
        phases=all_phases,
        distances=dists,
        zstart=mean_source_depth)

    ps = num.array([arrivals[i].p for i in range(len(arrivals))])

    slownesses = ps / (cake.r2d * cake.d2m / km)
    smax = slownesses.max()

    return (0.0, 0.0, 1.1 * float(smax), 1.3 * float(smax))


def get_fomosto_baseconfig(
    seismic_gfconfig, event, station, channels, crust_ind):
    """
    Initialise fomosto config.
    """
    sf = seismic_gfconfig

    # define phases
    tabulated_phases = []
    if 'Z' in channels:
        tabulated_phases.append(gf.TPDef(
            id='any_P',
            definition='p,P,p\\,P\\'))
    if 'T' in channels:
        tabulated_phases.append(gf.TPDef(
            id='any_S',
            definition='s,S,s\\,S\\'))

    # calculate event-station distance [m]
    distance = orthodrome.distance_accurate50m(event, station)
    distance_min = distance - (sf.source_distance_radius * km)

    if distance_min < 0.:
        logger.warn('Minimum grid distance is below zero. Setting it to zero!')
        distance_min = 0.

    return gf.ConfigTypeA(
        id='%s_%s_%.3fHz_%s' % (
            station.station,
            sf.earth_model_name.split('-')[0].split('.')[0],
            sf.sample_rate,
            crust_ind),
        ncomponents=10,
        sample_rate=sf.sample_rate,
        receiver_depth=0. * km,
        source_depth_min=sf.source_depth_min * km,
        source_depth_max=sf.source_depth_max * km,
        source_depth_delta=sf.source_depth_spacing * km,
        distance_min=distance_min,
        distance_max=distance + (sf.source_distance_radius * km),
        distance_delta=sf.source_distance_spacing * km,
        tabulated_phases=tabulated_phases)


backend_builders = {
    'qseis': qseis.build,
    'qssp': qssp.build,
    'qseis2d': qseis2d.build
                 }


def choose_backend(
    fomosto_config, code, source_model, receiver_model, distances,
    gf_directory='qseis2d_green'):
    """
    Get backend related config.
    """

    fc = fomosto_config
    receiver_basement_depth = 150 * km

    if code == 'qseis':
        # find common basement layer
        l = source_model.layer(receiver_basement_depth)
        receiver_model = receiver_model.extract(
            depth_max=l.ztop)
        receiver_model.append(l)

        version = '2006a'
        distances = num.array([fc.distance_min, fc.distance_max]) * cake.m2d
        slowness_taper = get_slowness_taper(fc, source_model, distances)

        conf = qseis.QSeisConfig(
            filter_shallow_paths=0,
            slowness_window=slowness_taper,
            wavelet_duration_samples=0.001,
            sw_flat_earth_transform=1,
            sw_algorithm=1,
            qseis_version=version)

    elif code == 'qssp':
        source_model = copy.deepcopy(receiver_model)
        receiver_model = None
        version = '2010'
        distances = num.array([fc.distance_min, fc.distance_max]) * cake.m2d
        slowness_taper = get_slowness_taper(fc, source_model, distances)

        conf = qssp.QSSPConfig(
            qssp_version=version,
            slowness_max=float(num.max(slowness_taper)),
            toroidal_modes=True,
            spheroidal_modes=True,
            source_patch_radius=(fc.distance_delta - \
                                 fc.distance_delta * 0.05) / km)

    elif code == 'qseis2d':
        version = '2014'
        slowness_taper = get_slowness_taper(fc, source_model, distances)

        conf = qseis2d.QSeis2dConfig()
        conf.qseis_s_config.slowness_window = slowness_taper
        conf.qseis_s_config.calc_slowness_window = 0
        conf.qseis_s_config.receiver_max_distance = \
            distances[1] * cake.d2m / km
        conf.qseis_s_config.sw_flat_earth_transform = 1
        conf.gf_directory = gf_directory

        # find common basement layer
        l = source_model.layer(receiver_basement_depth)
        conf.qseis_s_config.receiver_basement_depth = \
            round(l.zbot / km, 1)
        receiver_model = receiver_model.extract(
            depth_max=l.ztop)
        receiver_model.append(l)

    else:
        raise Exception('Backend not supported: %s' % code)

    # fill remaining fomosto params
    fc.earthmodel_1d = source_model.extract(depth_max='cmb')
    fc.earthmodel_receiver_1d = receiver_model
    fc.modelling_code_id = code + '.' + version

    window_extension = 60.   # [s]
    tp = fc.tabulated_phases

    conf.time_region = (
        gf.Timing(tp[0].id + '-%s' % (1.1 * window_extension)),
        gf.Timing(tp[1].id + '+%s' % (1.6 * window_extension)))

    conf.cut = (
        gf.Timing(tp[0].id + '-%s' % window_extension),
        gf.Timing(tp[1].id + '+%s' % (1.5 * window_extension)))

    conf.relevel_with_fade_in = True

    conf.fade = (
        gf.Timing(tp[0].id + '-%s' % (1.1 * window_extension)),
        gf.Timing(tp[0].id + '-%s' % window_extension),
        gf.Timing(tp[1].id + '+%s' % (1.5 * window_extension)),
        gf.Timing(tp[1].id + '+%s' % (1.6 * window_extension)))

    return conf


def seis_construct_gf(
    stations, event, seismic_config, crust_ind=0, execute=False, force=False):
    """
    Calculate seismic Greens Functions (GFs) and create a repository 'store'
    that is being used later on repeatetly to calculate the synthetic
    waveforms.

    Parameters
    ----------
    stations : list
        of :class:`pyrocko.model.Station`
        Station object that defines the distance from the event for which the
        GFs are being calculated
    event : :class:`pyrocko.model.Event`
        The event is used as a reference point for all the calculations
        According to the its location the earth model is being built
    seismic_config : :class:`config.SeismicConfig`
    crust_ind : int
        Index to set to the Greens Function store, 0 is reference store
        indexes > 0 use reference model and vary its parameters by a Gaussian
    execute : boolean
        Flag to execute the calculation, if False just setup tested
    force : boolean
        Flag to overwrite existing GF stores
    """

    sf = seismic_config.gf_config

    source_model = get_velocity_model(
        event, earth_model_name=sf.earth_model_name, crust_ind=crust_ind,
        gf_config=sf, custom_velocity_model=sf.custom_velocity_model)

    for station in stations:
        logger.info('Station %s' % station.station)
        logger.info('---------------------')

        fomosto_config = get_fomosto_baseconfig(
            sf, event, station, seismic_config.channels, crust_ind)

        store_dir = sf.store_superdir + fomosto_config.id

        if not os.path.exists(store_dir) or force:
            logger.info('Creating Store at %s' % store_dir)

            receiver_model = get_velocity_model(
                station, earth_model_name=sf.earth_model_name,
                crust_ind=crust_ind, gf_config=sf)

            gf_directory = os.path.join(
                sf.store_superdir, 'base_gfs_%i' % crust_ind)

            conf = choose_backend(
                fomosto_config, sf.code, source_model, receiver_model,
                seismic_config.distances, gf_directory)

            fomosto_config.validate()
            conf.validate()

            gf.Store.create_editables(
                store_dir,
                config=fomosto_config,
                extra={sf.code: conf},
                force=force)
        else:
            logger.info(
                'Store %s exists! Use force=True to overwrite!' % store_dir)

        traces_path = os.path.join(store_dir, 'traces')

        if execute and not os.path.exists(traces_path):
            logger.info('Filling store ...')
            store = gf.Store(store_dir, 'r')
            store.make_ttt(force=force)
            store.close()
            backend_builders[sf.code](
                store_dir, nworkers=sf.nworkers, force=force)

            if sf.rm_gfs and sf.code == 'qssp':
                gf_dir = os.path.join(store_dir, 'qssp_green')
                logger.info('Removing QSSP Greens Functions!')
                shutil.rmtree(gf_dir)
        else:
            logger.info('Traces exists use force=True to overwrite!')


def geo_construct_gf(
    event, geodetic_config, crust_ind=0, execute=True, force=False):
    """
    Calculate geodetic Greens Functions (GFs) and create a repository 'store'
    that is being used later on repeatetly to calculate the synthetic
    displacements.

    Parameters
    ----------
    event : :class:`pyrocko.model.Event`
        The event is used as a reference point for all the calculations
        According to the its location the earth model is being built
    geodetic_config : :class:`config.GeodeticConfig`
    crust_ind : int
        Index to set to the Greens Function store
    execute : boolean
        Flag to execute the calculation, if False just setup tested
    force : boolean
        Flag to overwrite existing GF stores
    """

    gf = geodetic_config.gf_config

    c = psgrn.PsGrnConfigFull()

    n_steps_depth = int((gf.source_depth_max - gf.source_depth_min) / \
        gf.source_depth_spacing) + 1
    n_steps_distance = int(
        (gf.source_distance_max - gf.source_distance_min) / \
        gf.source_distance_spacing) + 1

    c.distance_grid = psgrn.PsGrnSpatialSampling(
        n_steps=n_steps_distance,
        start_distance=gf.source_distance_min,
        end_distance=gf.source_distance_max)

    c.depth_grid = psgrn.PsGrnSpatialSampling(
        n_steps=n_steps_depth,
        start_distance=gf.source_depth_min,
        end_distance=gf.source_depth_max)

    c.sampling_interval = gf.sampling_interval

    # extract source crustal profile and check for water layer
    source_model = get_velocity_model(
        event, earth_model_name=gf.earth_model_name, crust_ind=crust_ind,
        gf_config=gf, custom_velocity_model=gf.custom_velocity_model).extract(
            depth_max=gf.source_depth_max * km)

    # potentially vary source model
    if crust_ind > 0:
        source_model = ensemble_earthmodel(
            source_model,
            num_vary=1,
            error_depth=gf.error_depth,
            error_velocities=gf.error_velocities)[0]

    c.earthmodel_1d = source_model
    c.psgrn_outdir = os.path.join(
        gf.store_superdir, 'psgrn_green_%i' % (crust_ind))
    c.validate()

    util.ensuredir(c.psgrn_outdir)

    runner = psgrn.PsGrnRunner(outdir=c.psgrn_outdir)

    if not execute:
        logger.info('Geo GFs can be created in directory: %s ! '
                    '(execute=True necessary)! GF params: \n' % c.psgrn_outdir)
        print c

    if execute:
        logger.info('Creating Geo GFs in directory: %s' % c.psgrn_outdir)
        runner.run(c, force)


def geo_layer_synthetics(store_superdir, crust_ind, lons, lats, sources,
                         keep_tmp=False, outmode='data'):
    """
    Calculate synthetic displacements for a given Greens Function database
    sources and observation points on the earths surface.

    Parameters
    ----------
    store_superdir : str
        main path to directory containing the different Greensfunction stores
    crust_ind : int
        index of Greens Function store to use
    lons : List of floats
        Longitudes [decimal deg] of observation points
    lats : List of floats
        Latitudes [decimal deg] of observation points
    sources : List of :class:`pscmp.PsCmpRectangularSource`
        Sources to calculate synthetics for
    keep_tmp : boolean
        Flag to keep directories (in '/tmp') where calculated synthetics are
        stored.
    outmode : str
        determines type of output

    Returns
    -------
    :class:`numpy.ndarray` (n_observations; ux-North, uy-East, uz-Down)
    """

    c = pscmp.PsCmpConfigFull()
    c.observation = pscmp.PsCmpScatter(lats=lats, lons=lons)
    c.psgrn_outdir = os.path.join(
        store_superdir, 'psgrn_green_%i/' % (crust_ind))

    # only coseismic displacement
    c.times_snapshots = [0]
    c.rectangular_source_patches = sources

    runner = pscmp.PsCmpRunner(keep_tmp=keep_tmp)
    runner.run(c)
    # returns list of displacements for each snapshot
    return runner.get_results(component='displ', flip_z=True)[0]


slip_directions = {
    'Uparr': {'slip': 1., 'rake': 0.},
    'Uperp': {'slip': 1., 'rake': -90.},
    'Utensile': {'slip': 0., 'rake': 0., 'opening': 1.}}


class FaultGeometry(gf.seismosizer.Cloneable):
    """
    Object to construct complex fault-geometries with several subfaults.
    Stores information for subfault geometries and
    inversion variables (e.g. slip-components).
    Yields patch objects for requested subfault, dataset and component.

    Parameters
    ----------
    datasets : list
        of str of potential dataset fault geometries to be stored
    components : list
        of str of potential inversion variables (e.g. slip-components) to
        be stored
    ordering : :class:`FaultOrdering`
        comprises patch information related to subfaults
    """

    def __init__(self, datasets, components, ordering):
        self.datasets = datasets
        self.components = components
        self._ext_sources = {}
        self.ordering = ordering

    def _check_dataset(self, dataset):
        if dataset not in self.datasets:
            raise Exception('Dataset not included in FaultGeometry')

    def _check_component(self, component):
        if component not in self.components:
            raise Exception('Component not included in FaultGeometry')

    def _check_index(self, index):
        if index > self.nsubfaults - 1:
            raise Exception('Subfault not defined!')

    def get_subfault_key(self, index, dataset, component):

        if dataset is not None:
            self._check_dataset(dataset)
        else:
            dataset = self.datasets[0]

        if component is not None:
            self._check_component(component)
        else:
            component = self.components[0]

        self._check_index(index)

        return dataset + '_' + component + '_' + str(index)

    def setup_subfaults(self, dataset, component, ext_sources, replace=False):

        self._check_dataset(dataset)
        self._check_component(component)

        if len(ext_sources) != self.nsubfaults:
            raise Exception('Setup does not match fault ordering!')

        for i, source in enumerate(ext_sources):
            source_key = self.get_subfault_key(i, dataset, component)

            if source_key not in self._ext_sources.keys() or replace:
                self._ext_sources[source_key] = copy.deepcopy(source)
            else:
                raise Exception('Subfault already specified in geometry!')

    def get_subfault(self, index, dataset=None, component=None):

        source_key = self.get_subfault_key(index, dataset, component)

        if source_key in self._ext_sources.keys():
            return self._ext_sources[source_key]
        else:
            raise Exception('Requested subfault not defined!')

    def get_subfault_patches(self, index, dataset=None, component=None):

        self._check_index(index)

        subfault = self.get_subfault(
            index, dataset=dataset, component=component)
        npw, npl = self.ordering.vmap[index].shp

        return subfault.patches(nl=npl, nw=npw, dataset=dataset)

    def get_all_patches(self, dataset=None, component=None):

        patches = []
        for i in range(self.nsubfaults):
            patches += self.get_subfault_patches(
                i, dataset=dataset, component=component)

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

    @property
    def nsubfaults(self):
        return len(self.ordering.vmap)

    @property
    def nsubpatches(self):
        return self.ordering.npatches


def discretize_sources(
    sources=None, extension_width=0.1, extension_length=0.1,
    patch_width=5000., patch_length=5000., datasets=['geodetic'],
    varnames=['']):
    """
    Extend sources into all directions and discretize sources into patches.
    Rounds dimensions to have no half-patches.

    Parameters
    ----------
    sources : :class:`RectangularSource`
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
    dict with dict of varnames with list of:
        :class:`pscmp.PsCmpRectangularSource` or
        :class:`pyrocko.gf.seismosizer.RectangularSource`
    """

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

    fault = FaultGeometry(datasets, varnames, ordering)

    for dataset in datasets:
        logger.info('Discretizing %s source(s)' % dataset)

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

                ext_sources.append(ext_source)
                logger.info('Extended fault(s): \n %s' % ext_source.__str__())

            fault.setup_subfaults(dataset, var, ext_sources)

    return fault


def geo_construct_gf_linear(
    store_superdir, outpath, crust_ind=0,
    targets=None, fault=None, varnames=[''],
    force=False):
    """
    Create geodetic Greens Function matrix for defined source geometry.

    Parameters
    ----------
    store_superdir : str
        main path to directory containing the different Greensfunction stores
    outpath : str
        absolute path to the directory and filename where to store the
        Green's Functions
    crust_ind : int
        of index of Greens Function store to use
    targets : list
        of :class:`heart.GeodeticTarget`
    fault : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry
    varnames : list
        of str with variable names that are being optimized for
    """

    if os.path.exists(outpath) and not force:
        logger.info("Green's Functions exist! Use --force to"
            " overwrite!")
    else:
        out_gfs = {}
        for var in varnames:
            logger.debug('For slip component: %s' % var)
            gfs_target = []
            for target in targets:
                logger.debug('Target %s' % target.__str__())

                gfs = []
                for source in fault.get_all_patches('geodetic', var):
                    disp = geo_layer_synthetics(
                        store_superdir=store_superdir,
                        crust_ind=crust_ind,
                        lons=target.lons,
                        lats=target.lats,
                        sources=[source],
                        keep_tmp=False)

                    gfs.append((
                        disp[:, 0] * target.los_vector[:, 0] + \
                        disp[:, 1] * target.los_vector[:, 1] + \
                        disp[:, 2] * target.los_vector[:, 2]) * \
                            target.odw)

                gfs_target.append(num.vstack(gfs).T)

        out_gfs[var] = gfs_target
        logger.info("Dumping Green's Functions to %s" % outpath)
        utility.dump_objects(outpath, [out_gfs])


def get_phase_arrival_time(engine, source, target):
    """
    Get arrival time from Greens Function store for respective
    :class:`pyrocko.gf.seismosizer.Target`,
    :class:`pyrocko.gf.meta.Location` pair. The channel of the target
    determines if S or P wave arrival time is returned.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    source : :class:`pyrocko.gf.meta.Location`
        can be therefore :class:`pyrocko.gf.seismosizer.Source` or
        :class:`pyrocko.model.Event`
    target : :class:`pyrocko.gf.seismosizer.Target`

    Returns
    -------
    scalar, float of the arrival time of the wave
    """
    store = engine.get_store(target.store_id)
    dist = target.distance_to(source)
    depth = source.depth
    if target.codes[3] == 'T':
        wave = 'any_S'
    elif target.codes[3] == 'Z':
        wave = 'any_P'
    else:
        raise Exception('Channel not supported! Either: "T" or "Z"')

    return store.t(wave, (depth, dist)) + source.time


def get_phase_taperer(engine, source, target, arrival_taper):
    """
    Create phase taperer according to synthetic travel times from
    source- target pair and taper return :class:`pyrocko.trace.CosTaper`
    according to defined arrival_taper times.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    source : :class:`pyrocko.gf.meta.Location`
        can be therefore :class:`pyrocko.gf.seismosizer.Source` or
        :class:`pyrocko.model.Event`
    target : :class:`pyrocko.gf.seismosizer.Target`
    arrival_taper : :class:`ArrivalTaper`

    Returns
    -------
    :class:`pyrocko.trace.CosTaper`
    """

    arrival_time = get_phase_arrival_time(engine, source, target)
    return trace.CosTaper(float(arrival_time + arrival_taper.a),
                          float(arrival_time + arrival_taper.b),
                          float(arrival_time + arrival_taper.c),
                          float(arrival_time + arrival_taper.d))
    

def seis_synthetics(engine, sources, targets, arrival_taper=None,
                    filterer=None, reference_taperer=None, plot=False,
                    nprocs=1, outmode='array', pre_stack_cut=False):
    """
    Calculate synthetic seismograms of combination of targets and sources,
    filtering and tapering afterwards (filterer)
    tapering according to arrival_taper around P -or S wave.
    If reference_taper the given taper is always used.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    sources : list
        containing :class:`pyrocko.gf.seismosizer.Source` Objects
        reference source is the first in the list!!!
    targets : list
        containing :class:`pyrocko.gf.seismosizer.Target` Objects
    arrival_taper : :class:`ArrivalTaper`
    filterer : :class:`Filterer`
    reference_taperer : :class:`ArrivalTaper`
        if set all the traces are tapered with the specifications of this Taper
    plot : boolean
        flag for looking at traces
    nprocs : int
        number of processors to use for synthetics calculation
    outmode : string
        output format of synthetics can be 'array', 'stacked_traces',
        'full' returns traces unstacked including post-processing
    pre_stack_cut : boolean
        flag to decide wheather prior to stacking the GreensFunction traces
        should be cutted according to the phase arival time and the defined
        taper

    Returns
    -------
    :class:`numpy.ndarray` or List of :class:`pyrocko.trace.Trace`
         with data each row-one target
    """
    taperers = []
    for target in targets:
        if arrival_taper is not None:
            if reference_taperer is None:
                tap = get_phase_taperer(
                    engine=engine,
                    source=sources[0],
                    target=target,
                    arrival_taper=arrival_taper)
		taperers.append(tap)
            else:
                taperers.append(reference_taperer)

    if pre_stack_cut and arrival_taper is not None:
        for t, taperer in zip(targets, taperers):
            t.update_target_times(sources[0], taperer)

        if outmode == 'data':
            logger.warn('data traces will be very short! pre_sum_flag set!')

    t_2 = time()
    response = engine.process(
        sources=sources,
        targets=targets, nprocs=nprocs)
    t_1 = time()
    logger.debug('Synthetics generation time: %f' % (t_1 - t_2))

    nt = len(targets)
    ns = len(sources)
    
    t0 = time()
    synt_trcs = []
    taper_index = [j for _ in range(ns) for j in range(nt)]

    for i, (source, target, tr) in enumerate(response.iter_results()):
        ti = taper_index[i]
        if arrival_taper is not None:
            tr.taper(taperers[ti], inplace=True)

        if filterer is not None:
            # filter traces
            tr.bandpass(corner_hp=filterer.lower_corner,
                    corner_lp=filterer.upper_corner,
                    order=filterer.order)

        if arrival_taper is not None:
            tr.chop(tmin=taperers[ti].a, tmax=taperers[ti].d)

        synt_trcs.append(tr)

    t1 = time()
    logger.debug('Post-process time %f' % (t1 - t0))
    if plot:
        trace.snuffle(synt_trcs)

    t2 = time()
    tmins = num.vstack([synt_trcs[i].tmin for i in range(nt)]).flatten()
    t3 = time()
    logger.debug('Assemble tmins time %f' % (t3 - t2))

    if arrival_taper is not None:
        t4 = time()
        synths = num.vstack(
            [synt_trcs[i].ydata for i in range(len(synt_trcs))])
        t5 = time()
        logger.debug('Assemble traces time %f' % (t5 - t4))
        # stack traces for all sources
        t6 = time()
        if ns > 1:
            outstack = num.zeros([nt, synths.shape[1]])
            for k in range(ns):
                outstack += synths[(k * nt):(k + 1) * nt, :]
        else:
            outstack = synths
        t7 = time()
        logger.debug('Stack traces time %f' % (t7 - t6))

    if outmode == 'stacked_traces':
        if arrival_taper is not None:
            outtraces = []
            for i in range(nt):
                synt_trcs[i].ydata = outstack[i, :]
                outtraces.append(synt_trcs[i])

            return outtraces, tmins
        else:
            raise TypeError(
                'arrival taper has to be defined for %s type!' % outmode)

    elif outmode == 'data':
        return synt_trcs, tmins

    elif outmode == 'array':
        return outstack, tmins

    else:
        raise TypeError('Outmode %s not supported!' % outmode)


def taper_filter_traces(data_traces, arrival_taper=None, filterer=None,
                        tmins=None, plot=False, outmode='array', chop=True):
    """
    Taper and filter data_traces according to given taper and filterers.
    Tapering will start at the given tmin.

    Parameters
    ----------
    data_traces : List
        containing :class:`pyrocko.trace.Trace` objects
    arrival_taper : :class:`ArrivalTaper`
    filterer : :class:`Filterer`
    tmins : list or:class:`numpy.ndarray`
        containing the start times [s] since 1st.January 1970 to start
        tapering
    outmode : str
        defines the output structure, options: "traces", "array"

    Returns
    -------
    :class:`numpy.ndarray`
        with tapered and filtered data traces, rows different traces,
        columns temporal values
    """

    cut_traces = []

    for i, tr in enumerate(data_traces):
        cut_trace = tr.copy()
        if arrival_taper is not None:
            taperer = trace.CosTaper(
                float(tmins[i]),
                float(tmins[i] - arrival_taper.b),
                float(tmins[i] - arrival_taper.a + arrival_taper.c),
                float(tmins[i] - arrival_taper.a + arrival_taper.d))

            # taper and cut traces
            cut_trace.taper(taperer, inplace=True, chop=chop)

        if filterer is not None:
            # filter traces
            cut_trace.bandpass(corner_hp=filterer.lower_corner,
                               corner_lp=filterer.upper_corner,
                               order=filterer.order)

        cut_traces.append(cut_trace)

        if plot:
            trace.snuffle(cut_traces)

    if outmode == 'array':
        if arrival_taper is not None:
            return num.vstack(
                [cut_traces[i].ydata for i in range(len(data_traces))])
        else:
            raise Exception('Cannot return array without tapering!')
    if outmode == 'traces':
        return cut_traces
