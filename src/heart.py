"""
Core module with functions to calculate Greens Functions and synthetics.
Also contains main classes for setup specific parameters.
"""

import os
import logging
import shutil
import copy

from beat import psgrn, pscmp, utility

import numpy as num

from pyrocko.guts import Object, String, Float, Int, Tuple
from pyrocko.guts_array import Array

from pyrocko import crust2x2, gf, cake, orthodrome, trace, util
from pyrocko.cake import GradientLayer
from pyrocko.fomosto import qseis
from pyrocko.fomosto import qssp
from pyrocko.gf.seismosizer import outline_rect_source
from pyrocko.orthodrome import ne_to_latlon
#from pyrocko.fomosto import qseis2d


logger = logging.getLogger('heart')

c = 299792458.  # [m/s]
km = 1000.
d2r = num.pi / 180.
err_depth = 0.1
err_velocities = 0.05

lambda_sensors = {
    'Envisat': 0.056,       # needs updating- no ressource file
    'ERS1': 0.05656461471698113,
    'ERS2': 0.056,          # needs updating
    'JERS': 0.23513133960784313,
    'RadarSat2': 0.055465772433
    }


class RectangularSource(gf.DCSource, gf.seismosizer.Cloneable):
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
                    dtype=num.float,
                    help='Data covariance matrix',
                    optional=True)
    pred_g = Array.T(shape=(None, None),
                    dtype=num.float,
                    help='Model prediction covariance matrix, fault geometry',
                    optional=True)
    pred_v = Array.T(shape=(None, None),
                    dtype=num.float,
                    help='Model prediction covariance matrix, velocity model',
                    optional=True)

    @property
    def p_total(self):
        if self.pred_g is None:
            self.pred_g = num.zeros_like(self.data)

        if self.pred_v is None:
            self.pred_v = num.zeros_like(self.data)

        return self.pred_g + self.pred_v

    @property
    def inverse(self):
        """
        Add and invert ALL uncertainty covariance Matrices.
        """
        Cx = self.p_total + self.data
        if Cx.sum() == 0:
            logger.debug('No covariances given, using I matrix!')
            return num.eye(Cx.shape[0])
        else:
            return num.linalg.inv(Cx)

    @property
    def inverse_p(self):
        """
        Add and invert different MODEL uncertainty covariance Matrices.
        """
        if self.p_total.sum() == 0:
            raise Exception('No model covariance defined!')
        return num.linalg.inv(self.p_total)

    @property
    def inverse_d(self):
        """
        Invert DATA covariance Matrix.
        """
        if self.data is None:
            raise Exception('No data covariance matrix defined!')
        return num.linalg.inv(self.data)

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

        return (N * num.log(2 * num.pi)) + ldet_x


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


class Parameter(Object):
    """
    Optimization parameter determines the bounds of the search space.
    """

    name = String.T(default='depth')
    form = String.T(default='Uniform',
                    help='Type of prior distribution to use. Options:'
                         ' "Uniform", ...')
    lower = Array.T(shape=(None,),
                    dtype=num.float,
                    serialize_as='list',
                    default=num.array([0., 0.]))
    upper = Array.T(shape=(None,),
                    dtype=num.float,
                    serialize_as='list',
                    default=num.array([1., 1.]))
    testvalue = Array.T(shape=(None,),
                        dtype=num.float,
                        serialize_as='list',
                        default=num.array([0.5, 0.5]))

    def __call__(self):
        if self.lower is not None:
            for i in range(self.dimension):
                if self.testvalue[i] > self.upper[i] or \
                    self.testvalue[i] < self.lower[i]:
                    raise Exception('the testvalue of parameter "%s" has to be'
                        'within the upper and lower bounds' % self.name)

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


class GeodeticTarget(Object):
    """
    Overall geodetic data set class
    """

    typ = String.T(default='SAR')


class IFG(GeodeticTarget):
    """
    Interferogram class as a dataset in the optimization.
    """

    track = String.T(default='A')
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
    lats = Array.T(shape=(None,), dtype=num.float, optional=True)
    lons = Array.T(shape=(None,), dtype=num.float, optional=True)
    locx = Array.T(shape=(None,), dtype=num.float, optional=True)
    locy = Array.T(shape=(None,), dtype=num.float, optional=True)
    satellite = String.T(default='Envisat')

    def __str__(self):
        s = 'IFG\n Acquisition Track: %s\n' % self.track
        s += '  timerange: %s - %s\n' % (self.master, self.slave)
        if self.lats is not None:
            s += '  number of pixels: %i\n' % self.lats.size
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

        self.locy, self.locx = orthodrome.latlon_to_ne_numpy(
            loc.lat, loc.lon, self.lats, self.lons)
        return self.locy, self.locx

    @property
    def samples(self):
        if self.lats is not None:
            n = self.lats.size
        elif self.utmn is not None:
            n = self.utmn.size
        elif self.locy is not None:
            n = self.locy.size
        else:
            raise Exception('No coordinates defined!')
        return n


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


def vary_model(earthmod, err_depth=0.1, err_velocities=0.1,
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
    err_depth : scalar, float
        2 sigma error in percent of the depth for the respective layers
    err_velocities : scalar, float
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
    discont_unc = {'410': 3 * km,
                   '520': 4 * km,
                   '660': 8 * km}

    # uncertainties in velocity for upper and lower mantle from Woodward 1991
    mantle_vel_unc = {'200': 0.02,     # above 200
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
                    err_velocities = vel_unc
                    logger.debug('Velocity error: %f ', err_velocities)

            deltavp = float(num.random.normal(
                        0, layer.mtop.vp * err_velocities / 3., 1))

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
            factor_d = err_depth

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


def ensemble_earthmodel(ref_earthmod, num_vary=10, err_depth=0.1,
                        err_velocities=0.1, depth_limit_variation=600 * km):
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
    err_depth : scalar, float
        2 sigma error in percent of the depth for the respective layers
    err_velocities : scalar, float
        2 sigma error in percent of the velocities for the respective layers
    depth_limit_variations : scalar, float
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
            err_depth,
            err_velocities,
            depth_limit_variation)

        if cost > 20:
            logger.debug('Skipped unlikely model %f' % cost)
        else:
            i += 1
            earthmods.append(new_model)

    return earthmods


def seis_construct_gf(
    station, event, store_superdir, code='qssp',
    source_depth_min=0., source_depth_max=10., source_depth_spacing=1.,
    source_distance_radius=10., source_distance_spacing=1.,
    sample_rate=2., depth_limit_variation=600,
    earth_model='ak135-f-average.m', crust_ind=0,
    execute=False, rm_gfs=True, nworkers=1, use_crust2=True,
    replace_water=True, custom_velocity_model=None, force=False):
    """
    Calculate seismic Greens Functions (GFs) and create a repository 'store'
    that is being used later on repeatetly to calculate the synthetic
    waveforms.

    Parameters
    ----------
    station : :class:`pyrocko.model.Station`
        Station object that defines the distance from the event for which the
        GFs are being calculated
    event : :class:`pyrocko.model.Event`
        The event is used as a reference point for all the calculations
        According to the its location the earth model is being built
    store_superdir : str
        Path to the main directory where all the GF stores are stored
    code : str
        Modeling code to use for the calculation of waveforms.
        implemented so far: `qseis`, `qssp`, coming soon `qseis2d`
        QSSP does calculations on a circle thus it is recommended to use it for
        teleseismic distances. QSEIS does calculations on a cylinder and is
        more accurate for near-field seismic waveforms, QSEIS2d is a
        significantly, computationally more efficient version of QSEIS, outputs
        are almost identical
    source_distance_min : scalar, float
        Lower bound [km] for the source-distance grid of GFs to calculate
    source_distance_max : scalar, float
        Upper bound [km] for the source-distance grid of GFs to calculate
    source_distance_spacing : scalar, float
        Spacing [km] for the source-distance grid of GFs to calculate
    source_depth_min : scalar, float
        Lower bound [km] for the source-depth grid of GFs to calculate
    source_depth_max : scalar, float
        Upper bound [km] for the source-depth grid of GFs to calculate
    source_depth_spacing : scalar, float
        Spacing [km] for the source-depth grid of GFs to calculate
    sample_rate : scalar, float
        Temporal sampling rate [Hz] of seismic waveforms
    crust_ind : int
        Index to set to the Greens Function store, 0 is reference store
        indexes > 0 use reference model and vary its parameters by a Gaussian
    depth_limit_variation : scalar, float
        depth threshold [m], layers with depth > than this limit are not varied
    earth_model : str
        Name of the base earth model to be used, check
        :func:`pyrocko.cake.builtin_models` for alternatives,
        default ak135 with medium resolution
    nworkers : int
        Number of processors to use for computations
    rm_gfs : boolean
        Valid if qssp or qseis2d are being used, remove the intermediate
        files after finishing the computation
    replace_water : boolean
        Flag to remove water layers from the crust2.0 profile
    use_crust2 : boolean
        Flag to use the crust2.0 model for the crustal earth model
    custom_velocity_model : :class:`pyrocko.cake.LayeredModel`
        If the implemented velocity models should not be used, a custom
        velocity model can be given here
    execute : boolean
        Flag to execute the calculation, if False just setup tested
    force : boolean
        Flag to overwrite existing GF stores
    """

    # calculate distance to station [m]
    distance = orthodrome.distance_accurate50m(event, station)
    logger.info('Station %s' % station.station)
    logger.info('---------------------')

    if use_crust2:
        # load velocity profile from CRUST2x2 and check for water layer
        profile_station = crust2x2.get_profile(station.lat, station.lon)

        if replace_water:
            thickness_lwater = profile_station.get_layer(crust2x2.LWATER)[0]
            if thickness_lwater > 0.0:
                logger.info('Water layer %f in CRUST model!'
                    ' Remove and add to lower crust' % thickness_lwater)
                thickness_llowercrust = profile_station.get_layer(
                                                crust2x2.LLOWERCRUST)[0]
                thickness_lsoftsed = profile_station.get_layer(
                    crust2x2.LSOFTSED)[0]

                profile_station.set_layer_thickness(crust2x2.LWATER, 0.0)
                profile_station.set_layer_thickness(crust2x2.LSOFTSED,
                        num.ceil(thickness_lsoftsed / 3))
                profile_station.set_layer_thickness(crust2x2.LLOWERCRUST,
                        thickness_llowercrust + \
                        thickness_lwater + \
                        (thickness_lsoftsed - num.ceil(thickness_lsoftsed / 3))
                        )
                profile_station._elevation = 0.0
                logger.info('New Lower crust layer thickness %f' % \
                    profile_station.get_layer(crust2x2.LLOWERCRUST)[0])

        profile_event = crust2x2.get_profile(event.lat, event.lon)

        #extract model for source region
        source_model = cake.load_model(
            earth_model, crust2_profile=profile_event)

        # extract model for receiver stations,
        # lowest layer has to be as well in source layer structure!
        receiver_model = cake.load_model(
            earth_model, crust2_profile=profile_station)

    else:
        global_model = cake.load_model(earth_model)
        source_model = utility.join_models(
            global_model, custom_velocity_model)

        receiver_model = copy.deepcopy(source_model)

    # randomly vary receiver site crustal model
    if crust_ind > 0:
        #moho_depth = receiver_model.discontinuity('moho').z
        receiver_model = ensemble_earthmodel(
            receiver_model,
            num_vary=1,
            err_depth=err_depth,
            err_velocities=err_velocities,
            depth_limit_variation=depth_limit_variation * km)[0]

    # define phases
    tabulated_phases = [
        gf.TPDef(
            id='any_P',
            definition='p,P,p\\,P\\'),
        gf.TPDef(
            id='any_S',
            definition='s,S,s\\,S\\')]

    distance_min = distance - (source_distance_radius * km)
    if distance_min < 0.:
        logger.warn('Minimum grid distance is below zero. Setting it to zero!')
        distance_min = 0.

    # fill config files for fomosto
    fom_conf = gf.ConfigTypeA(
        id='%s_%s_%.3fHz_%s' % (station.station,
                        earth_model.split('-')[0].split('.')[0],
                        sample_rate,
                        crust_ind),
        ncomponents=10,
        sample_rate=sample_rate,
        receiver_depth=0. * km,
        source_depth_min=source_depth_min * km,
        source_depth_max=source_depth_max * km,
        source_depth_delta=source_depth_spacing * km,
        distance_min=distance_min,
        distance_max=distance + (source_distance_radius * km),
        distance_delta=source_distance_spacing * km,
        tabulated_phases=tabulated_phases)

   # slowness taper
    phases = [
        fom_conf.tabulated_phases[i].phases
        for i in range(len(
            fom_conf.tabulated_phases))]

    all_phases = []
    map(all_phases.extend, phases)

    mean_source_depth = num.mean((source_depth_min, source_depth_max))
    distances = num.linspace(fom_conf.distance_min,
                             fom_conf.distance_max,
                             100) * cake.m2d

    arrivals = receiver_model.arrivals(
                            phases=all_phases,
                            distances=distances,
                            zstart=mean_source_depth)

    ps = num.array(
        [arrivals[i].p for i in range(len(arrivals))])

    slownesses = ps / (cake.r2d * cake.d2m / km)

    slowness_taper = (0.0,
                      0.0,
                      1.1 * float(slownesses.max()),
                      1.3 * float(slownesses.max()))

    if code == 'qseis':
        from pyrocko.fomosto.qseis import build
        receiver_model = receiver_model.extract(depth_max=200 * km)
        model_code_id = code
        version = '2006a'
        conf = qseis.QSeisConfig(
            filter_shallow_paths=0,
            slowness_window=slowness_taper,
            wavelet_duration_samples=0.001,
            sw_flat_earth_transform=1,
            sw_algorithm=1,
            qseis_version=version)

    elif code == 'qssp':
        from pyrocko.fomosto.qssp import build
        source_model = copy.deepcopy(receiver_model)
        receiver_model = None
        model_code_id = code
        version = '2010'
        conf = qssp.QSSPConfig(
            qssp_version=version,
            slowness_max=float(num.max(slowness_taper)),
            toroidal_modes=True,
            spheroidal_modes=True,
            source_patch_radius=(fom_conf.distance_delta - \
                                 fom_conf.distance_delta * 0.05) / km)

    ## elif code == 'QSEIS2d':
    ##     from pyrocko.fomosto.qseis2d import build
    ##     model_code_id = 'qseis2d'
    ##     version = '2014'
    ##     conf = qseis2d.QSeis2dConfig()
    ##     conf.qseis_s_config.slowness_window = slowness_taper
    ##     conf.qseis_s_config.calc_slowness_window = 0
    ##     conf.qseis_s_config.receiver_max_distance = 11000.
    ##     conf.qseis_s_config.receiver_basement_depth = 35.
    ##     conf.qseis_s_config.sw_flat_earth_transform = 1
    ##     # extract method still buggy!!!
    ##     receiver_model = receiver_model.extract(
    ##             depth_max=conf.qseis_s_config.receiver_basement_depth * km)

    # fill remaining fomosto params
    fom_conf.earthmodel_1d = source_model.extract(depth_max='cmb')
    fom_conf.earthmodel_receiver_1d = receiver_model
    fom_conf.modelling_code_id = model_code_id + '.' + version

    window_extension = 60.   # [s]

    conf.time_region = (
        gf.Timing(tabulated_phases[0].id + '-%s' % (1.1 * window_extension)),
        gf.Timing(tabulated_phases[1].id + '+%s' % (1.6 * window_extension)))

    conf.cut = (
        gf.Timing(tabulated_phases[0].id + '-%s' % window_extension),
        gf.Timing(tabulated_phases[1].id + '+%s' % (1.5 * window_extension)))

    conf.relevel_with_fade_in = True

    conf.fade = (
        gf.Timing(tabulated_phases[0].id + '-%s' % (1.1 * window_extension)),
        gf.Timing(tabulated_phases[0].id + '-%s' % window_extension),
        gf.Timing(tabulated_phases[1].id + '+%s' % (1.5 * window_extension)),
        gf.Timing(tabulated_phases[1].id + '+%s' % (1.6 * window_extension)))

    fom_conf.validate()
    conf.validate()

    store_dir = store_superdir + fom_conf.id
    logger.info('Creating Store at %s' % store_dir)
    gf.Store.create_editables(store_dir,
                              config=fom_conf,
                              extra={model_code_id: conf},
                              force=force)
    if execute:
        store = gf.Store(store_dir, 'r')
        store.make_ttt(force=force)
        store.close()
        build(store_dir, nworkers=nworkers, force=force)
        if rm_gfs and code == 'qssp':
            gf_dir = os.path.join(store_dir, 'qssp_green')
            logger.info('Removing QSSP Greens Functions!')
            shutil.rmtree(gf_dir)


def geo_construct_gf(
    event, store_superdir,
    source_distance_min=0., source_distance_max=100.,
    source_depth_min=0., source_depth_max=40.,
    source_distance_spacing=5., source_depth_spacing=0.5,
    sampling_interval=1.,
    earth_model='ak135-f-average.m', crust_ind=0,
    replace_water=True, use_crust2=True, custom_velocity_model=None,
    execute=True, force=False):
    """
    Calculate geodetic Greens Functions (GFs) and create a repository 'store'
    that is being used later on repeatetly to calculate the synthetic
    displacements.

    Parameters
    ----------
    event : :class:`pyrocko.model.Event`
        The event is used as a reference point for all the calculations
        According to the its location the earth model is being built
    store_superdir : str
        Path to the main directory where all the GF stores are stored
    source_distance_min : scalar, float
        Lower bound [km] for the source-distance grid of GFs to calculate
    source_distance_max : scalar, float
        Upper bound [km] for the source-distance grid of GFs to calculate
    source_distance_spacing : scalar, float
        Spacing [km] for the source-distance grid of GFs to calculate
    source_depth_min : scalar, float
        Lower bound [km] for the source-depth grid of GFs to calculate
    source_depth_max : scalar, float
        Upper bound [km] for the source-depth grid of GFs to calculate
    source_depth_spacing : scalar, float
        Spacing [km] for the source-depth grid of GFs to calculate
    sampling_interval : scalar, float >= 1.
        Source-distance dependend sampling density of grid points, if == 1
        linear distance sampling, if > 1. exponentially decreasing sampling
        with increasing distance
    earth_model : str
        Name of the base earth model to be used, check
        :func:`pyrocko.cake.builtin_models` for alternatives,
        default ak135 with medium resolution
    crust_ind : int
        Index to set to the Greens Function store
    replace_water : boolean
        Flag to remove water layers from the crust2.0 profile
    use_crust2 : boolean
        Flag to use the crust2.0 model for the crustal earth model
    custom_velocity_model : :class:`pyrocko.cake.LayeredModel`
        If the implemented velocity models should not be used, a custom
        velocity model can be given here
    execute : boolean
        Flag to execute the calculation, if False just setup tested
    force : boolean
        Flag to overwrite existing GF stores
    """

    c = psgrn.PsGrnConfigFull()

    n_steps_depth = int((source_depth_max - source_depth_min) / \
        source_depth_spacing) + 1
    n_steps_distance = int((source_distance_max - source_distance_min) / \
        source_distance_spacing) + 1

    c.distance_grid = psgrn.PsGrnSpatialSampling(
        n_steps=n_steps_distance,
        start_distance=source_distance_min,
        end_distance=source_distance_max)

    c.depth_grid = psgrn.PsGrnSpatialSampling(
        n_steps=n_steps_depth,
        start_distance=source_depth_min,
        end_distance=source_depth_max)

    c.sampling_interval = sampling_interval

    # extract source crustal profile and check for water layer
    if use_crust2:
        source_profile = crust2x2.get_profile(event.lat, event.lon)

        if replace_water:
            thickness_lwater = source_profile.get_layer(crust2x2.LWATER)[0]

            if thickness_lwater > 0.0:
                logger.info('Water layer %f in CRUST model! '
                        'Remove and add to lower crust' % thickness_lwater)

                thickness_llowercrust = source_profile.get_layer(
                                        crust2x2.LLOWERCRUST)[0]

                source_profile.set_layer_thickness(crust2x2.LWATER, 0.0)
                source_profile.set_layer_thickness(
                    crust2x2.LLOWERCRUST,
                    thickness_llowercrust + thickness_lwater)

                source_profile._elevation = 0.0

                logger.info('New Lower crust layer thickness %f' % \
                    source_profile.get_layer(crust2x2.LLOWERCRUST)[0])

        source_model = cake.load_model(
            earth_model,
            crust2_profile=source_profile).extract(
                depth_max=source_depth_max * km)

    else:
        if custom_velocity_model is None:
            raise Exception('custom velocity model not given!')

        logger.info('Using custom model from config file')
        source_model = custom_velocity_model

    # potentially vary source model
    if crust_ind > 0:
        source_model = ensemble_earthmodel(
            source_model,
            num_vary=1,
            err_depth=err_depth,
            err_velocities=err_velocities)[0]

    c.earthmodel_1d = source_model
    c.psgrn_outdir = os.path.join(
        store_superdir, 'psgrn_green_%i' % (crust_ind))
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
    :class:`numpy.ndarray` (n_observations; ux, uy, uz)
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
    ext_sources = []
    npls = []
    npws = []

    data_dict = {}

    for dataset in datasets:
        source_dict = {}
        logger.info('Discretizing %s source(s)' % dataset)
        for var in varnames:
            logger.info('%s slip component' % var)
            param_mod = copy.deepcopy(slip_directions[var])

            for source in sources:

                s = copy.deepcopy(source)
                param_mod['rake'] += s.rake
                s.update(**param_mod)

                ext_source = s.extent_source(
                    extension_width, extension_length,
                    patch_width, patch_length)

                npls.append(int(num.ceil(ext_source.length / patch_length)))
                npws.append(int(num.ceil(ext_source.width / patch_width)))
                ext_sources.append(ext_source)
                logger.info('Extended fault(s): \n %s' % ext_source.__str__())

                patches = []
                for source, npl, npw in zip(ext_sources, npls, npws):
                    patches += source.patches(nl=npl, nw=npw, dataset=dataset)

            source_dict[var] = patches

        data_dict[dataset] = source_dict

    return data_dict


def geo_construct_gf_linear(
    store_superdir, outpath, crust_ind=0,
    targets=None, dsources=None, varnames=[''],
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
    dsources : dict
        discretized sources of :class:`pscmp.PsCmpRectangularSource`
        Sources i.e. sources to calculate synthetics for
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
                for source in dsources['geodetic'][var]:
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

    taperer = trace.CosTaper(float(arrival_time + arrival_taper.a),
                             float(arrival_time + arrival_taper.b),
                             float(arrival_time + arrival_taper.c),
                             float(arrival_time + arrival_taper.d))
    return taperer


def update_targets_times(targets, source, taperers):
    """
    Update the target attributes tmin and tmax to do the stacking
    only in this interval. Adds twice taper fade in time to each taper side.

    Parameters
    ----------
    targets : list
        containing :class:`pyrocko.gf.seismosizer.Target` Objects
    taperers : list
        of :class:`pyrocko.trace.CosTaper`

    Returns
    -------
    list containing :class:`pyrocko.gf.seismosizer.Target` Objects
    """

    utargets = []
    for t, taper in zip(targets, taperers):
        tolerance = 2 * (taper.b - taper.a)
        ct = copy.deepcopy(t)
        ct.tmin = taper.a - tolerance - source.time
        ct.tmax = taper.d + tolerance - source.time
        utargets.append(ct)

    return utargets


def seis_synthetics(engine, sources, targets, arrival_taper=None,
                    filterer=None, reference_taperer=None, plot=False,
                    nprocs=1, outmode='array', pre_sum_cut=True):
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
    pre_sum_cut : boolean
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
                taperers.append(get_phase_taperer(
                    engine=engine,
                    source=sources[0],
                    target=target,
                    arrival_taper=arrival_taper))
            else:
                taperers.append(reference_taperer)

    if pre_sum_cut and arrival_taper is not None:
        targets = update_targets_times(
            targets, sources[0], taperers)
        if outmode == 'data':
            logger.warn('data traces will be very short! pre_sum_flag set!')

    response = engine.process(sources=sources,
                              targets=targets, nprocs=nprocs)

    synt_trcs = []
    for i, (source, target, tr) in enumerate(response.iter_results()):
        if arrival_taper is not None:
            tr.taper(taperers[i], inplace=True)

        if filterer is not None:
            # filter traces
            tr.bandpass(corner_hp=filterer.lower_corner,
                    corner_lp=filterer.upper_corner,
                    order=filterer.order)

        tr.chop(tmin=taperers[i].a, tmax=taperers[i].d)

        synt_trcs.append(tr)

    if plot:
        trace.snuffle(synt_trcs)

    nt = len(targets)
    ns = len(sources)

    tmins = num.vstack([synt_trcs[i].tmin for i in range(nt)]).flatten()

    if arrival_taper is not None:
        synths = num.vstack(
            [synt_trcs[i].ydata for i in range(len(synt_trcs))])

        # stack traces for all sources
        if ns > 1:
            for k in range(ns):
                outstack = num.zeros([nt, synths.shape[1]])
                outstack += synths[(k * nt):(k + 1) * nt, :]
        else:
            outstack = synths

    if outmode == 'stacked_traces':
        if arrival_taper is not None:
            outtraces = []
            for i in range(nt):
                synt_trcs[i].ydata = outstack[i, :]
                outtraces.append(synt_trcs[i])

            return outtraces, tmins
        else:
            raise Exception(
                'arrival taper has to be defined for %s type!' % outmode)

    elif outmode == 'data':
        return synt_trcs, tmins

    elif outmode == 'array':
        return outstack, tmins

    else:
        raise Exception('Outmode %s not supported!' % outmode)


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
