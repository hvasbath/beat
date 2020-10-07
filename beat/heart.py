"""
Core module with functions to calculate Greens Functions and synthetics.
Also contains main classes for setup specific parameters.
"""

import os
import logging
import shutil
import copy
from time import time
from collections import OrderedDict

from beat import psgrn, pscmp, utility, qseis2d

from theano import config as tconfig
from theano import shared
import numpy as num
from scipy import linalg

from pyrocko.guts import (Dict, Object, String, StringChoice,
                          Float, Int, Tuple, List, Bool)
from pyrocko.guts_array import Array

from pyrocko import crust2x2, gf, cake, orthodrome, trace, util
from pyrocko.cake import GradientLayer
from pyrocko.fomosto import qseis, qssp
from pyrocko.model import gnss

# from pyrocko.fomosto import qseis2d


logger = logging.getLogger('heart')

c = 299792458.  # [m/s]
km = 1000.
d2r = num.pi / 180.
r2d = 180. / num.pi
near_field_threshold = 9.  # [deg] below that surface waves are calculated

lambda_sensors = {
    'Envisat': 0.056,       # needs updating- no ressource file
    'ERS1': 0.05656461471698113,
    'ERS2': 0.056,          # needs updating
    'JERS': 0.23513133960784313,
    'RadarSat2': 0.055465772433}


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

    cholesky = linalg.cholesky(A, lower=True)
    if inverse:
        cholesky = num.linalg.inv(cholesky)
    return num.log(num.diag(cholesky)).sum() * 2.


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

    data = Array.T(
        shape=(None, None),
        dtype=tconfig.floatX,
        help='Data covariance matrix',
        optional=True)
    pred_g = Array.T(
        shape=(None, None),
        dtype=tconfig.floatX,
        help='Model prediction covariance matrix, fault geometry',
        optional=True)
    pred_v = Array.T(
        shape=(None, None),
        dtype=tconfig.floatX,
        help='Model prediction covariance matrix, velocity model',
        optional=True)

    def __init__(self, **kwargs):
        self.slog_pdet = shared(0., name='cov_normalisation', borrow=True)
        Object.__init__(self, **kwargs)
        self.update_slog_pdet()

    def covs_supported(self):
        return ['pred_g', 'pred_v', 'data']

    def check_matrix_init(self, cov_mat_str=''):
        """
        Check if matrix is initialised and if not set with zeros of size data.
        """
        if cov_mat_str not in self.covs_supported():
            raise NotImplementedError(
                'Covariance term %s not supported' % cov_mat_str)

        cov_mat = getattr(self, cov_mat_str)
        if cov_mat is None:
            cov_mat = num.zeros_like(self.data, dtype=tconfig.floatX)

        if cov_mat.size != self.data.size:
            if cov_mat.sum() == 0.:
                cov_mat = num.zeros_like(self.data, dtype=tconfig.floatX)
            else:
                raise ValueError(
                    '%s covariances defined but size '
                    'inconsistent!' % cov_mat_str)

        setattr(self, cov_mat_str, cov_mat)

    @property
    def p_total(self):

        self.check_matrix_init('pred_g')
        self.check_matrix_init('pred_v')

        return self.pred_g + self.pred_v

    @property
    def inverse(self):
        """
        Add and invert ALL uncertainty covariance Matrices.
        """
        Cx = self.p_total + self.data
        if Cx.sum() == 0:
            raise ValueError('No covariances given!')
        else:
            return num.linalg.inv(Cx).astype(tconfig.floatX)

    @property
    def inverse_p(self):
        """
        Add and invert different MODEL uncertainty covariance Matrices.
        """
        if self.p_total.sum() == 0:
            raise ValueError('No model covariance defined!')
        return num.linalg.inv(self.p_total).astype(tconfig.floatX)

    @property
    def inverse_d(self):
        """
        Invert DATA covariance Matrix.
        """
        if self.data is None:
            raise AttributeError('No data covariance matrix defined!')
        return num.linalg.inv(self.data).astype(tconfig.floatX)

    @property
    def chol(self):
        """
        Cholesky decomposition of ALL uncertainty covariance matrices.
        """
        Cx = self.p_total + self.data
        if Cx.sum() == 0:
            raise ValueError('No covariances given!')
        else:
            return linalg.cholesky(Cx, lower=True).astype(tconfig.floatX)

    @property
    def chol_inverse(self):
        """
        Cholesky decomposition of the Inverse of the Covariance matrix of
        ALL uncertainty covariance
        matrices. To be used as weight in the optimization.

        Returns
        -------
        lower triangle of the cholesky decomposition
        """
        return num.linalg.cholesky(
            self.inverse).T.astype(tconfig.floatX)

    @property
    def log_pdet(self):
        """
        Calculate the log of the determinant of the total matrix.
        """
        ldet_x = num.log(num.diag(self.chol)).sum() * 2.
        return utility.scalar2floatX(ldet_x)

    def update_slog_pdet(self):
        """
        Update shared variable with current log_norm_factor (lnf)
        (for theano models).
        """
        self.slog_pdet.set_value(self.log_pdet)
        self.slog_pdet.astype(tconfig.floatX)


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

    def check_sample_rate_consistency(self, deltat):
        """
        Check if taper durations are consistent with GF sample rate.
        """
        for chop_b in (['b', 'c'], ['a', 'd']):
            duration = self.duration(chop_b)
            ratio = duration / deltat
            utility.error_not_whole(
                ratio,
                errstr='Taper duration %g of %s is inconsistent with'
                       ' sampling rate of %g! Please adjust Taper values!' % (
                           duration, utility.list2string(chop_b), deltat))

    def duration(self, chop_bounds=['b', 'c']):
        t0 = getattr(self, chop_bounds[0])
        t1 = getattr(self, chop_bounds[1])
        return t1 - t0

    def nsamples(self, sample_rate, chop_bounds=['b', 'c']):
        """
        Returns the number of samples a tapered trace would have given
        its sample rate and chop_bounds

        Parameters
        ----------
        sample_rate : float
        """
        return int(num.ceil(sample_rate * self.duration(chop_bounds)))

    @property
    def fadein(self):
        return self.b - self.a

    @property
    def fadeout(self):
        return self.d - self.c

    def get_pyrocko_taper(self, arrival_time):
        """
        Get pyrocko CosTaper object that may be applied to trace operations.

        Parameters
        ----------
        arrival_time : float
            [s] of the reference time around which the taper will be applied

        Returns
        -------
        :class:`pyrocko.trace.CosTaper`
        """
        if not self.a < self.b < self.c < self.d:
            raise ValueError('Taper values violate: a < b < c < d')

        return trace.CosTaper(
            arrival_time + self.a,
            arrival_time + self.b,
            arrival_time + self.c,
            arrival_time + self.d)


class Trace(Object):
    pass


class FilterBase(Object):
    pass


class Filter(FilterBase):
    """
    Filter object defining frequency range of traces after time-domain
    filtering.
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
    stepwise = Bool.T(
        default=False,
        help='If set to true the bandpass filter is done it two'
             ' consecutive steps, first high-pass then low-pass.')


class FrequencyFilter(FilterBase):

    freqlimits = Tuple.T(
        4, Float.T(),
        default=(0.005, 0.006, 166., 200.),
        help='Corner frequencies 4-tuple [Hz] for frequency domain filter.')
    tfade = Float.T(
        default=20.,
        help='Rise/fall time in seconds of taper applied in timedomain at both'
             ' ends of trace.')


class ResultPoint(Object):
    """
    Containing point in solution space.
    """
    post_llk = String.T(
        optional=True,
        help='describes which posterior likelihood value the point belongs to')
    point = Dict.T(
        String.T(),
        Array.T(
            serialize_as='list',
            dtype=tconfig.floatX),
        default={},
        help='Point in Solution space for which result is produced.')
    variance_reductions = Dict.T(
        String.T(),
        Float.T(),
        default={},
        optional=True,
        help='Variance reductions for each dataset.')


class SeismicResult(Object):
    """
    Result object assembling different traces of misfit.
    """
    point = ResultPoint.T(default=ResultPoint.D())
    processed_obs = Trace.T(optional=True)
#    processed_syn = Trace.T(optional=True)
#    processed_res = Trace.T(optional=True)
    arrival_taper = trace.Taper.T(optional=True)
    llk = Float.T(default=0., optional=True)
    taper = trace.Taper.T(optional=True)
    source_contributions = List.T(
        Trace.T(),
        help='synthetics of source individual contributions.')

    @property
    def processed_syn(self):
        if self.source_contributions is not None:
            tr0 = copy.deepcopy(self.source_contributions[0])
            tr0.ydata = num.zeros_like(tr0.ydata)
            for tr in self.source_contributions:
                tr0.ydata += tr.ydata

        return tr0

    @property
    def processed_res(self):
        tr = copy.deepcopy(self.processed_obs)
        tr.set_ydata(
            self.processed_obs.get_ydata() - self.processed_syn.get_ydata())
        return tr


def results_for_export(results, datatype=None, attributes=None):

    if attributes is None:
        if datatype is None:
            raise ValueError(
                'Either datatype or attributes need to be defined!')
        elif datatype == 'geodetic' or datatype == 'seismic':
            attributes = ['processed_obs', 'processed_syn', 'processed_res']
        else:
            raise NotImplementedError(
                'datatype %s not implemented!' % datatype)

    for attribute in attributes:
        try:
            data = [getattr(result, attribute) for result in results]
        except AttributeError:
            raise AttributeError(
                'Result object does not have the attribute '
                '"%s" to export!' % attribute)

        yield data, attribute


sqrt2 = num.sqrt(2.)

physical_bounds = dict(
    east_shift=(-500., 500.),
    north_shift=(-500., 500.),
    depth=(0., 1000.),
    strike=(-90., 420.),
    strike1=(-90., 420.),
    strike2=(-90., 420.),
    dip=(-45., 135.),
    dip1=(-45., 135.),
    dip2=(-45., 135.),
    rake=(-180., 270.),
    rake1=(-180., 270.),
    rake2=(-180., 270.),
    mix=(0, 1),

    diameter=(0., 100.),
    volume_change=(-1e12, 1e12),

    mnn=(-sqrt2, sqrt2),
    mee=(-sqrt2, sqrt2),
    mdd=(-sqrt2, sqrt2),
    mne=(-1., 1.),
    mnd=(-1., 1.),
    med=(-1., 1.),

    w=(-3. / 8. * num.pi, 3. / 8. * num.pi),
    v=(-1. / 3, 1. / 3.),
    kappa=(0., 2 * num.pi),
    sigma=(-num.pi / 2., num.pi / 2.),
    h=(0., 1.),

    length=(0., 7000.),
    width=(0., 500.),
    slip=(0., 150.),
    nucleation_x=(-1., 1.),
    nucleation_y=(-1., 1.),
    magnitude=(-5., 10.),

    time=(-300., 300.),
    time_shift=(-40., 40.),

    delta_time=(0., 100.),
    delta_depth=(0., 300.),
    distance=(0., 300.),

    duration=(0., 600.),
    peak_ratio=(0., 1.),

    durations=(0., 600.),
    uparr=(-0.3, 150.),
    uperp=(-150., 150.),
    nucleation_strike=(0., num.inf),
    nucleation_dip=(0., num.inf),
    velocities=(0.5, 20.0),

    azimuth=(0, 360),
    amplitude=(1., 10e25),
    bl_azimuth=(0, 360),
    bl_amplitude=(0., 0.2),
    locking_depth=(0.1, 100.),

    hypers=(-20., 20.),

    ramp=(-0.01, 0.01),
    offset=(-1.0, 1.0),
    lat=(-90., 90.),
    lon=(-180., 180.),
    omega=(-10., 10.))


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

        supported_vars = list(physical_bounds.keys())

        if self.name not in supported_vars:
            candidate = self.name.split('_')[-1]
            if candidate in supported_vars:
                name = candidate
            elif self.name[0:2] != 'h_':
                raise TypeError(
                    'The parameter "%s" cannot'
                    ' be optimized for!' % self.name)
            else:
                name = 'hypers'
        else:
            name = self.name

        phys_b = physical_bounds[name]
        if self.lower is not None:
            for i in range(self.dimension):
                if self.upper[i] < self.lower[i]:
                    raise ValueError(
                        'The upper parameter bound for'
                        ' parameter "%s" must be higher than the lower'
                        ' bound' % self.name)

                if self.testvalue[i] > self.upper[i] or \
                        self.testvalue[i] < self.lower[i]:
                    raise ValueError(
                        'The testvalue of parameter "%s" has to'
                        ' be within the upper and lower bounds' % self.name)

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

    def random(self, dimension=None):
        """
        Create random samples within the parameter bounds.

        Parameters
        ----------
        dimensions : int
            number of draws from distribution

        Returns
        -------
        :class:`numpy.ndarray` of size (n, m)
        """
        if dimension is None:
            dimension = self.dimension

        try:
            return (self.upper - self.lower) * num.random.rand(
                dimension) + self.lower
        except ValueError:
            raise ValueError(
                'Either use number of patches input vector size for'
                ' variable {} or only [1]]! Now the size is {}!'.format(
                    self.name, self.lower.size))

    @property
    def dimension(self):
        return self.lower.size

    def bound_to_array(self):
        return num.array([self.lower, self.testval, self.upper],
                         dtype=num.float)


class DynamicTarget(gf.Target):

    response = trace.PoleZeroResponse.T(default=None, optional=True)

    def update_response(self, magnification, damping, period):
        z, p, k = proto2zpk(
            magnification, damping, period, quantity='displacement')
        # b, a = zpk2tf(z, p, k)

        if self.response:
            self.response.zeros = z
            self.response.poles = p
            self.response.constant = k
        else:
            logger.debug('Initializing new response!')
            self.response = trace.PoleZeroResponse(
                zeros=z, poles=p, constant=k)

    def update_target_times(self, sources=None, taperer=None):
        """
        Update the target attributes tmin and tmax to do the stacking
        only in this interval. Adds twice taper fade in time to each taper
        side.

        Parameters
        ----------
        source : list
            containing :class:`pyrocko.gf.seismosizer.Source` Objects
        taperer : :class:`pyrocko.trace.CosTaper`
        """

        if sources is None or taperer is None:
            self.tmin = None
            self.tmax = None
        else:
            tolerance = 2 * (taperer.b - taperer.a)
            self.tmin = taperer.a - tolerance
            self.tmax = taperer.d + tolerance


class SeismicDataset(trace.Trace):
    """
    Extension to :class:`pyrocko.trace.Trace` to have
    :class:`Covariance` as an attribute.
    """

    wavename = None
    covariance = None

    @property
    def samples(self):
        if self.covariance.data is not None:
            return self.covariance.data.shape[0]
        else:
            logger.warn(
                'Dataset has no uncertainties! Return full data length!')
            return self.data_len()

    def set_wavename(self, wavename):
        self.wavename = wavename

    @property
    def typ(self):
        return self.wavename + '_' + self.channel

    @classmethod
    def from_pyrocko_trace(cls, trace, **kwargs):
        d = dict(
            tmin=trace.tmin,
            tmax=trace.tmax,
            ydata=trace.ydata,
            station=trace.station,
            location=trace.location,
            channel=trace.channel,
            network=trace.network,
            deltat=trace.deltat)
        return cls(**d)

    def __getstate__(self):
        return (self.network, self.station, self.location, self.channel,
                self.tmin, self.tmax, self.deltat, self.mtime,
                self.ydata, self.meta, self.wavename, self.covariance)

    def __setstate__(self, state):
        self.network, self.station, self.location, self.channel, \
            self.tmin, self.tmax, self.deltat, self.mtime, \
            self.ydata, self.meta, self.wavename, self.covariance = state

        self._growbuffer = None
        self._update_ids()


class GeodeticDataset(gf.meta.MultiLocation):
    """
    Overall geodetic data set class
    """

    typ = String.T(
        default='SAR',
        help='Type of geodetic data, e.g. SAR, GNSS, ...')
    name = String.T(
        default='A',
        help='e.g. GNSS campaign name or InSAR satellite track ')

    def __init__(self, **kwargs):
        self.has_correction = False
        self.corrections = None
        super(GeodeticDataset, self).__init__(**kwargs)

    def get_corrections(self, hierarchicals, point=None):
        """
        Needs to be specified on inherited dataset classes.
        """
        raise NotImplementedError('Needs implementation in subclass')

    def setup_corrections(self, event, correction_configs):
        """
        Initialise geodetic dataset corrections such as Ramps or Euler Poles.
        """
        self.corrections = []
        self.update_local_coords(event)
        for corr_conf in correction_configs:
            corr = corr_conf.init_correction()
            if self.name in corr_conf.dataset_names and corr_conf.enabled:
                logger.info(
                    'Setting up %s correction for %s' % (
                        corr_conf.feature, self.name))
                locx_name, locy_name = corr.get_required_coordinate_names()

                locx = getattr(self, locx_name)
                locy = getattr(self, locy_name)

                blacklist = self.get_blacklist(corr_conf)

                corr.setup_correction(
                    locy=locy, locx=locx, los_vector=self.los_vector,
                    blacklist=blacklist,
                    dataset_name=self.name)
                self.corrections.append(corr)
                self.has_correction = True
            else:
                logger.info(
                    'Not correcting %s for %s' % (
                        self.name, corr_conf.feature))

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

    def get_distances_to_event(self, loc):
        north_shifts, east_shifts = orthodrome.latlon_to_ne_numpy(
            loc.lat, loc.lon, self.lats, self.lons)
        return num.sqrt(north_shifts ** 2 + east_shifts ** 2)

    @property
    def samples(self):
        if self.lats is not None:
            n = self.lats.size
        elif self.north_shifts is not None:
            n = self.north_shifts.size
        else:
            raise ValueError('No coordinates defined!')
        return n


class GNSSCompoundComponent(GeodeticDataset):
    """
    Collecting many GNSS components and merging them into arrays.
    Make synthetics generation more efficient.
    """
    los_vector = Array.T(shape=(None, 3), dtype=num.float, optional=True)
    displacement = Array.T(shape=(None,), dtype=num.float, optional=True)
    component = String.T(
        default='east',
        help='direction of measurement, north/east/up')
    stations = List.T(gnss.GNSSStation.T(optional=True))
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

    def __init__(self, **kwargs):
        self._station2index = None
        super(GNSSCompoundComponent, self).__init__(**kwargs)

    def update_los_vector(self):
        if self.component == 'east':
            c = num.array([0, 1, 0])
        elif self.component == 'north':
            c = num.array([1, 0, 0])
        elif self.component == 'up':
            c = num.array([0, 0, 1])
        else:
            raise ValueError('Component %s not supported' % self.component)

        self.los_vector = num.tile(c, self.samples).reshape(self.samples, 3)
        if num.isnan(self.los_vector).any():
            raise ValueError(
                'There are Nan values in LOS vector for dataset: %s! '
                'Please check source of imported data!' % self.name)
        return self.los_vector

    def __str__(self):
        s = 'GNSS\n compound: \n'
        s += '  component: %s\n' % self.component
        if self.lats is not None:
            s += '  number of stations: %i\n' % self.samples
        return s

    def to_kite_scene(self, bins=(600, 600)):
        from kite.scene import Scene, SceneConfig
        from scipy.stats import binned_statistic_2d

        bin_disp, bin_lat, bin_lon, _ = binned_statistic_2d(
            self.lats, self.lons, self.displacement,
            statistic='mean', bins=bins)

        logger.debug('Setting up the Kite Scene')
        config = SceneConfig()
        config.frame.llLat = bin_lat.min()
        config.frame.llLon = bin_lon.min()
        config.frame.dE = bin_lon[1] - bin_lon[0]
        config.frame.dN = bin_lat[1] - bin_lat[0]
        config.frame.spacing = 'degree'

        config.meta.scene_title = '%s %s' % (self.name, self.component)
        config.meta.scene_id = self.name
        los_vec = self.los_vector[0]
        theta_rad = num.arccos(los_vec[2])
        theta_bin = num.full_like(bin_disp, theta_rad * 180 / num.pi)
        theta_bin[num.isnan(bin_disp)] = num.nan
        if theta_rad == 0:
            phi_rad = 0.
        else:
            phi_rad = num.arcsin(los_vec[1] / num.sin(theta_rad))

        phi_bin = num.full_like(bin_disp, phi_rad * 180 / num.pi)
        phi_bin[num.isnan(theta_bin)] = num.nan

        scene = Scene(
            theta=theta_bin,
            phi=phi_bin,
            displacement=bin_disp,
            config=config)

        return scene

    def get_blacklist(self, corr_config):
        s2idx = self.station_name_index_mapping()
        station_blacklist_idxs = []
        for code in corr_config.station_blacklist:
            try:
                station_blacklist_idxs.append(s2idx[code])
            except KeyError:
                logger.warning(
                    'Blacklisted station %s not in dataset,'
                    ' skipping ...' % code)

        logger.info(
            'Stations with idxs %s got blacklisted!' %
            utility.list2string(station_blacklist_idxs))
        return num.array(station_blacklist_idxs)

    def station_name_index_mapping(self):
        if self._station2index is None:
            self._station2index = dict(
                (station.code, i) for (i, station) in enumerate(
                    self.stations))
        return self._station2index

    @classmethod
    def from_pyrocko_gnss_campaign(
            cls, campaign, components=['north', 'east', 'up']):

        valid_components = ['north', 'east', 'up']

        compounds = []
        for comp in components:
            logger.info('Loading "%s" GNSS component' % comp)
            if comp not in valid_components:
                raise ValueError(
                    'Component: %s not available! '
                    'Valid GNSS components are: %s' % (
                        comp, utility.list2string(valid_components)))

            comp_stations = []
            components = []
            for st in campaign.stations:
                try:
                    components.append(st.components[comp])
                    comp_stations.append(st)
                except KeyError:
                    logger.warngin(
                        'No data for GNSS station: {}'.format(st.code))

            lats, lons = num.array(
                [loc.effective_latlon for loc in comp_stations]).T
            vs = num.array([c.shift for c in components])
            variances = num.power(
                num.array([c.sigma for c in components]), 2)
            compounds.append(cls(
                name=campaign.name,
                typ='GNSS',
                stations=comp_stations,
                displacement=vs,
                covariance=Covariance(data=num.eye(lats.size) * variances),
                lats=lats,
                lons=lons,
                east_shifts=num.zeros_like(lats),
                north_shifts=num.zeros_like(lats),
                component=comp,
                odw=num.ones_like(lats.size)))

        return compounds


class ResultReport(Object):

    solution_point = Dict.T(help='result point')
    post_llk = StringChoice.T(
        choices=['max', 'mean', 'min'],
        default='max',
        help='Value of point of the likelihood distribution.')
    mean_point = Dict.T(
        optional=True,
        default=None,
        help='mean of distributions, used for model'
             ' prediction covariance calculation.')


class IFG(GeodeticDataset):
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

    def update_los_vector(self, force=False):
        """
        Calculate LOS vector for given attributes incidence and heading angles.

        Returns
        -------
        :class:`numpy.ndarray` (n_points, 3)
        """
        if self.los_vector is None or force:
            if self.incidence is None and self.heading is None:
                raise AttributeError(
                    'Incidence and Heading need to be provided!')

            Su = num.cos(num.deg2rad(self.incidence))
            Sn = - num.sin(num.deg2rad(self.incidence)) * \
                num.cos(num.deg2rad(self.heading - 270))
            Se = - num.sin(num.deg2rad(self.incidence)) * \
                num.sin(num.deg2rad(self.heading - 270))
            self.los_vector = num.array([Sn, Se, Su], dtype=num.float).T
            if num.isnan(self.los_vector).any():
                raise ValueError(
                    'There are Nan values in LOS vector for dataset: %s! '
                    'Please check source of imported data!' % self.name)
            return self.los_vector
        else:
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
    mask = Array.T(
        shape=(None,),
        dtype=num.bool,
        help='Mask values for Euler pole region determination. '
             'Click polygon mask in kite!',
        optional=True)

    def export_to_csv(self, filename, displacement=None):
        logger.debug('Exporting dataset as csv to %s', filename)

        if displacement is None:
            displacement = self.displacement

        with open(filename, mode='w') as f:
            f.write(
                'lat[deg], lon[deg], incidence[deg], heading[deg], '
                'displacement[m]\n')
            for lat, lon, inci, head, dis in zip(
                    self.lats, self.lons,
                    self.incidence, self.heading, displacement):
                f.write(
                    '{}, {}, {}, {}, {} \n'.format(
                        lat, lon, inci, head, dis))

    @classmethod
    def from_kite_scene(cls, scene, **kwargs):
        name = os.path.basename(scene.meta.filename)
        logger.info(
            'Attempting to access the full covariance matrix of the kite'
            ' scene %s. If this is not precalculated it will be calculated '
            'now, which may take a significant amount of time...' % name)
        covariance = Covariance(data=scene.covariance.covariance_matrix)

        if scene.quadtree.frame.isDegree():
            lats = num.empty(scene.quadtree.nleaves)
            lons = num.empty(scene.quadtree.nleaves)
            lats.fill(scene.quadtree.frame.llLat)
            lons.fill(scene.quadtree.frame.llLon)
            lons += scene.quadtree.leaf_eastings
            lats += scene.quadtree.leaf_northings
        elif scene.quadtree.frame.isMeter():
            loce = scene.quadtree.leaf_eastings
            locn = scene.quadtree.leaf_northings
            lats, lons = orthodrome.ne_to_latlon(
                lat0=scene.frame.llLat, lon0=scene.frame.llLon,
                north_m=locn, east_m=loce)

        if hasattr(scene, 'polygon_mask'):
            polygons = scene.polygon_mask.polygons
        else:
            polygons = None

        mask = num.full(lats.size, False)
        if polygons:
            logger.info('Found polygon mask in %s! Importing for Euler Pole'
                        ' correction ...' % name)
            from matplotlib.path import Path

            leaf_idxs_rows = scene.quadtree.leaf_northings / scene.frame.dN
            leaf_idxs_cols = scene.quadtree.leaf_eastings / scene.frame.dE

            points = num.vstack([leaf_idxs_cols, leaf_idxs_rows]).T
            for vertices in polygons.values():    # vertexes [cols, rows]
                p = Path(vertices)
                mask |= p.contains_points(points)

        else:
            logger.info('No polygon mask in %s!' % name)

        d = dict(
            name=name,
            displacement=scene.quadtree.leaf_means,
            lons=lons,
            lats=lats,
            covariance=covariance,
            incidence=90 - num.rad2deg(scene.quadtree.leaf_thetas),
            heading=-num.rad2deg(scene.quadtree.leaf_phis) + 180,
            odw=num.ones_like(scene.quadtree.leaf_phis),
            mask=mask)
        return cls(**d)

    def get_blacklist(self, corr_conf):
        """
        Extracts mask from kite scene and returns blacklist indexes-
        maybe during import?!!!
        """
        if corr_conf.feature == 'Euler Pole':
            logger.info('Masking data for Euler Pole estimation!')
            return self.mask
        else:
            return None


class GeodeticResult(Object):
    """
    Result object assembling different geodetic data.
    """
    point = ResultPoint.T(default=ResultPoint.D())
    processed_obs = Array.T(shape=(None,), dtype=num.float, optional=True)
    processed_syn = Array.T(shape=(None,), dtype=num.float, optional=True)
    processed_res = Array.T(shape=(None,), dtype=num.float, optional=True)
    llk = Float.T(default=0., optional=True)


def init_seismic_targets(
        stations, earth_model_name='ak135-f-average.m', channels=['T', 'Z'],
        sample_rate=1.0, crust_inds=[0], interpolation='multilinear',
        reference_location=None, blacklist=[]):
    """
    Initiate a list of target objects given a list of indexes to the
    respective GF store velocity model variation index (crust_inds).

    Parameters
    ----------
    stations : List of :class:`pyrocko.model.Station`
        List of station objects for which the targets are being initialised
    earth_model_name = str
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
    blacklist : stations that are blacklisted later

    Returns
    -------
    List of :class:`DynamicTarget`
    """

    if reference_location is None:
        store_prefixes = [
            copy.deepcopy(station.station) for station in stations]
    else:
        store_prefixes = [
            copy.deepcopy(reference_location.station)
            for station in stations]

    em_name = get_earth_model_prefix(earth_model_name)

    targets = []
    for sta_num, station in enumerate(stations):
        for channel in channels:
            for crust_ind in crust_inds:
                cha = station.get_channel(channel)
                if cha is None:
                    if station.station not in blacklist:
                        logger.warn(
                            'Channel "%s" for station "%s" does not exist!'
                            ' Putting station into blacklist!' % (
                                channel, station.station))
                        blacklist.append(station.station)
                else:
                    targets.append(DynamicTarget(
                        quantity='displacement',
                        codes=(station.network,
                               station.station,
                               '%i' % crust_ind, channel),  # n, s, l, c
                        lat=station.lat,
                        lon=station.lon,
                        azimuth=cha.azimuth,
                        dip=cha.dip,
                        interpolation=interpolation,
                        store_id=get_store_id(
                            store_prefixes[sta_num],
                            em_name,
                            sample_rate,
                            crust_ind)))
    return targets


def get_store_id(prefix, earth_model_name, sample_rate, crust_ind=0):
    return '%s_%s_%.3fHz_%s' % (
        prefix, earth_model_name, sample_rate, crust_ind)


def init_geodetic_targets(
        datasets, earth_model_name='ak135-f-average.m',
        interpolation='nearest_neighbor', crust_inds=[0],
        sample_rate=0.0):
    """
    Initiate a list of Static target objects given a list of indexes to the
    respective GF store velocity model variation index (crust_inds).

    Parameters
    ----------
    datasets : list
        of :class:`heart.GeodeticDataset` for which the targets are being
        initialised
    earth_model_name = str
        Name of the earth model that has been used for GF calculation.
    sample_rate : scalar, float
        sample rate [Hz] of the Greens Functions to use
    crust_inds : List of int
        Indexes of different velocity model realisations, 0 - reference model
    interpolation : str
        Method of interpolation for the Greens Functions, can be 'multilinear'
        or 'nearest_neighbor'

    Returns
    -------
    List of :class:`pyrocko.gf.targets.StaticTarget`
    """

    em_name = get_earth_model_prefix(earth_model_name)

    targets = [gf.StaticTarget(
        lons=d.lons,
        lats=d.lats,
        interpolation=interpolation,
        quantity='displacement',
        store_id=get_store_id('statics', em_name, sample_rate, crust_ind))
        for crust_ind in crust_inds for d in datasets]

    return targets


def vary_model(
        earthmod, error_depth=0.1, error_velocities=0.1,
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

            deltavp = float(
                num.random.normal(
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
                        layer.mbot.vs += (
                            deltavp / layer.mbot.vp_vs_ratio())
                        repeat = 0
                        cost += count
                elif layer.mtop.vp + deltavp / 10 < last_l.mbot.vp:
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
            deltaz = float(
                num.random.normal(
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
    avail_models = cake.builtin_models()

    if earth_model_name not in avail_models and earth_model_name != 'local':
        raise NotImplementedError(
            'Earthmodel name "%s" not available!'
            ' Implemented models: %s' % (
                earth_model_name, utility.list2string(avail_models)))

    if custom_velocity_model is not None:
        logger.info('Using custom model from config file')
        if earth_model_name == 'local':
            logger.info(
                'Using only custom velocity model, not pasting into '
                'global background model.')
            source_model = custom_velocity_model
        else:
            global_model = cake.load_model(earth_model_name)
            source_model = utility.join_models(
                global_model, custom_velocity_model)

    elif gfc.use_crust2:
        logger.info('Using crust2 profile')
        # load velocity profile from CRUST2x2 and check for water layer
        profile = crust2x2.get_profile(location.lat, location.lon)

        if gfc.replace_water:
            logger.debug('Replacing water layers! ...')
            thickness_lwater = profile.get_layer(crust2x2.LWATER)[0]
            if thickness_lwater > 0.0:
                logger.info(
                    'Water layer %f in CRUST model!'
                    ' Remove and add to lower crust' % thickness_lwater)
                thickness_llowercrust = profile.get_layer(
                    crust2x2.LLOWERCRUST)[0]
                thickness_lsoftsed = profile.get_layer(
                    crust2x2.LSOFTSED)[0]

                profile.set_layer_thickness(crust2x2.LWATER, 0.0)
                profile.set_layer_thickness(
                    crust2x2.LSOFTSED,
                    num.ceil(thickness_lsoftsed / 3))
                profile.set_layer_thickness(
                    crust2x2.LLOWERCRUST,
                    thickness_llowercrust +
                    thickness_lwater +
                    (thickness_lsoftsed - num.ceil(thickness_lsoftsed / 3)))

                profile._elevation = 0.0
                logger.info('New Lower crust layer thickness %f' %
                            profile.get_layer(crust2x2.LLOWERCRUST)[0])
        else:
            logger.debug('Not replacing water layers')

        source_model = cake.load_model(
            earth_model_name, crust2_profile=profile)

    else:
        logger.info('Using global model ...')
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

    phases = [phase.phases for phase in fc.tabulated_phases]

    all_phases = []
    for phase in phases:
        all_phases.extend(phase)

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


def get_earth_model_prefix(earth_model_name):
    return earth_model_name.split('-')[0].split('.')[0]


def get_fomosto_baseconfig(
        gfconfig, event, station, waveforms, crust_ind):
    """
    Initialise fomosto config.

    Parameters
    ----------
    gfconfig : :class:`config.NonlinearGFConfig`
    event : :class:`pyrocko.model.Event`
        The event is used as a reference point for all the calculations
        According to the its location the earth model is being built
    station : :class:`pyrocko.model.Station` or
        :class:`heart.ReferenceLocation`
    waveforms : List of str
        Waveforms to calculate GFs for, determines the length of traces
    crust_ind : int
        Index to set to the Greens Function store
    """
    sf = gfconfig

    if gfconfig.code != 'psgrn' and len(waveforms) < 1:
        raise IOError('No waveforms specified! No GFs to be calculated!')

    # calculate event-station distance [m]
    distance = orthodrome.distance_accurate50m(event, station)
    distance_min = distance - (sf.source_distance_radius * km)

    if distance_min < 0.:
        logger.warn(
            'Minimum grid distance is below zero. Setting it to zero!')
        distance_min = 0.

    # define phases
    tabulated_phases = []
    if 'any_P' in waveforms:
        if sf.earth_model_name == 'local':
            definition = 'p,P,p\\,P\\'
        else:
            definition = 'p,P,p\\,P\\,Pv_(cmb)p'

        tabulated_phases.append(gf.TPDef(
            id='any_P',
            definition=definition))

    if 'any_S' in waveforms:
        tabulated_phases.append(gf.TPDef(
            id='any_S',
            definition='s,S,s\\,S\\'))

    # surface waves
    if 'slowest' in waveforms:
        tabulated_phases.append(gf.TPDef(
            id='slowest',
            definition='0.8'))

    return gf.ConfigTypeA(
        id='%s_%s_%.3fHz_%s' % (
            station.station,
            get_earth_model_prefix(sf.earth_model_name),
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
    'qseis2d': qseis2d.build}


def choose_backend(
        fomosto_config, code, source_model, receiver_model,
        gf_directory='qseis2d_green'):
    """
    Get backend related config.
    """

    fc = fomosto_config
    receiver_basement_depth = 150 * km

    distances = num.array([fc.distance_min, fc.distance_max]) * cake.m2d
    slowness_taper = get_slowness_taper(fc, source_model, distances)

    waveforms = [phase.id for phase in fc.tabulated_phases]

    if 'slowest' in waveforms and code != 'qseis':
        raise TypeError(
            'For near-field phases the "qseis" backend has to be used!')

    if code == 'qseis':
        version = '2006a'
        if 'slowest' in waveforms or distances.min() < 10:
            logger.info(
                'Receiver and source'
                ' site structures have to be identical as distance'
                ' and ray depth not high enough for common receiver'
                ' depth!')
            receiver_model = None
            slowness_taper = (0., 0., 0., 0.)
            sw_algorithm = 0
            sw_flat_earth_transform = 0
        else:
            # find common basement layer
            common_basement = source_model.layer(receiver_basement_depth)
            receiver_model = receiver_model.extract(
                depth_max=common_basement.ztop)
            receiver_model.append(common_basement)
            sw_algorithm = 1
            sw_flat_earth_transform = 1

        conf = qseis.QSeisConfig(
            filter_shallow_paths=0,
            slowness_window=slowness_taper,
            wavelet_duration_samples=0.001,
            sw_flat_earth_transform=sw_flat_earth_transform,
            sw_algorithm=sw_algorithm,
            qseis_version=version)

    elif code == 'qssp':
        source_model = copy.deepcopy(receiver_model)
        receiver_model = None
        version = '2010'

        conf = qssp.QSSPConfig(
            qssp_version=version,
            slowness_max=float(num.max(slowness_taper)),
            toroidal_modes=True,
            spheroidal_modes=True,
            source_patch_radius=(
                fc.distance_delta - fc.distance_delta * 0.05) / km)

    elif code == 'qseis2d':
        version = '2014'

        conf = qseis2d.QSeis2dConfig()
        conf.qseis_s_config.slowness_window = slowness_taper
        conf.qseis_s_config.calc_slowness_window = 0
        conf.qseis_s_config.receiver_max_distance = \
            distances[1] * cake.d2m / km
        conf.qseis_s_config.sw_flat_earth_transform = 1
        conf.gf_directory = gf_directory

        # find common basement layer
        layer = source_model.layer(receiver_basement_depth)
        conf.qseis_s_config.receiver_basement_depth = \
            round(layer.zbot / km, 1)
        receiver_model = receiver_model.extract(
            depth_max=layer.ztop)
        receiver_model.append(layer)

    else:
        raise NotImplementedError('Backend not supported: %s' % code)

    # fill remaining fomosto params
    fc.earthmodel_1d = source_model
    fc.earthmodel_receiver_1d = receiver_model
    fc.modelling_code_id = code + '.' + version

    window_extension = 60.   # [s]

    pids = ['stored:' + wave for wave in waveforms]

    conf.time_region = (
        gf.Timing(
            phase_defs=pids, offset=(-1.1 * window_extension), select='first'),
        gf.Timing(
            phase_defs=pids, offset=(1.6 * window_extension), select='last'))

    conf.cut = (
        gf.Timing(
            phase_defs=pids, offset=-window_extension, select='first'),
        gf.Timing(
            phase_defs=pids, offset=(1.5 * window_extension), select='last'))

    conf.relevel_with_fade_in = True

    conf.fade = (
        gf.Timing(
            phase_defs=pids, offset=-window_extension, select='first'),
        gf.Timing(
            phase_defs=pids, offset=(-0.1) * window_extension, select='first'),
        gf.Timing(
            phase_defs=pids, offset=(window_extension), select='last'),
        gf.Timing(
            phase_defs=pids, offset=(1.6 * window_extension), select='last'))

    return conf


def seis_construct_gf(
        stations, event, seismic_config, crust_ind=0, execute=False,
        force=False):
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

    waveforms = seismic_config.get_waveform_names()

    for station in stations:
        logger.info('Station %s' % station.station)
        logger.info('---------------------')

        fomosto_config = get_fomosto_baseconfig(
            sf, event, station, waveforms, crust_ind)

        store_dir = os.path.join(sf.store_superdir, fomosto_config.id)

        if not os.path.exists(store_dir) or force:
            logger.info('Creating Store at %s' % store_dir)

            if len(stations) == 1:
                custom_velocity_model = sf.custom_velocity_model
            else:
                custom_velocity_model = None

            receiver_model = get_velocity_model(
                station, earth_model_name=sf.earth_model_name,
                crust_ind=crust_ind, gf_config=sf,
                custom_velocity_model=custom_velocity_model)

            gf_directory = os.path.join(
                sf.store_superdir, 'base_gfs_%i' % crust_ind)

            conf = choose_backend(
                fomosto_config, sf.code, source_model, receiver_model,
                gf_directory)

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

        if execute:
            if not os.path.exists(traces_path) or force:
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
                logger.info('Traces exist use force=True to overwrite!')


def geo_construct_gf(
        event, geodetic_config, crust_ind=0, execute=True, force=False):
    """
    Calculate geodetic Greens Functions (GFs) and create a fomosto 'GF store'
    that is being used repeatetly later on to calculate the synthetic
    displacements. Enables various different source geometries.

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
    from pyrocko.fomosto import psgrn_pscmp as ppp

    version = '2008a'
    gfc = geodetic_config.gf_config

    # extract source crustal profile and check for water layer
    source_model = get_velocity_model(
        event, earth_model_name=gfc.earth_model_name,
        crust_ind=crust_ind, gf_config=gfc,
        custom_velocity_model=gfc.custom_velocity_model).extract(
            depth_max=gfc.source_depth_max * km)

    c = ppp.PsGrnPsCmpConfig()

    c.pscmp_config.version = version

    c.psgrn_config.version = version
    c.psgrn_config.sampling_interval = gfc.sampling_interval
    c.psgrn_config.gf_depth_spacing = gfc.medium_depth_spacing
    c.psgrn_config.gf_distance_spacing = gfc.medium_distance_spacing

    station = ReferenceLocation(
        station='statics',
        lat=event.lat,
        lon=event.lon)

    fomosto_config = get_fomosto_baseconfig(
        gfconfig=gfc, event=event, station=station,
        waveforms=[], crust_ind=crust_ind)

    store_dir = os.path.join(gfc.store_superdir, fomosto_config.id)

    if not os.path.exists(store_dir) or force:
        logger.info('Create Store at: %s' % store_dir)
        logger.info('---------------------------')

        # potentially vary source model
        if crust_ind > 0:
            source_model = ensemble_earthmodel(
                source_model,
                num_vary=1,
                error_depth=gfc.error_depth,
                error_velocities=gfc.error_velocities)[0]

        fomosto_config.earthmodel_1d = source_model
        fomosto_config.modelling_code_id = 'psgrn_pscmp.%s' % version

        c.validate()
        fomosto_config.validate()

        gf.store.Store.create_editables(
            store_dir, config=fomosto_config,
            extra={'psgrn_pscmp': c}, force=force)

    else:
        logger.info(
            'Store %s exists! Use force=True to overwrite!' % store_dir)

    traces_path = os.path.join(store_dir, 'traces')

    if execute:
        if not os.path.exists(traces_path) or force:
            logger.info('Filling store ...')

            store = gf.store.Store(store_dir, 'r')
            store.close()

            # build store
            try:
                ppp.build(store_dir, nworkers=gfc.nworkers, force=force)
            except ppp.PsCmpError as e:
                if str(e).find('could not start psgrn/pscmp') != -1:
                    logger.warn('psgrn/pscmp not installed')
                    return
                else:
                    raise

        else:
            logger.info('Traces exist use force=True to overwrite!')


def geo_construct_gf_psgrn(
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
    logger.warn(
        'This function is deprecated and might be removed in later versions!')
    gfc = geodetic_config.gf_config

    c = psgrn.PsGrnConfigFull()

    n_steps_depth = int(
        (gfc.source_depth_max - gfc.source_depth_min) /
        gfc.source_depth_spacing) + 1
    n_steps_distance = int(
        (gfc.source_distance_max - gfc.source_distance_min) /
        gfc.source_distance_spacing) + 1

    c.distance_grid = psgrn.PsGrnSpatialSampling(
        n_steps=n_steps_distance,
        start_distance=gfc.source_distance_min,
        end_distance=gfc.source_distance_max)

    c.depth_grid = psgrn.PsGrnSpatialSampling(
        n_steps=n_steps_depth,
        start_distance=gfc.source_depth_min,
        end_distance=gfc.source_depth_max)

    c.sampling_interval = gfc.sampling_interval

    # extract source crustal profile and check for water layer
    source_model = get_velocity_model(
        event, earth_model_name=gfc.earth_model_name,
        crust_ind=crust_ind, gf_config=gfc,
        custom_velocity_model=gfc.custom_velocity_model).extract(
            depth_max=gfc.source_depth_max * km)

    # potentially vary source model
    if crust_ind > 0:
        source_model = ensemble_earthmodel(
            source_model,
            num_vary=1,
            error_depth=gfc.error_depth,
            error_velocities=gfc.error_velocities)[0]

    c.earthmodel_1d = source_model
    c.psgrn_outdir = os.path.join(
        gfc.store_superdir, 'psgrn_green_%i' % (crust_ind))
    c.validate()

    util.ensuredir(c.psgrn_outdir)

    runner = psgrn.PsGrnRunner(outdir=c.psgrn_outdir)

    if not execute:
        logger.info('Geo GFs can be created in directory: %s ! '
                    '(execute=True necessary)! GF params: \n' % c.psgrn_outdir)
        print(c)

    if execute:
        logger.info('Creating Geo GFs in directory: %s' % c.psgrn_outdir)
        runner.run(c, force)


def geo_layer_synthetics_pscmp(
        store_superdir, crust_ind, lons, lats, sources,
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


class RayPathError(Exception):
    pass


def get_phase_arrival_time(engine, source, target, wavename=None, snap=True):
    """
    Get arrival time from Greens Function store for respective
    :class:`pyrocko.gf.seismosizer.Target`,
    :class:`pyrocko.gf.meta.Location` pair.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    source : :class:`pyrocko.gf.meta.Location`
        can be therefore :class:`pyrocko.gf.seismosizer.Source` or
        :class:`pyrocko.model.Event`
    target : :class:`pyrocko.gf.seismosizer.Target`
    wavename : string
        of the tabulated phase that determines the phase arrival
        needs to be the Id of a tabulated phase in the respective target.store
        if "None" uses first tabulated phase
    snap : if True
        force arrival time on discrete samples of the store

    Returns
    -------
    scalar, float of the arrival time of the wave
    """
    dist = target.distance_to(source)
    try:
        store = engine.get_store(target.store_id)
    except gf.seismosizer.NoSuchStore:
        raise gf.seismosizer.NoSuchStore(
            'No such store with ID %s found, distance [deg] to event: %f ' % (
                target.store_id, cake.m2d * dist))

    if wavename is None:
        wavename = store.config.tabulated_phases[0].id
        logger.debug(
            'Wavename not specified using '
            'first tabulated phase! %s' % wavename)

    logger.debug('Arrival time for wavename "%s" distance %f [deg]' % (
        wavename, cake.m2d * dist))

    try:
        atime = store.t(wavename, (source.depth, dist)) + source.time
    except TypeError:
        raise RayPathError(
            'No wave-arrival for wavename "%s" distance %f [deg]! '
            'Please adjust the distance range in the wavemap config!' % (
                wavename, cake.m2d * dist))

    if snap:
        deltat = 1. / store.config.sample_rate
        atime = trace.t2ind(atime, deltat, snap=round) * deltat
    return atime


def get_phase_taperer(
        engine, source, wavename, target, arrival_taper, arrival_time=num.nan):
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
    wavename : string
        of the tabulated phase that determines the phase arrival
    target : :class:`pyrocko.gf.seismosizer.Target`
    arrival_taper : :class:`ArrivalTaper`
    arrival_time : shift on arrival time (optional)

    Returns
    -------
    :class:`pyrocko.trace.CosTaper`
    """
    if num.isnan(arrival_time):
        logger.warning('Using source reference for tapering!')
        arrival_time = get_phase_arrival_time(
            engine=engine, source=source, target=target, wavename=wavename)

    return arrival_taper.get_pyrocko_taper(float(arrival_time))


class WaveformMapping(object):
    """
    Maps synthetic waveform parameters to targets, stations and data

    Parameters
    ----------
    name : str
        name of the waveform according to travel time tables
    stations : list
        of :class:`pyrocko.model.Station`
    weights : list
        of theano.shared variables
    channels : list
        of channel names valid for all the stations of this wavemap
    datasets : list
        of :class:`heart.Dataset` inherited from :class:`pyrocko.trace.Trace`
    targets : list
        of :class:`pyrocko.gf.target.Target`
    """
    def __init__(self, name, stations, weights=None, channels=['Z'],
                 datasets=[], targets=[]):

        self.name = name
        self.stations = stations
        self.weights = weights
        self.datasets = datasets
        self.targets = targets
        self.channels = channels
        self.station_correction_idxs = None
        self._prepared_data = None
        self._arrival_times = None
        self._target2index = None
        self._station2index = None

        if self.datasets is not None:
            self._update_trace_wavenames()
            self._update_station_corrections()

    def target_index_mapping(self):
        if self._target2index is None:
            self._target2index = dict(
                (target, i) for (i, target) in enumerate(
                    self.targets))
        return self._target2index

    def station_index_mapping(self):
        if self._station2index is None:
            self._station2index = dict(
                (station, i) for (i, station) in enumerate(
                    self.stations))
        return self._station2index

    def add_weights(self, weights, force=False):
        n_w = len(weights)
        if n_w != self.n_t:
            raise CollectionError(
                'Number of Weights %i inconsistent with targets %i!' % (
                    n_w, self.n_t))

        self.weights = weights

    def _update_station_corrections(self):
        """
        Update station_correction_idx
        """
        s2i = self.station_index_mapping()
        station_idxs = []
        for channel in self.channels:
            station_idxs.extend(
                [s2i[station] for station in self.stations])

        self.station_correction_idxs = num.array(
            station_idxs, dtype='int16')

    def station_weeding(self, event, distances, blacklist=[]):
        """
        Weed stations and related objects based on distances and blacklist.
        Works only a single time after init!
        """

        self.stations = utility.apply_station_blacklist(
            self.stations, blacklist)

        self.stations = utility.weed_stations(
            self.stations, event, distances=distances)

        if self.n_data > 0:
            self.datasets = utility.weed_data_traces(
                self.datasets, self.stations)

        self.targets = utility.weed_targets(self.targets, self.stations)

        # reset mappings
        self._target2index = None
        self._station2index = None

        if self.n_t > 0:
            self._update_station_corrections()

        self.check_consistency()

    def get_station_names(self):
        """
        Returns list of strings of station names
        """
        return [station.station for station in self.stations]

    def check_consistency(self):
        if self.n_t != self.n_data:
            raise CollectionError(
                'Inconsistent number of datasets and targets!')
        elif self.n_t == 0:
            raise CollectionError(
                'No data left in wavemap "%s" after applying the distance '
                'filter! Either (1) Adjust distance range (set "distances" '
                ' parameter in beat.WaveformFitConfig, given in degrees '
                ' epicentral distance) or (2) deactivate the wavemap '
                'completely by setting include=False!' % self._mapid)
        else:
            logger.info('Consistent number of '
                        'datasets and targets in %s wavemap!' % self._mapid)

    def update_interpolation(self, method):
        for target in self.targets:
            target.interpolation = method

    def _update_trace_wavenames(self, wavename=None):
        if wavename is None:
            wavename = self.name

        for dtrace in self.datasets:
            dtrace.set_wavename(wavename)

    @property
    def _mapid(self):
        if hasattr(self, 'mapnumber'):
            return '_'.join((self.name, str(self.mapnumber)))
        else:
            return self.name

    @property
    def time_shifts_id(self):
        return 'time_shifts_' + self._mapid

    @property
    def n_t(self):
        return len(self.targets)

    @property
    def hypersize(self):
        """
        Return the size of the related hyperparameters as an integer.
        """
        nhyp = self.n_t / len(self.channels)
        if nhyp.is_integer():
            return int(nhyp)
        else:
            raise ValueError(
                'hyperparameter size is not integer '
                'for wavemap %s' % self._mapid)

    @property
    def n_data(self):
        return len(self.datasets)

    def get_target_idxs(self, channels=['Z']):

        t2i = self.target_index_mapping()
        dtargets = utility.gather(self.targets, lambda t: t.codes[3])

        tidxs = []
        for cha in channels:
            tidxs.extend([t2i[target] for target in dtargets[cha]])

        return tidxs

    def prepare_data(
            self, source, engine, outmode='array', chop_bounds=['b', 'c']):
        """
        Taper, filter data traces according to given reference event.
        Traces are concatenated to one single array.
        """
        if self._prepared_data is not None:
            logger.debug(
                'Overwriting observed data windows in "%s"!' %
                (self.name + '_' + str(self.mapnumber)))

        if hasattr(self, 'config'):
            arrival_times = num.zeros((self.n_t), dtype=tconfig.floatX)
            for i, target in enumerate(self.targets):
                arrival_times[i] = get_phase_arrival_time(
                    engine=engine, source=source,
                    target=target, wavename=self.name)

            if self.config.preprocess_data:
                logger.debug('Pre-processing data ...')
                filterer = self.config.filterer
            else:
                logger.debug('Not pre-processing data ...')
                filterer = None

            self._prepared_data = taper_filter_traces(
                self.datasets,
                arrival_taper=self.config.arrival_taper,
                filterer=filterer,
                arrival_times=arrival_times,
                outmode=outmode,
                chop_bounds=chop_bounds)

            self._arrival_times = arrival_times
        else:
            raise ValueError('Wavemap needs configuration!')

    @property
    def shared_data_array(self):
        if self._prepared_data is None:
            raise ValueError('Data array is not initialized')
        elif isinstance(self._prepared_data, list):
            raise ValueError(
                'Data got initialized as pyrocko traces, need array!')
        else:
            return shared(
                self._prepared_data,
                name='%s_data' % self.name, borrow=True)


class CollectionError(Exception):
    pass


class DataWaveformCollection(object):
    """
    Collection of available datasets, data-weights, waveforms and
    DynamicTargets used to create synthetics.

    Is used to return Mappings of the waveforms of interest to fit to the
    involved data, weights and synthetics generating objects.

    Parameters
    ----------
    waveforms : list
        of strings of tabulated phases that are to be used for misfit
        calculation
    """
    def __init__(self, stations, waveforms=None):
        self.stations = stations
        self.waveforms = waveforms
        self._deltat = None
        self._targets = OrderedDict()
        self._datasets = OrderedDict()
        self._raw_datasets = OrderedDict()
        self._responses = None
        self._target2index = None
        self._station2index = None

    def adjust_sampling_datasets(self, deltat, snap=False, force=False):

        for tr in self._raw_datasets.values():
            if tr.nslc_id not in self._datasets or force:
                self._datasets[tr.nslc_id] = \
                    utility.downsample_trace(tr, deltat, snap=snap)
            else:
                raise CollectionError(
                    'Downsampled trace %s already in'
                    ' collection!' % utility.list2string(tr.nslc_id))

        self._deltat = deltat

    def _check_collection(self, waveform, errormode='not_in', force=False):
        if errormode == 'not_in':
            if waveform not in self.waveforms:
                raise CollectionError(
                    'Waveform is not contained in collection!')
            else:
                pass

        elif errormode == 'in':
            if waveform in self.waveforms and not force:
                raise CollectionError('Wavefom already in collection!')
            else:
                pass

    @property
    def n_t(self):
        return len(self._targets.keys())

    def add_collection(self, waveform=None, datasets=None, targets=None,
                       weights=None, force=False):
        self.add_waveform(waveform, force=force)
        self.add_targets(waveform, targets, force=force)
        self.add_datasets(waveform, datasets, force=force)

    @property
    def n_waveforms(self):
        return len(self.waveforms)

    def target_index_mapping(self):
        if self._target2index is None:
            self._target2index = dict(
                (target, i) for (i, target) in enumerate(
                    self._targets.values()))
        return self._target2index

    def get_waveform_names(self):
        return self.waveforms

    def get_dataset(self, nslc, raw=False):
        if not raw:
            return self._datasets[nslc]
        else:
            return self._raw_datasets[nslc]

    def add_waveforms(self, waveforms=[], force=False):
        for waveform in waveforms:
            self._check_collection(waveform, errormode='in', force=force)
            self.waveforms.append(waveform)

    def add_responses(self, responses, location=None):

        self._responses = OrderedDict()

        for k, v in responses.items():
            if location is not None:
                k = list(k)
                k[2] = str(location)
                k = tuple(k)

            self._responses[k] = v

    def add_targets(self, targets, replace=False, force=False):

        if replace:
            self._targets = OrderedDict()

        current_targets = self._targets.values()
        for target in targets:
            if target not in current_targets or force:
                self._targets[target.codes] = target
            else:
                logger.warn(
                    'Target %s already in collection!' % str(target.codes))

    def add_datasets(self, datasets, location=None, replace=False,
                     force=False):

        if replace:
            self._datasets = OrderedDict()
            self._raw_datasets = OrderedDict()

        entries = self._raw_datasets.keys()
        for d in datasets:
            if location is not None:
                d.set_location(str(location))

            nslc_id = d.nslc_id
            if nslc_id not in entries or force:
                self._raw_datasets[nslc_id] = d
            else:
                logger.warn(
                    'Dataset %s already in collection!' % str(nslc_id))

    @property
    def n_data(self):
        return len(self._datasets.keys())

    def get_waveform_mapping(
            self, waveform, channels=['Z', 'T', 'R'], quantity='displacement'):

        self._check_collection(waveform, errormode='not_in')

        dtargets = utility.gather(
            self._targets.values(), lambda t: t.codes[3])

        targets = []
        for cha in channels:
            targets.extend(dtargets[cha])

        datasets = []
        discard_targets = []
        for target in targets:
            target.quantity = quantity
            nslc_id = target.codes
            try:
                dtrace = self._datasets[nslc_id]
                datasets.append(dtrace)
            except KeyError:
                logger.warn(
                    'No data trace for target %s in '
                    'the collection! Removing target!' % str(nslc_id))
                discard_targets.append(target)

            if self._responses:
                try:
                    target.update_response(*self._responses[nslc_id])
                except KeyError:
                    logger.warn(
                        'No response for target %s in '
                        'the collection!' % str(nslc_id))

        targets = utility.weed_targets(
            targets, self.stations, discard_targets=discard_targets)

        ndata = len(datasets)
        n_t = len(targets)

        if ndata != n_t:
            logger.warn(
                'Inconsistent number of targets %i '
                'and datasets %i! in wavemap %s init' % (n_t, ndata, waveform))

        return WaveformMapping(
            name=waveform,
            stations=copy.deepcopy(self.stations),
            datasets=copy.deepcopy(datasets),
            targets=copy.deepcopy(targets),
            channels=channels)


def concatenate_datasets(datasets):
    """
    Concatenate datasets to single arrays

    Parameters
    ----------
    datasets : list
        of :class:`GeodeticDataset`

    Returns
    -------
    datasets : 1d :class:numpy.NdArray` n x 1
    los_vectors : 2d :class:numpy.NdArray` n x 3
    odws : 1d :class:numpy.NdArray` n x 1
    Bij : :class:`utility.ListToArrayBijection`
    """

    _disp_list = [data.displacement.astype(tconfig.floatX)
                  for data in datasets]
    _odws_list = [data.odw.astype(tconfig.floatX)
                  for data in datasets]
    _lv_list = [data.update_los_vector().astype(tconfig.floatX)
                for data in datasets]

    # merge geodetic data to calculate residuals on single array
    ordering = utility.ListArrayOrdering(_disp_list, intype='numpy')
    Bij = utility.ListToArrayBijection(ordering, _disp_list)

    odws = Bij.l2a(_odws_list).astype(tconfig.floatX)
    datasets = Bij.l2a(_disp_list).astype(tconfig.floatX)
    los_vectors = Bij.f3map(_lv_list).astype(tconfig.floatX)
    return datasets, los_vectors, odws, Bij


def init_datahandler(
        seismic_config, seismic_data_path='./', responses_path=None):
    """
    Initialise datahandler.

    Parameters
    ----------
    seismic_config : :class:`config.SeismicConfig`
    seismic_data_path : str
        absolute path to the directory of the seismic data

    Returns
    -------
    datahandler : :class:`DataWaveformCollection`
    """
    sc = seismic_config

    stations, data_traces = utility.load_objects(seismic_data_path)

    wavenames = sc.get_waveform_names()

    target_deltat = 1. / sc.gf_config.sample_rate

    targets = init_seismic_targets(
        stations,
        earth_model_name=sc.gf_config.earth_model_name,
        channels=sc.get_unique_channels(),
        sample_rate=sc.gf_config.sample_rate,
        crust_inds=[sc.gf_config.reference_model_idx],
        reference_location=sc.gf_config.reference_location)

    datahandler = DataWaveformCollection(stations, wavenames)
    datahandler.add_datasets(
        data_traces, location=sc.gf_config.reference_model_idx)
    datahandler.adjust_sampling_datasets(target_deltat, snap=True)
    datahandler.add_targets(targets)
    if responses_path:
        responses = utility.load_objects(responses_path)
        datahandler.add_responses(
            responses, location=sc.gf_config.reference_model_idx)
    return datahandler


def init_wavemap(
        waveformfit_config, datahandler=None, event=None, mapnumber=0):
    """
    Initialise wavemap, which sets targets, datasets and stations into
    relation to the seismic Phase of interest and allows individual
    specificiations.

    Parameters
    ----------
    waveformfit_config : :class:`config.WaveformFitConfig`
    datahandler : :class:`DataWaveformCollection`
    event : :class:`pyrocko.model.Event`
    mapnumber : int
        number of wavemap in list of wavemaps

    Returns
    -------
    wmap : :class:`WaveformMapping`
    """
    wc = waveformfit_config
    wmap = datahandler.get_waveform_mapping(
        wc.name, channels=wc.channels, quantity=wc.quantity)
    wmap.config = wc
    wmap.mapnumber = mapnumber

    wmap.config.arrival_taper.check_sample_rate_consistency(
        datahandler._deltat)

    wmap.station_weeding(event, wc.distances, blacklist=wc.blacklist)

    wmap.update_interpolation(wc.interpolation)
    wmap._update_trace_wavenames('_'.join([wc.name, str(wmap.mapnumber)]))

    logger.info('Number of seismic datasets for wavemap: %s: %i ' % (
        wmap._mapid, wmap.n_data))
    return wmap


def post_process_trace(
        trace, taper, filterer, taper_tolerance_factor=0.,
        outmode=None, chop_bounds=['b', 'c'], transfer_function=None):
    """
    Taper, filter and then chop one trace in place.

    Parameters
    ----------
    trace : :class:`SeismicDataset`
    arrival_taper : :class:`pyrocko.trace.Taper`
    filterer : :class:`Filterer`
    taper_tolerance_factor : float
        default: 0 , cut exactly at the taper edges
        taper.fadein times this factor determines added tolerance
    chop_bounds : str
        determines where to chop the trace on the taper attributes
        may be combination of [a, b, c, d]
    """
    if transfer_function:
        # convolve invert False deconvolve invert True
        dummy_filterer = FrequencyFilter()
        trace = trace.transfer(
            dummy_filterer.tfade, dummy_filterer.freqlimits,
            transfer_function=transfer_function,
            invert=False, cut_off_fading=False)
        logger.debug('transfer trace: %s' % trace.__str__())

    if filterer:
        if isinstance(filterer, Filter):
            # filter traces
            # stepwise
            if filterer.stepwise:
                logger.debug('Stepwise HP LP filtering')
                trace.highpass(
                    corner=filterer.lower_corner,
                    order=filterer.order,
                    demean=True)
                trace.lowpass(
                    corner=filterer.upper_corner,
                    order=filterer.order,
                    demean=False)
            else:
                logger.debug('Single BP filtering')
                trace.bandpass(
                    corner_hp=filterer.lower_corner,
                    corner_lp=filterer.upper_corner,
                    order=filterer.order)

        if isinstance(filterer, FrequencyFilter):
            trace = trace.transfer(
                filterer.tfade, filterer.freqlimits,
                invert=False, cut_off_fading=False)

    if taper and outmode != 'data':
        tolerance = (taper.b - taper.a) * taper_tolerance_factor
        lower_cut = getattr(taper, chop_bounds[0]) - tolerance
        upper_cut = getattr(taper, chop_bounds[1]) + tolerance

        logger.debug('taper times: %s' % taper.__str__())
        logger.debug('trace: %s' % trace.__str__())

        trace.extend(lower_cut, upper_cut, fillmethod='zeros')
        trace.taper(taper, inplace=True)
        trace.chop(tmin=lower_cut, tmax=upper_cut, snap=(num.floor, num.floor))
        logger.debug('chopped trace: %s' % trace.__str__())

    return trace


class StackingError(Exception):
    pass


nzeros = {
    'displacement': 2,
    'velocity': 3,
}


def proto2zpk(magnification, damping, period, quantity='displacement'):
    """
    Convert magnification, damping and period of a station to poles and zeros.

    Parameters
    ----------
    magnification : float
        gain of station
    damping : float
        in []
    period : float
        in [s]
    quantity : string
        in which related data are recorded

    Returns
    -------
    lists of zeros, poles and gain
    """
    import cmath

    zeros = num.zeros(nzeros[quantity]).tolist()
    omega0 = 2.0 * num.pi / period
    preal = - damping * omega0
    pimag = 1.0J * omega0 * cmath.sqrt(1.0 - damping ** 2)
    poles = [preal + pimag, preal - pimag]
    return zeros, poles, magnification


def seis_synthetics(
        engine, sources, targets, arrival_taper=None,
        wavename='any_P', filterer=None, reference_taperer=None,
        plot=False, nprocs=1, outmode='array',
        pre_stack_cut=False, taper_tolerance_factor=0.,
        arrival_times=None, chop_bounds=['b', 'c']):
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
    wavename : string
        of the tabulated phase that determines the phase arrival
    filterer : :class:`Filterer`
    plot : boolean
        flag for looking at traces
    nprocs : int
        number of processors to use for synthetics calculation
        --> currently no effect !!!
    outmode : string
        output format of synthetics can be 'array', 'stacked_traces',
        'data' returns traces unstacked including post-processing,
        'tapered_data' returns unstacked but tapered traces
    pre_stack_cut : boolean
        flag to decide wheather prior to stacking the GreensFunction traces
        should be cutted according to the phase arival time and the defined
        taper
    taper_tolerance_factor : float
        tolerance to chop traces around taper.a and taper.d
    arrival_times : None or :class:`numpy.NdArray`
        of phase to apply taper, if None theoretic arrival of ray tracing used
    chop_bounds : list  of str
        determines where to chop the trace on the taper attributes
        may be combination of [a, b, c, d]
    transfer_functions : list
        of transfer functions to convolve the synthetics with

    Returns
    -------
    :class:`numpy.ndarray` or List of :class:`pyrocko.trace.Trace`
         with data each row-one target
    :class:`numpy.ndarray` of tmins for traces
    """
    stackmodes = ['array', 'data', 'stacked_traces', 'tapered_data']

    if outmode not in stackmodes:
        raise StackingError(
            'Outmode "%s" not available! Available: %s' % (
                outmode, utility.list2string(stackmodes)))

    if not arrival_times.all():
        arrival_times = num.zeros((len(targets)), dtype=tconfig.floatX)
        arrival_times[:] = None

    taperers = []
    tapp = taperers.append
    for i, target in enumerate(targets):
        if arrival_taper:
            tapp(get_phase_taperer(
                engine=engine,
                source=sources[0],
                wavename=wavename,
                target=target,
                arrival_taper=arrival_taper,
                arrival_time=arrival_times[i]))

    if pre_stack_cut and arrival_taper and outmode != 'data':
        for t, taperer in zip(targets, taperers):
            t.update_target_times(sources, taperer)

    t_2 = time()
    try:
        response = engine.process(
            sources=sources,
            targets=targets, nprocs=nprocs)
        t_1 = time()
    except IndexError:
        for source in sources:
            print(source)
        raise ValueError('The GF store returned an empty trace!')

    logger.debug('Synthetics generation time: %f' % (t_1 - t_2))
    # logger.debug('Details: %s \n' % response.stats)

    nt = len(targets)
    ns = len(sources)

    t0 = time()
    synt_trcs = []
    sapp = synt_trcs.append
    taper_index = [j for _ in range(ns) for j in range(nt)]

    for i, (source, target, tr) in enumerate(response.iter_results()):
        if arrival_taper:
            taper = taperers[taper_index[i]]
        else:
            taper = None

        tr = post_process_trace(
            trace=tr,
            taper=taper,
            filterer=filterer,
            taper_tolerance_factor=taper_tolerance_factor,
            outmode=outmode,
            chop_bounds=chop_bounds,
            transfer_function=target.response)

        sapp(tr)

    t1 = time()
    logger.debug('Post-process time %f' % (t1 - t0))
    if plot:
        trace.snuffle(synt_trcs)

    if arrival_taper and outmode != 'data':
        try:
            synths = num.vstack([trc.ydata for trc in synt_trcs])
        except ValueError:
            lengths = [trc.ydata.size for trc in synt_trcs]
            tmins = num.array([trc.tmin for trc in synt_trcs])
            tmaxs = num.array([trc.tmax for trc in synt_trcs])
            tmins -= tmins.min()

            print('lengths', lengths)
            print('tmins', tmins)
            print('tmaxs', tmins)
            print('duration', tmaxs - tmins)
            print('arrival_times', arrival_times)
            print('arrival_times norm', arrival_times - arrival_times.min())
            trace.snuffle(synt_trcs)
            raise ValueError('Stacking error, traces different lengths!')

        # stack traces for all sources
        t6 = time()
        if ns == 1:
            outstack = synths
        else:
            outstack = num.zeros([nt, synths.shape[1]])
            for k in range(ns):
                outstack += synths[(k * nt):(k + 1) * nt, :]

        t7 = time()
        logger.debug('Stack traces time %f' % (t7 - t6))

        # get taper times for tapering data as well
        tmins = num.array([getattr(at, chop_bounds[0]) for at in taperers])
    else:
        # no taper defined so return trace tmins
        tmins = num.array([trc.tmin for trc in synt_trcs])

    if outmode == 'stacked_traces':
        if arrival_taper:
            outtraces = []
            oapp = outtraces.append
            for i in range(nt):
                synt_trcs[i].ydata = outstack[i, :]
                oapp(synt_trcs[i])

            return outtraces, tmins
        else:
            raise TypeError(
                'arrival taper has to be defined for %s type!' % outmode)

    elif outmode == 'data':
        return synt_trcs, tmins

    elif outmode == 'tapered_data':
        outlist = [[] for i in range(nt)]
        for i, tr in enumerate(synt_trcs):
            outlist[taper_index[i]].append(tr)

        return outlist, tmins

    elif outmode == 'array':
        logger.debug('Returning...')
        return outstack, tmins

    else:
        raise TypeError('Outmode %s not supported!' % outmode)


def geo_synthetics(
        engine, targets, sources, outmode='stacked_array', plot=False,
        nprocs=1):
    """
    Calculate synthetic displacements for a given static fomosto Greens
    Function database for sources and targets on the earths surface.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    sources : list
        containing :class:`pyrocko.gf.seismosizer.Source` Objects
        reference source is the first in the list!!!
    targets : list
        containing :class:`pyrocko.gf.seismosizer.Target` Objects
    plot : boolean
        flag for looking at synthetics - not implemented yet
    nprocs : int
        number of processors to use for synthetics calculation
        --> currently no effect !!!
    outmode : string
        output format of synthetics can be: 'array', 'arrays',
        'stacked_array','stacked_arrays'

    Returns
    -------
    depends on outmode:
    'stacked_array'
    :class:`numpy.ndarray` (n_observations; ux-North, uy-East, uz-Down)
    'stacked_arrays'
    or list of
    :class:`numpy.ndarray` (target.samples; ux-North, uy-East, uz-Down)
    """

    response = engine.process(sources, targets)
    ns = len(sources)
    nt = len(targets)

    def stack_arrays(targets, disp_arrays):
        stacked_arrays = []
        sapp = stacked_arrays.append
        for target in targets:
            sapp(num.zeros([target.lons.size, 3]))

        for k in range(ns):
            for l in range(nt):
                idx = l + (k * nt)
                stacked_arrays[l] += disp_arrays[idx]

        return stacked_arrays

    disp_arrays = []
    dapp = disp_arrays.append
    for sresult in response.static_results():
        n = sresult.result['displacement.n']
        e = sresult.result['displacement.e']
        u = -sresult.result['displacement.d']
        dapp(num.vstack([n, e, u]).T)

    if outmode == 'arrays':
        return disp_arrays

    elif outmode == 'array':
        return num.vstack(disp_arrays)

    elif outmode == 'stacked_arrays':
        return stack_arrays(targets, disp_arrays)

    elif outmode == 'stacked_array':
        return num.vstack(stack_arrays(targets, disp_arrays))

    else:
        raise ValueError('Outmode %s not available' % outmode)


def taper_filter_traces(
        traces, arrival_taper=None, filterer=None,
        arrival_times=None, plot=False, outmode='array',
        taper_tolerance_factor=0., chop_bounds=['b', 'c']):
    """
    Taper and filter data_traces according to given taper and filterers.
    Tapering will start at the given tmin.

    Parameters
    ----------
    traces : List
        containing :class:`pyrocko.trace.Trace` objects
    arrival_taper : :class:`ArrivalTaper`
    filterer : :class:`Filterer`
    arrival_times : list or:class:`numpy.ndarray`
        containing the start times [s] since 1st.January 1970 to start
        tapering
    outmode : str
        defines the output structure, options: "stacked_traces", "array",
        "data"
    taper_tolerance_factor : float
        tolerance to chop traces around taper.a and taper.d
    chop_bounds : list of len 2
        of taper attributes a, b, c, or d

    Returns
    -------
    :class:`numpy.ndarray`
        with tapered and filtered data traces, rows different traces,
        columns temporal values
    """
    cut_traces = []
    ctpp = cut_traces.append
    for i, tr in enumerate(traces):
        cut_trace = tr.copy()
        cut_trace.set_location('f')

        if arrival_taper:
            taper = arrival_taper.get_pyrocko_taper(float(arrival_times[i]))
        else:
            taper = None

        logger.debug(
            'Filtering, tapering, chopping ... '
            'trace_samples: %i' % cut_trace.ydata.size)

        cut_trace = post_process_trace(
            trace=cut_trace,
            taper=taper,
            filterer=filterer,
            taper_tolerance_factor=taper_tolerance_factor,
            outmode=outmode,
            chop_bounds=chop_bounds)

        ctpp(cut_trace)

    if plot:
        trace.snuffle(cut_traces + traces)

    if outmode == 'array':
        if arrival_taper:
            logger.debug('Returning chopped traces ...')
            return num.vstack(
                [cut_traces[i].ydata for i in range(len(traces))])
        else:
            raise IOError('Cannot return array without tapering!')
    else:
        return cut_traces


def velocities_from_pole(
        lats, lons, plat, plon, omega, earth_shape='ellipsoid'):
    """
    Return horizontal velocities at input locations for rotation around
    given Euler pole

    Parameters
    ----------
    lats: :class:`numpy.NdArray`
        of geographic latitudes [deg] of points to calculate velocities for
    lons: :class:`numpy.NdArray`
        of geographic longitudes [deg] of points to calculate velocities for
    plat: float
        Euler pole latitude [deg]
    plon: float
        Euler pole longitude [deg]
    omega: float
        angle of rotation around Euler pole [deg / million yrs]

    Returns
    -------
    :class:`numpy.NdArray` of velocities [m / yrs] npoints x 3 (NEU)
    """
    r_earth = orthodrome.earthradius

    def cartesian_to_local(lat, lon):
        rlat = lat * d2r
        rlon = lon * d2r
        return num.array([
            [-num.sin(rlat) * num.cos(rlon), -num.sin(rlat) * num.sin(rlon),
             num.cos(rlat)],
            [-num.sin(rlon), num.cos(rlon), num.zeros_like(rlat)],
            [-num.cos(rlat) * num.cos(rlon), -num.cos(rlat) * num.sin(rlon),
             -num.sin(rlat)]])

    npoints = lats.size
    if earth_shape == 'sphere':
        latlons = num.atleast_2d(num.vstack([lats, lons]).T)
        platlons = num.hstack([plat, plon])
        xyz_points = orthodrome.latlon_to_xyz(latlons)
        xyz_pole = orthodrome.latlon_to_xyz(platlons)

    elif earth_shape == 'ellipsoid':
        xyz = orthodrome.geodetic_to_ecef(lats, lons, num.zeros_like(lats))
        xyz_points = num.atleast_2d(num.vstack(xyz).T) / r_earth
        xyz_pole = num.hstack(
            orthodrome.geodetic_to_ecef(plat, plon, 0.)) / r_earth

    omega_rad_yr = omega * 1e-6 * d2r * r_earth
    xyz_poles = num.tile(xyz_pole, npoints).reshape(npoints, 3)

    v_vecs = num.cross(xyz_poles, xyz_points)
    vels_cartesian = omega_rad_yr * v_vecs

    T = cartesian_to_local(lats, lons)
    return num.einsum('ijk->ik', T * vels_cartesian.T).T


def get_ramp_displacement(locx, locy, azimuth_ramp, range_ramp, offset):
    """
    Get synthetic residual plane in azimuth and range direction of the
    satellite.

    Parameters
    ----------
    locx : shared array-like :class:`numpy.ndarray`
        local coordinates [km] in east direction
    locy : shared array-like :class:`numpy.ndarray`
        local coordinates [km] in north direction
    azimuth_ramp : :class:`theano.tensor.Tensor` or :class:`numpy.ndarray`
        vector with ramp parameter in azimuth
    range_ramp : :class:`theano.tensor.Tensor` or :class:`numpy.ndarray`
        vector with ramp parameter in range
    offset : :class:`theano.tensor.Tensor` or :class:`numpy.ndarray`
        scalar of offset in [m]
    """
    return locy * azimuth_ramp + locx * range_ramp + offset


def check_problem_stores(problem, datatypes):
    """
    Check GF stores for empty traces.
    """

    logger.info('Checking stores for empty traces ...')
    corrupted_stores = {}
    for datatype in datatypes:
        engine = problem.composites[datatype].engine
        storeids = engine.get_store_ids()

        cstores = []
        for store_id in storeids:
            store = engine.get_store(store_id)
            stats = store.stats()
            if stats['empty'] > 0:
                cstores.append(store_id)

            engine.close_cashed_stores()

        corrupted_stores[datatype] = cstores

    return corrupted_stores
