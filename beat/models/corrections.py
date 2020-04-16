import numpy as num
from pyrocko import orthodrome
from theano import shared
from theano import config as tconfig

from beat.theanof import EulerPole

from collections import OrderedDict
import logging


logger = logging.getLogger('models.corrections')

km = 1000.
d2r = orthodrome.d2r


class Correction(object):

    def __init__(self, correction_config):
        """
        Setup dataset correction
        """
        self.correction_config = correction_config
        self.correction_names = None

    def get_required_coordinate_names(self):
        raise NotImplementedError('Needs implementation in subclass!')

    def setup_correction(
            self, locy, locx, los_vector, blacklist, correction_names):
        raise NotImplementedError('Needs implementation in subclass!')


class RampCorrection(Correction):

    def get_required_coordinate_names(self):
        return ['east_shifts', 'north_shifts']

    def setup_correction(
            self, locy, locx, los_vector, blacklist, correction_names):

        self.east_shifts = locx
        self.north_shifts = locy

        self.slocx = shared(
            locx.astype(tconfig.floatX) / km, name='localx', borrow=True)
        self.slocy = shared(
            locy.astype(tconfig.floatX) / km, name='localy', borrow=True)

        self.correction_names = correction_names

    def get_displacements(self, hierarchicals, point=None):
        """
        Return synthetic correction displacements caused by orbital ramp.
        """
        if not self.correction_names:
            raise ValueError(
                'Requested correction, but is not setup or configured!')

        azimuth_ramp_name, range_ramp_name, offset_name = self.correction_names
        if not point:
            locx = self.slocx
            locy = self.slocy
            azimuth_ramp = hierarchicals[azimuth_ramp_name]
            range_ramp = hierarchicals[range_ramp_name]
            offset = hierarchicals[offset_name]
        else:
            locx = self.east_shifts / km
            locy = self.north_shifts / km
            try:
                azimuth_ramp = point[azimuth_ramp_name]
                range_ramp = point[range_ramp_name]
                offset = point[offset_name]
            except KeyError:
                azimuth_ramp = hierarchicals[azimuth_ramp_name]
                range_ramp = hierarchicals[range_ramp_name]
                offset = hierarchicals[offset_name]

        ramp = get_ramp_displacement(
            locx=locx, locy=locy,
            azimuth_ramp=azimuth_ramp, range_ramp=range_ramp, offset=offset)
        return ramp


class EulerPoleCorrection(Correction):

    def get_required_coordinate_names(self):
        return ['lons', 'lats']

    def setup_correction(
            self, lats, lons, los_vector, blacklist, correction_names):

        self.los_vector = los_vector
        self.lats = lats
        self.lons = lons
        self.correction_names = correction_names
        self.blacklist = num.array(blacklist)

        self.euler_pole = EulerPole(
            self.lats, self.lons, blacklist)

        self.slos_vector = shared(
            self.los_vector.astype(tconfig.floatX), name='los', borrow=True)

    def get_displacements(self, hierarchicals, point=None):
        """
        Get synthetic correction velocity due to Euler pole rotation.
        """
        if not self.correction_names:
            raise ValueError(
                'Requested correction, but is not setup or configured!')

        pole_lat_name, pole_lon_name, rotation_vel_name = self.correction_names
        if not point:   # theano instance for get_formula
            inputs = OrderedDict()
            for corr_name in self.correction_names:
                inputs[corr_name] = hierarchicals[corr_name]

            vels = self.euler_pole(inputs)
            return (vels * self.slos_vector).sum(axis=1)
        else:       # numpy instance else
            locx = self.lats
            locy = self.lons

            try:
                pole_lat = point[pole_lat_name]
                pole_lon = point[pole_lon_name]
                omega = point[rotation_vel_name]
            except KeyError:
                if len(hierarchicals) == 0:
                    raise ValueError(
                        'No hierarchical parameters initialized,'
                        'but requested! Please check init!')

                pole_lat = hierarchicals[pole_lat_name]
                pole_lon = hierarchicals[pole_lon_name]
                omega = hierarchicals[rotation_vel_name]

            vels = velocities_from_pole(locx, locy, pole_lat, pole_lon, omega)
            if self.blacklist.size > 0:
                vels[self.blacklist] = 0.
            return (vels * self.los_vector).sum(axis=1)


def velocities_from_pole(lats, lons, plat, plon, omega, earth_shape='ellipsoid'):
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
