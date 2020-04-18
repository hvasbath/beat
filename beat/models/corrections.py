from pyrocko import orthodrome
from theano import shared
from theano import config as tconfig

from beat.theanof import EulerPole
from beat.heart import velocities_from_pole, get_ramp_displacement

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
        self.config = correction_config
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
            self, locy, locx, los_vector, blacklist, dataset_name):

        self.east_shifts = locx
        self.north_shifts = locy

        self.slocx = shared(
            locx.astype(tconfig.floatX) / km, name='localx', borrow=True)
        self.slocy = shared(
            locy.astype(tconfig.floatX) / km, name='localy', borrow=True)

        self.correction_names = self.config.get_hierarchical_names(
                    name=dataset_name)

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
            self, lats, lons, los_vector, blacklist, dataset_name):

        self.los_vector = los_vector
        self.lats = lats
        self.lons = lons
        self.correction_names = self.config.get_hierarchical_names(
            name=dataset_name)
        self.blacklist = blacklist

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
