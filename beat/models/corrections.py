from pyrocko import orthodrome
from theano import shared
from theano import config as tconfig

from beat.theanof import EulerPole, StrainRateTensor
from beat.heart import (velocities_from_pole, get_ramp_displacement,
                        velocities_from_strain_rate_tensor)

from collections import OrderedDict
import logging

from numpy import array, zeros


logger = logging.getLogger('models.corrections')

km = 1000.
d2r = orthodrome.d2r


def get_specific_point_rvs(point, varnames, attributes):
    return {attr: point[varname]
            for varname, attr in zip(varnames, attributes)}


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
            self, locy, locx, los_vector, data_mask, dataset_name):

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

        if not point:
            locx = self.slocx
            locy = self.slocy

            kwargs = get_specific_point_rvs(
                hierarchicals,
                varnames=self.correction_names,
                attributes=self.config.get_suffixes())
        else:
            locx = self.east_shifts / km
            locy = self.north_shifts / km
            try:
                kwargs = get_specific_point_rvs(
                    point,
                    varnames=self.correction_names,
                    attributes=self.config.get_suffixes())
            except KeyError:    # fixed variables
                kwargs = get_specific_point_rvs(
                    hierarchicals,
                    varnames=self.correction_names,
                    attributes=self.config.get_suffixes())

        return get_ramp_displacement(locx, locy, **kwargs)


class EulerPoleCorrection(Correction):

    def get_required_coordinate_names(self):
        return ['lons', 'lats']

    def setup_correction(
            self, locy, locx, los_vector, data_mask, dataset_name):

        self.los_vector = los_vector
        self.lats = locy
        self.lons = locx
        self.correction_names = self.config.get_hierarchical_names(
            name=dataset_name)
        self.data_mask = data_mask

        self.euler_pole = EulerPole(
            self.lats, self.lons, data_mask)

        self.slos_vector = shared(
            self.los_vector.astype(tconfig.floatX), name='los', borrow=True)

    def get_displacements(self, hierarchicals, point=None):
        """
        Get synthetic correction velocity due to Euler pole rotation.
        """
        if not self.correction_names:
            raise ValueError(
                'Requested correction, but is not setup or configured!')

        if not point:   # theano instance for get_formula
            inputs = OrderedDict()
            for corr_name in self.correction_names:
                inputs[corr_name] = hierarchicals[corr_name]

            vels = self.euler_pole(inputs)
            return (vels * self.slos_vector).sum(axis=1)
        else:       # numpy instance else
            try:
                kwargs = get_specific_point_rvs(
                    point,
                    varnames=self.correction_names,
                    attributes=self.config.get_suffixes())
            except KeyError:
                if len(hierarchicals) == 0:
                    raise ValueError(
                        'No hierarchical parameters initialized,'
                        'but requested! Please check init!')

                kwargs = get_specific_point_rvs(
                    hierarchicals,
                    varnames=self.correction_names,
                    attributes=self.config.get_suffixes())

            vels = velocities_from_pole(self.lats, self.lons, **kwargs)
            if self.data_mask.size > 0:
                vels[self.data_mask] = 0.
            return (vels * self.los_vector).sum(axis=1)


class StrainRateCorrection(Correction):

    def get_required_coordinate_names(self):
        return ['lons', 'lats']

    def setup_correction(
            self, locy, locx, los_vector, data_mask, dataset_name):

        self.los_vector = los_vector
        self.lats = locy
        self.lons = locx
        self.correction_names = self.config.get_hierarchical_names(
            name=dataset_name)
        self.data_mask = data_mask

        self.strain_rate_tensor = StrainRateTensor(
            self.lats, self.lons, data_mask)

        self.slos_vector = shared(
            self.los_vector.astype(tconfig.floatX), name='los', borrow=True)

    def get_displacements(self, hierarchicals, point=None):
        """
        Get synthetic correction velocity due to Euler pole rotation.
        """
        if not self.correction_names:
            raise ValueError(
                'Requested correction, but is not setup or configured!')

        if not point:   # theano instance for get_formula
            inputs = OrderedDict()
            for corr_name in self.correction_names:
                inputs[corr_name] = hierarchicals[corr_name]

            vels = self.strain_rate_tensor(inputs)
            return (vels * self.slos_vector).sum(axis=1)
        else:       # numpy instance else
            try:
                kwargs = get_specific_point_rvs(
                    point,
                    varnames=self.correction_names,
                    attributes=self.config.get_suffixes())

            except KeyError:
                if len(hierarchicals) == 0:
                    raise ValueError(
                        'No hierarchical parameters initialized,'
                        'but requested! Please check init!')

                kwargs = get_specific_point_rvs(
                    hierarchicals,
                    varnames=self.correction_names,
                    attributes=self.config.get_suffixes())

        valid = array(self.strain_rate_tensor.station_idxs)

        v_xyz = velocities_from_strain_rate_tensor(
            array(self.lats)[valid],
            array(self.lons)[valid],
            **kwargs)

        if valid.size > 0:
            vels = zeros((self.lats.size, 3))
            vels[valid, :] = v_xyz
        else:
            vels = v_xyz

        return (vels * self.los_vector).sum(axis=1)
