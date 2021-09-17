from logging import getLogger
import os
import copy

from theano import shared

from pyrocko.gf import LocalEngine
from pyrocko.model import load_stations

from beat import config as bconfig
from beat.heart import (PolarityMapping, PolarityResult,
                        pol_synthetics, results_for_export)
from beat.theanof import PolaritySynthesizer

from beat.utility import (adjust_point_units, split_point,
                          update_source, unique_list)

from beat.models.base import Composite
from beat.models.distributions import polarity_llk

from theano.tensor import concatenate

from pymc3 import Deterministic

from collections import OrderedDict


logger = getLogger('polarity')


__all__ = [
    'PolarityComposite']


class PolarityComposite(Composite):

    def __init__(self, polc, project_dir, sources, events, hypers=False):

        super(PolarityComposite, self).__init__(events)

        logger.debug('Setting up polarity structure ...\n')

        self.name = 'polarity'
        self._like_name = 'polarity_like'
        self.synthesizers = {}
        self.sources = sources
        self.config = polc
        self.gamma = shared(0.01, name='gamma', borrow=True)
        self.fixed_rvs = {}

        # TODO think about dataset class, now in config ... maybe very tedious
        self.pmaps = [None] * self.nevents

        self.engine = LocalEngine(
            store_superdirs=[polc.gf_config.store_superdir])
        self.stations_event = OrderedDict()

        for i in range(self.nevents):
            # stations input in pyrocko stations.txt format
            polarity_stations_path = os.path.join(
                project_dir, bconfig.multi_event_stations_name(i))

            logger.info(
                'Loading polarity stations for event %i'
                ' from: %s ' % (i, polarity_stations_path))
            stations = load_stations(polarity_stations_path)
            self.stations_event[i] = stations

        for i, pmap_config in enumerate(self.config.waveforms):
            pmap = PolarityMapping(
                config=pmap_config,
                stations=stations)
            pmap.update_targets(
                self.engine, self.sources[pmap.event_idx])
            self.pmaps[i] = pmap

    @property
    def is_location_fixed(self):
        """
        Returns true if the source location random variables are fixed.
        """
        if 'north_shift' and 'east_shift' and 'depth' in self.fixed_rvs:
            return True
        else:
            return False

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs
        self.input_rvs.update(fixed_rvs)

        hp_names = self.get_hypernames()

        logpts = []
        for i, pmap in enumerate(self.pmaps):
            self.synthesizers[i] = PolaritySynthesizer(
                self.engine, self.sources[pmap.event_idx],
                pmap, self.is_location_fixed)
            llk = polarity_llk(
                pmap.dataset,
                self.synthesizers[i](self.input_rvs),
                self.gamma,
                hyperparams[hp_names[i]])
            logpts.append(llk)

        llks = Deterministic(self._like_name, concatenate((logpts)))
        return llks.sum()

    def get_hypersize(self, hp_name):
        """
        Return size of the hyperparameter

        Parameters
        ----------
        hp_name: str
            of hyperparameter name

        Returns
        -------
        int
        """
        return 1

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__.copy()

    def point2sources(self, point):
        tpoint = copy.deepcopy(point)
        tpoint = adjust_point_units(tpoint)

        hps = self.config.get_hypernames()
        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source_params = list(
            self.sources[0].keys()) + list(self.sources[0].stf.keys())

        for param in list(tpoint.keys()):
            if param not in source_params:
                tpoint.pop(param)

        if 'time' in tpoint:
            if self.nevents == 1:
                tpoint['time'] += self.event.time       # single event
            else:
                for i, event in enumerate(self.events):     # multi event
                    tpoint['time'][i] += event.time

            source_points = split_point(tpoint)

            for i, source in enumerate(self.sources):
                update_source(source, **source_points[i])

    def get_all_station_names(self):
        """
        Returns list of station names in the order of polarity maps.
        """
        us = []
        for pmap in self.polmaps:
            us.extend(pmap.get_station_names())

        return us

    def get_unique_station_names(self):
        """
        Return unique station names from all polarity maps
        """
        return unique_list(self.get_all_station_names())

    def export(
            self, point, results_path, stage_number,
            fix_output=False, force=False, update=False):

        results = self.assemble_results(point)
        for pols, attribute in results_for_export(
                results=results, datatype='polarity'):

            # TODO need human readable format like e.g.: .csv
            filename = '%s_%i.bin' % (attribute, stage_number)
            output = os.path.join(results_path, filename)
            with open(output, 'a+') as fh:
                fh.write('{} {}\n'.format(pols, attribute))
                fh.close()

    def assemble_results(self, point, order='list'):

        if point is None:
            raise ValueError('A point has to be provided!')

        logger.debug('Assembling polarities ...')
        syn_proc_pols, obs_proc_pols = self.get_synthetics(
            point, order='pmap')
        results = []
        for i, pmap in enumerate(self.polmaps):
            pmap_results = []
            for j, observed_polarties in enumerate(obs_proc_pols[i]):
                source_contribution = syn_proc_pols[i][j]
                pmap_results.append(
                    PolarityResult(
                        point=point,
                        processed_obs=observed_polarties,
                        source_contributions=source_contribution))

            if order == 'list':
                results.extend(pmap_results)

            elif order == 'pmap':
                results.append(pmap_results)

            else:
                raise ValueError('Order "%s" is not supported' % order)

        return results

    def get_synthetics(self, point, **kwargs):

        order = kwargs.pop('order', 'list')

        self.point2sources(point)

        synths = []
        obs = []

        for pmap in zip(self.polmaps):
            source = self.sources[pmap.event_idx]
            pmap.update_targets(self.engine, source)
            pmap.update_radiation_weights()
            synthetics = pol_synthetics(
                source, radiation_weights=pmap.get_radiation_weights())

            if order == 'list':
                synths.extend(synthetics)
                obs.extend(pmap.dataset)

            elif order == 'pmap':
                synths.append(synthetics)
                obs.append(pmap.dataset)
            else:
                raise ValueError('Order "%s" is not supported' % order)

        return synths, obs
