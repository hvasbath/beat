import copy
import os
from collections import OrderedDict
from logging import getLogger

from pymc3 import Deterministic
from pyrocko.gf import LocalEngine
from pyrocko.guts import dump
from pyrocko.model import load_stations
from theano import shared
from theano.tensor import concatenate

from beat import config as bconfig
from beat.heart import (
    PolarityMapping,
    PolarityResult,
    ResultPoint,
    init_polarity_targets,
    pol_synthetics,
    results_for_export,
)
from beat.models.base import Composite
from beat.models.distributions import polarity_llk
from beat.theanof import PolaritySynthesizer
from beat.utility import adjust_point_units, split_point, unique_list, update_source

logger = getLogger("polarity")


__all__ = ["PolarityComposite"]


class PolarityComposite(Composite):
    def __init__(self, polc, project_dir, sources, events, hypers=False):

        super(PolarityComposite, self).__init__(events)

        logger.debug("Setting up polarity structure ...\n")

        self.name = "polarity"
        self._like_name = "polarity_like"
        self._targets = None
        self.synthesizers = {}
        self.sources = sources
        self.config = polc
        self.gamma = shared(0.2, name="gamma", borrow=True)
        self.fixed_rvs = {}
        self.wavemaps = []

        self.engine = LocalEngine(store_superdirs=[polc.gf_config.store_superdir])
        self.stations_event = OrderedDict()

        gfc = self.config.gf_config
        for i in range(self.nevents):
            # stations input in pyrocko stations.txt format
            polarity_stations_path = os.path.join(
                project_dir, bconfig.multi_event_stations_name(i)
            )

            logger.info(
                "Loading polarity stations for event %i"
                " from: %s " % (i, polarity_stations_path)
            )
            stations = load_stations(polarity_stations_path)
            self.stations_event[i] = stations

        for i, pmap_config in enumerate(self.config.waveforms):
            logger.info('Initialising Polarity Map for "%s"' % pmap_config.name)
            targets = init_polarity_targets(
                stations,
                earth_model_name=gfc.earth_model_name,
                sample_rate=gfc.sample_rate,
                crust_inds=[gfc.reference_model_idx],
                reference_location=gfc.reference_location,
                wavename=pmap_config.name,
            )

            pmap = PolarityMapping(
                config=pmap_config, stations=stations, targets=targets, mapnumber=i
            )

            pmap.prepare_data()
            pmap.update_targets(
                self.engine,
                self.sources[pmap.config.event_idx],
                check=True,
                always_raytrace=self.config.gf_config.always_raytrace,
            )
            self.wavemaps.append(pmap)

    @property
    def is_location_fixed(self):
        """
        Returns true if the source location random variables are fixed.
        """
        if "north_shift" and "east_shift" and "depth" in self.fixed_rvs:
            return True
        else:
            return False

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            "Polarity optimization on: \n " " %s" % ", ".join(self.input_rvs.keys())
        )

        self.input_rvs.update(fixed_rvs)

        hp_names = self.get_hypernames()

        logpts = []
        for i, pmap in enumerate(self.wavemaps):

            self.synthesizers[i] = PolaritySynthesizer(
                self.engine,
                self.sources[pmap.config.event_idx],
                pmap,
                self.is_location_fixed,
                self.config.gf_config.always_raytrace,
            )
            llk = polarity_llk(
                pmap.shared_data_array,
                self.synthesizers[i](self.input_rvs),
                self.gamma,
                hyperparams[hp_names[i]],
            )
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
        tpoint.update(self.fixed_rvs)
        tpoint = adjust_point_units(tpoint)

        hps = self.config.get_hypernames()
        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source_params = list(self.sources[0].keys())

        for param in list(tpoint.keys()):
            if param not in source_params:
                tpoint.pop(param)

        if "time" in tpoint:
            if self.nevents == 1:
                tpoint["time"] += self.event.time  # single event
            else:
                for i, event in enumerate(self.events):  # multi event
                    tpoint["time"][i] += event.time

        source_points = split_point(tpoint)

        for i, source in enumerate(self.sources):
            update_source(source, **source_points[i])

    def get_all_station_names(self):
        """
        Returns list of station names in the order of polarity maps.
        """
        us = []
        for pmap in self.wavemaps:
            us.extend(pmap.get_station_names())

        return us

    def get_unique_station_names(self):
        """
        Return unique station names from all polarity maps
        """
        return unique_list(self.get_all_station_names())

    def export(
        self,
        point,
        results_path,
        stage_number,
        fix_output=False,
        force=False,
        update=False,
    ):

        results = self.assemble_results(point)
        for i, result in enumerate(results):
            # TODO need human readable format like e.g.: .csv
            filename = "polarity_result_pmap_%i_%i.yaml" % (i, stage_number)
            output = os.path.join(results_path, filename)
            dump(result, filename=output)

    def assemble_results(self, point, order="list"):

        if point is None:
            raise ValueError("A point has to be provided!")

        logger.debug("Assembling polarities ...")
        synthetic_polarities, observed_polarities = self.get_synthetics(
            point, order="pmap"
        )
        results = []
        pmap_results = []
        res_point = ResultPoint(point=point, post_llk="max")
        for i, pmap in enumerate(self.wavemaps):
            source_contribution = synthetic_polarities[i]
            pmap_results.append(
                PolarityResult(
                    point=res_point,
                    processed_obs=observed_polarities[i],
                    source_contributions=[source_contribution],
                )
            )

            if order == "list":
                results.extend(pmap_results)

            elif order == "pmap":
                results.append(pmap_results)

            else:
                raise ValueError('Order "%s" is not supported' % order)

        return results

    def get_synthetics(self, point, **kwargs):

        order = kwargs.pop("order", "list")

        self.point2sources(point)

        synths = []
        obs = []

        for pmap in self.wavemaps:
            source = self.sources[pmap.config.event_idx]
            pmap.update_targets(
                self.engine,
                source,
                always_raytrace=self.config.gf_config.always_raytrace,
            )
            pmap.update_radiation_weights()
            synthetics = pol_synthetics(
                source, radiation_weights=pmap.get_radiation_weights()
            )

            if order == "list":
                synths.extend(synthetics)
                obs.extend(pmap._prepared_data)

            elif order == "pmap":
                synths.append(synthetics)
                obs.append(pmap._prepared_data)
            else:
                raise ValueError('Order "%s" is not supported' % order)

        return synths, obs

    @property
    def targets(self):
        if self._targets is None:
            ts = []
            for pmap in self.wavemaps:
                ts.extend(pmap.targets)

            self._targets = ts
        return self._targets
