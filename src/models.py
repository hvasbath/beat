import os
import time
import copy

import pymc3 as pm

from pymc3 import Metropolis

from pyrocko import gf, util, model, trace
from pyrocko.guts import Object

import numpy as num

import theano.tensor as tt
from theano import config as tconfig
from theano import shared

from beat import theanof, heart, utility, atmcmc, backend
from beat import covariance as cov
from beat import config as bconfig

import logging

logger = logging.getLogger('models')

__all__ = ['GeometryOptimizer', 'sample', 'load_model', 'load_stage',
    'choose_proposal']


class Problem(Object):
    """
    Overarching class for the optimization problems to be solved.

    Parameters
    ----------
    pc : :class:`beat.ProblemConfig`
        Configuration object that contains the problem definition.
    """

    event = None
    model = None
    _seis_like_name = 'seis_like'
    _seismic_flag = False

    _geo_like_name = 'geo_like'
    _geodetic_flag = False

    _like_name = 'like'

    def __init__(self, pc):

        logger.info('Analysing problem ...')
        logger.info('---------------------\n')

        if 'seismic' in pc.datasets:
            self._seismic_flag = True

        if 'geodetic' in pc.datasets:
            self._geodetic_flag = True

    def init_sampler(self, hypers=False):
        """
        Initialise the Sampling algorithm as defined in the configuration file.
        """

        if hypers:
            sc = self.config.hyper_sampler_config
        else:
            sc = self.config.sampler_config

        if self.model is None:
            raise Exception(
                'Model has to be built before initialising the sampler.')

        with self.model:
            if sc.name == 'Metropolis':
                logger.info(
                    '... Initiate Metropolis ... \n'
                    'proposal_distribution %s, tune_interval=%i\n' % (
                    sc.parameters.proposal_dist, sc.parameters.tune_interval))

                t1 = time.time()
                step = Metropolis(
                    tune_interval=sc.parameters.tune_interval,
                    proposal_dist=choose_proposal(sc.parameters.proposal_dist))
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

            elif sc.name == 'ATMCMC':
                logger.info(
                    '... Initiate Adaptive Transitional Metropolis ... \n'
                    ' n_chains=%i, tune_interval=%i\n' % (
                        sc.parameters.n_chains, sc.parameters.tune_interval))

                t1 = time.time()
                step = atmcmc.ATMCMC(
                    n_chains=sc.parameters.n_chains,
                    tune_interval=sc.parameters.tune_interval,
                    coef_variation=sc.parameters.coef_variation,
                    likelihood_name=self._like_name)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

        if self._seismic_flag:
            self.engine.close_cashed_stores()

        return step


class GeometryOptimizer(Problem):
    """
    Defines the model setup to solve the non-linear fault geometry and
    returns the model object.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config):
        logger.info('... Initialising Geometry Optimizer ... \n')

        pc = config.problem_config

        super(GeometryOptimizer, self).__init__(pc)

        # Load event
        if config.event is None:
            if self._seismic_flag:
                self.event = model.load_one_event(
                    os.path.join(
                        config.seismic_config.datadir, 'event.txt'))
            else:
                logger.warn('Found no event information!')
        else:
            self.event = config.event

        # Init sources
        self.sources = []
        for i in range(pc.n_faults):
            if self.event:
                source = heart.RectangularSource.from_pyrocko_event(self.event)
                # hardcoded inversion for hypocentral time
                source.stf.anchor = -1.
            else:
                source = heart.RectangularSource()

            self.sources.append(source)

        dsources = utility.transform_sources(self.sources, pc.datasets)

        if self._seismic_flag:
            logger.debug('Setting up seismic structure ...\n')
            sc = config.seismic_config
            self.engine = gf.LocalEngine(
                store_superdirs=[sc.gf_config.store_superdir])

            seismic_data_path = os.path.join(
                config.project_dir, bconfig.seismic_data_name)
            stations, data_traces = utility.load_objects(
                seismic_data_path)
            stations = utility.apply_station_blacklist(stations, sc.blacklist)

            self.stations = utility.weed_stations(
                stations, self.event, distances=sc.distances)

            self.data_traces = utility.weed_data_traces(
                data_traces, self.stations)

            target_deltat = 1. / sc.gf_config.sample_rate

            if self.data_traces[0].deltat != target_deltat:
                utility.downsample_traces(
                    self.data_traces, deltat=target_deltat)

            self.stargets = heart.init_targets(
                self.stations,
                channels=sc.channels,
                sample_rate=sc.gf_config.sample_rate,
                crust_inds=[0],  # always reference model
                interpolation='multilinear')

            self.ns_t = len(self.stargets)
            logger.info('Number of seismic datasets: %i ' % self.ns_t)

            if sc.calc_data_cov:
                logger.info('Estimating seismic data-covariances ...\n')
                cov_ds_seismic = cov.get_seismic_data_covariances(
                    data_traces=self.data_traces,
                    filterer=sc.filterer,
                    sample_rate=sc.gf_config.sample_rate,
                    arrival_taper=sc.arrival_taper,
                    engine=self.engine,
                    event=self.event,
                    targets=self.stargets)
            else:
                logger.info('No data-covariance estimation ...\n')
                cov_ds_seismic = []
                at = sc.arrival_taper
                n_samples = int(num.ceil(
                    (num.abs(at.a) + at.d) * sc.gf_config.sample_rate))

                for tr in self.data_traces:
                    cov_ds_seismic.append(num.eye(n_samples))

            self.sweights = []
            for s_t in range(self.ns_t):
                self.stargets[s_t].covariance.data = cov_ds_seismic[s_t]
                icov = self.stargets[s_t].covariance.inverse
                self.sweights.append(shared(icov))

            # syntetics generation
            logger.debug('Initialising synthetics functions ... \n')
            self.get_seis_synths = theanof.SeisSynthesizer(
                engine=self.engine,
                sources=dsources['seismic'],
                targets=self.stargets,
                event=self.event,
                arrival_taper=sc.arrival_taper,
                filterer=sc.filterer)

            self.chop_traces = theanof.SeisDataChopper(
                sample_rate=sc.gf_config.sample_rate,
                traces=self.data_traces,
                arrival_taper=sc.arrival_taper,
                filterer=sc.filterer)

        if self._geodetic_flag:
            logger.debug('Setting up geodetic structure ...\n')
            gc = config.geodetic_config

            geodetic_data_path = os.path.join(
                config.project_dir, bconfig.geodetic_data_name)
            self.gtargets = utility.load_objects(geodetic_data_path)

            self.ng_t = len(self.gtargets)
            logger.info('Number of geodetic datasets: %i ' % self.ng_t)

            # geodetic data
            _disp_list = [self.gtargets[i].displacement
                 for i in range(self.ng_t)]
            _lons_list = [self.gtargets[i].lons for i in range(self.ng_t)]
            _lats_list = [self.gtargets[i].lats for i in range(self.ng_t)]
            _odws_list = [self.gtargets[i].odw for i in range(self.ng_t)]
            _lv_list = [self.gtargets[i].update_los_vector()
                            for i in range(self.ng_t)]

            self.gweights = []
            for g_t in range(self.ng_t):
                icov = self.gtargets[g_t].covariance.inverse
                self.gweights.append(shared(icov))

            # merge geodetic data to call pscmp only once each forward model
            ordering = utility.ListArrayOrdering(_disp_list, intype='numpy')
            self.Bij = utility.ListToArrayBijection(ordering, _disp_list)

            odws = self.Bij.fmap(_odws_list)
            lons = self.Bij.fmap(_lons_list)
            lats = self.Bij.fmap(_lats_list)

            logger.info('Number of geodetic data points: %i ' % lats.shape[0])

            self.wdata = shared(self.Bij.fmap(_disp_list) * odws)
            self.lv = shared(self.Bij.f3map(_lv_list))
            self.odws = shared(odws)

            # syntetics generation
            logger.debug('Initialising synthetics functions ... \n')
            self.get_geo_synths = theanof.GeoLayerSynthesizerStatic(
                lats=lats,
                lons=lons,
                store_superdir=gc.gf_config.store_superdir,
                crust_ind=0,    # always reference model
                sources=dsources['geodetic'])

        self.config = config

    def __getstate__(self):
        outstate = (self.config, self.sources)

        if self._seismic_flag:
            outstate = outstate + (
                self.sweights,
                self.stargets,
                self.stations,
                self.engine)

        if self._geodetic_flag:
            outstate = outstate + (
                self.gweights,
                self.gtargets)

        return outstate

    def __setstate__(self, state):
        c = state[0]

        if 'geodetic' in c.problem_config.datasets:
            self._geodetic_flag = True
        else:
            self._geodetic_flag = False

        if 'seismic' in c.problem_config.datasets:
            self._seismic_flag = True
        else:
            self._seismic_flag = False

        if self._seismic_flag and self._geodetic_flag:
            self.config, self.sources, \
            self.sweights, \
            self.stargets, \
            self.stations, \
            self.engine, \
            self.gweights, \
            self.gtargets = state

        elif self._seismic_flag and not self._geodetic_flag:
            self.config, self.sources, \
            self.sweights, \
            self.stargets, \
            self.stations, \
            self.engine = state

        elif not self._seismic_flag and self._geodetic_flag:
            self.config, self.sources, \
            self.gweights, \
            self.gtargets = state

    def apply(self, updates):
        """
        Update problem object with covariance matrixes
        """

        if self._seismic_flag:
            for i, sw in enumerate(updates.sweights):
                A = sw.get_value()
                self.sweights[i].set_value(A)

        if self._geodetic_flag:
            for j, gw in enumerate(updates.gweights):
                B = gw.get_value()
                self.gweights[j].set_value(B)

    def built_model(self):
        """
        Initialise :class:`pymc3.Model` depending on configuration file,
        geodetic and/or seismic data are included. Estimates the fault(s)
        geometry.
        """

        logger.info('... Building model ...\n')

        self.outfolder = os.path.join(self.config.project_dir, 'geometry')
        util.ensuredir(self.outfolder)

        with pm.Model() as self.model:

            logger.debug('Optimization for %i sources', len(self.sources))

            pc = self.config.problem_config

            input_rvs = []
            for param in pc.priors:
                input_rvs.append(pm.Uniform(
                    param.name,
                    shape=param.dimension,
                    lower=param.lower,
                    upper=param.upper,
                    testval=param.testvalue,
                    transform=None))

            self.hyperparams = {}
            n_hyp = len(pc.hyperparameters.keys())

            for hyperpar in pc.hyperparameters.itervalues():
                if not self._seismic_flag and n_hyp == 1:
                    self.hyperparams[hyperpar.name] = 1.
                else:
                    self.hyperparams[hyperpar.name] = pm.Uniform(
                        hyperpar.name,
                        shape=hyperpar.dimension,
                        lower=hyperpar.lower,
                        upper=hyperpar.upper,
                        testval=hyperpar.testvalue,
                        transform=None)

            total_llk = tt.zeros((1), tconfig.floatX)

            if self._seismic_flag:
                self.seis_input_rvs = utility.weed_input_rvs(
                    input_rvs, dataset='seismic')
                # seis
                seis_names = [param.name for param in self.seis_input_rvs]
                logger.debug(
                    'Teleseismic optimization on: \n '
                    ' %s' % ', '.join(seis_names))

                t2 = time.time()
                synths, tmins = self.get_seis_synths(*self.seis_input_rvs)
                t3 = time.time()
                logger.debug(
                    'Teleseismic forward model on test model takes: %f' % \
                        (t3 - t2))

                data_trcs = self.chop_traces(tmins)

                seis_res = data_trcs - synths

                logpts_s = tt.zeros((self.ns_t), tconfig.floatX)
                M = seis_res.shape[1]

                for k, target in enumerate(self.stargets):
                    sfactor = target.covariance.log_norm_factor
                    hp_name = bconfig.hyper_pars[target.codes[3]]

                    logpts_s = tt.set_subtensor(logpts_s[k:k + 1],
                        (-0.5) * (sfactor - \
                        (M * 2 * self.hyperparams[hp_name]) + \
                        tt.exp(self.hyperparams[hp_name] * 2) * \
                        (seis_res[k, :].dot(
                            self.sweights[k]).dot(seis_res[k, :].T))
                                 )
                                                )

                seis_llk = pm.Deterministic(self._seis_like_name, logpts_s)

                total_llk = total_llk + seis_llk.sum()

            if self._geodetic_flag:
                self.geo_input_rvs = utility.weed_input_rvs(
                    input_rvs, dataset='geodetic')

                ## calc residuals
                # geo
                geo_names = [param.name for param in self.geo_input_rvs]
                logger.debug(
                    'Geodetic optimization on: \n '
                    '%s' % ', '.join(geo_names))

                t0 = time.time()
                disp = self.get_geo_synths(*self.geo_input_rvs)
                t1 = time.time()
                logger.debug(
                    'Geodetic forward model on test model takes: %f' % \
                        (t1 - t0))

                los = (disp[:, 0] * self.lv[:, 0] + \
                       disp[:, 1] * self.lv[:, 1] + \
                       disp[:, 2] * self.lv[:, 2]) * self.odws
                geo_res = self.Bij.srmap(
                    tt.cast((self.wdata - los), tconfig.floatX))

                logpts_g = tt.zeros((self.ng_t), tconfig.floatX)

                for l, target in enumerate(self.gtargets):
                    M = target.displacement.size
                    gfactor = target.covariance.log_norm_factor
                    hp_name = bconfig.hyper_pars[target.typ]

                    logpts_g = tt.set_subtensor(logpts_g[l:l + 1],
                         (-0.5) * (gfactor - \
                         (M * 2 * self.hyperparams[hp_name]) + \
                         tt.exp(self.hyperparams[hp_name] * 2) * \
                         (geo_res[l].dot(self.gweights[l]).dot(geo_res[l].T))
                                  )
                                               )

                geo_llk = pm.Deterministic(self._geo_like_name, logpts_g)

                total_llk = total_llk + geo_llk.sum()

            like = pm.Deterministic(
                self._like_name, total_llk)

            llk = pm.Potential(self._like_name, like)
            logger.info('Model building was successful!')

    def built_hyper_model(self):
        """
        Initialise :class:`pymc3.Model` depending on configuration file,
        geodetic and/or seismic data are included. Estimates initial parameter
        bounds for hyperparameters.
        """

        logger.info('... Building Hyper model ...\n')

        self.outfolder = os.path.join(
            self.config.project_dir, 'geometry', 'hypers')
        util.ensuredir(self.outfolder)

        pc = self.config.problem_config

        point = {}
        for param in pc.priors:
            point[param.name] = param.testvalue

        self.update_llks(point)

        with pm.Model() as self.model:
            self.hyperparams = {}
            n_hyp = len(pc.hyperparameters.keys())

            logger.debug('Optimization for %i hyperparemeters', n_hyp)

            for hyperpar in pc.hyperparameters.itervalues():
                if not self._seismic_flag and n_hyp == 1:
                    self.hyperparams[hyperpar.name] = 1.
                else:
                    self.hyperparams[hyperpar.name] = pm.Uniform(
                        hyperpar.name,
                        shape=hyperpar.dimension,
                        lower=hyperpar.lower,
                        upper=hyperpar.upper,
                        testval=hyperpar.testvalue,
                        transform=None)

            total_llk = tt.zeros((1), tconfig.floatX)

            if self._seismic_flag:

                logpts_s = tt.zeros((self.ns_t), tconfig.floatX)
                sc = self.config.seismic_config

                for k, target in enumerate(self.stargets):
                    M = sc.arrival_taper.duration * sc.gf_config.sample_rate
                    sfactor = target.covariance.log_norm_factor
                    hp_name = bconfig.hyper_pars[target.codes[3]]

                    logpts_s = tt.set_subtensor(logpts_s[k:k + 1],
                        (-0.5) * (sfactor - \
                        (M * 2 * self.hyperparams[hp_name]) + \
                        tt.exp(self.hyperparams[hp_name] * 2) * \
                            self._seis_llks[k]
                                 )
                                                )

                seis_llk = pm.Deterministic(self._seis_like_name, logpts_s)

                total_llk = total_llk + seis_llk.sum()

            if self._geodetic_flag:

                logpts_g = tt.zeros((self.ng_t), tconfig.floatX)

                for l, target in enumerate(self.gtargets):
                    M = target.displacement.size
                    gfactor = target.covariance.log_norm_factor
                    hp_name = bconfig.hyper_pars[target.typ]

                    logpts_g = tt.set_subtensor(logpts_g[l:l + 1],
                         (-0.5) * (gfactor - \
                         (M * 2 * self.hyperparams[hp_name]) + \
                         tt.exp(self.hyperparams[hp_name] * 2) * \
                         self._geo_llks[l]
                                  )
                                               )

                geo_llk = pm.Deterministic(self._geo_like_name, logpts_g)

                total_llk = total_llk + geo_llk.sum()

            like = pm.Deterministic(
                self._like_name, total_llk)

            llk = pm.Potential(self._like_name, like)
            logger.info('Hyper model building was successful!')

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Calculate and update model prediction uncertainty covariances
        due to uncertainty in the velocity model with respect to one point
        in the solution space. Shared variables are updated.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters, for which the covariance matrixes
            with respect to velocity model uncertainties are calculated
        n_jobs : int
            Number of processors to use for calculation of seismic covariances
        plot : boolean
            Flag for opening the seismic waveforms in the snuffler
        """

        # update sources
        point = utility.adjust_point_units(point)

        # remove hyperparameters from point
        hps = self.config.problem_config.hyperparameters

        if len(hps) > 0:
            for hyper in hps.keys():
                point.pop(hyper)

        if self._seismic_flag:
            point['time'] += self.event.time

        source_points = utility.split_point(point)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])

        dsources = utility.transform_sources(
            self.sources, self.config.problem_config.datasets)

        # seismic
        if self._seismic_flag:
            sc = self.config.seismic_config

            for j, channel in enumerate(sc.channels):
                for i, station in enumerate(self.stations):
                    logger.debug('Channel %s of Station %s ' % (
                        channel, station.station))
                    crust_targets = heart.init_targets(
                        stations=[station],
                        channels=channel,
                        sample_rate=sc.gf_config.sample_rate,
                        crust_inds=range(sc.gf_config.n_variations))

                    cov_pv = cov.get_seis_cov_velocity_models(
                        engine=self.engine,
                        sources=dsources['seismic'],
                        targets=crust_targets,
                        arrival_taper=sc.arrival_taper,
                        filterer=sc.filterer,
                        plot=plot, n_jobs=n_jobs)

                    cov_pv = utility.ensure_cov_psd(cov_pv)

                    self.engine.close_cashed_stores()

                    index = j * len(self.stations) + i

                    self.stargets[index].covariance.pred_v = cov_pv
                    icov = self.stargets[index].covariance.inverse
                    self.sweights[index].set_value(icov)

        # geodetic
        if self._geodetic_flag:
            gc = self.config.geodetic_config

            for i, gtarget in enumerate(self.gtargets):
                logger.debug('Track %s' % gtarget.track)
                cov_pv = cov.get_geo_cov_velocity_models(
                    store_superdir=gc.gf_config.store_superdir,
                    crust_inds=range(gc.gf_config.n_variations),
                    dataset=gtarget,
                    sources=dsources['geodetic'])

                cov_pv = utility.ensure_cov_psd(cov_pv)

                gtarget.covariance.pred_v = cov_pv
                icov = gtarget.covariance.inverse
                self.gweights[i].set_value(icov)

    def get_synthetics(self, point, **kwargs):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of seismic forward model
            outmode = 'traces'/ 'array' / 'data'

        Returns
        -------
        Dictionary with keys according to datasets containing the synthetics
        as lists.
        """
        tpoint = copy.deepcopy(point)

        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = self.config.problem_config.hyperparameters

        if len(hps) > 0:
            for hyper in hps.keys():
                if hyper in tpoint:
                    tpoint.pop(hyper)
                else:
                    pass

        d = dict()

        if self._seismic_flag:
            tpoint['time'] += self.event.time

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])

        dsources = utility.transform_sources(
            self.sources, self.config.problem_config.datasets)

        # seismic
        if self._seismic_flag:
            sc = self.config.seismic_config
            seis_synths, _ = heart.seis_synthetics(
                engine=self.engine,
                sources=dsources['seismic'],
                targets=self.stargets,
                arrival_taper=sc.arrival_taper,
                filterer=sc.filterer, **kwargs)

            d['seismic'] = seis_synths

        # geodetic
        if self._geodetic_flag:
            gc = self.config.geodetic_config

            crust_inds = [0]

            geo_synths = []
            for crust_ind in crust_inds:
                for gtarget in self.gtargets:
                    disp = heart.geo_layer_synthetics(
                        gc.gf_config.store_superdir,
                        crust_ind,
                        lons=gtarget.lons,
                        lats=gtarget.lats,
                        sources=dsources['geodetic'])
                    geo_synths.append((
                        disp[:, 0] * gtarget.los_vector[:, 0] + \
                        disp[:, 1] * gtarget.los_vector[:, 1] + \
                        disp[:, 2] * gtarget.los_vector[:, 2]))

            d['geodetic'] = geo_synths

        return d

    def assemble_seismic_results(self, point):
        """
        Assemble seismic traces for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters

        Returns
        -------
        List with :class:`heart.SeismicResult`
        """
        assert self._seismic_flag

        logger.debug('Assembling seismic waveforms ...')

        if self._geodetic_flag:
            self._geodetic_flag = False
            reset_flag = True
        else:
            reset_flag = False

        syn_proc_traces = self.get_synthetics(
            point, outmode='stacked_traces')['seismic']

        tmins = [tr.tmin for tr in syn_proc_traces]

        at = copy.deepcopy(self.config.seismic_config.arrival_taper)

        obs_proc_traces = heart.taper_filter_traces(
            self.data_traces,
            arrival_taper=at,
            filterer=self.config.seismic_config.filterer,
            tmins=tmins,
            outmode='traces')

        self.config.seismic_config.arrival_taper = None

        syn_filt_traces = self.get_synthetics(
            point, outmode='data')['seismic']

        obs_filt_traces = heart.taper_filter_traces(
            self.data_traces,
            filterer=self.config.seismic_config.filterer,
            outmode='traces')

        factor = 2.
        for i, (trs, tro) in enumerate(zip(syn_filt_traces, obs_filt_traces)):

            trs.chop(tmin=tmins[i] - factor * at.fade,
                     tmax=tmins[i] + factor * at.fade + at.duration)
            tro.chop(tmin=tmins[i] - factor * at.fade,
                     tmax=tmins[i] + factor * at.fade + at.duration)

        self.config.seismic_config.arrival_taper = at

        results = []
        for i, obstr in enumerate(obs_proc_traces):
            dtrace = obstr.copy()
            dtrace.set_ydata(
                (obstr.get_ydata() - syn_proc_traces[i].get_ydata()))

            taper = trace.CosTaper(
                tmins[i],
                tmins[i] + at.fade,
                tmins[i] + at.duration - at.fade,
                tmins[i] + at.duration)
            results.append(heart.SeismicResult(
                    processed_obs=obstr,
                    processed_syn=syn_proc_traces[i],
                    processed_res=dtrace,
                    filtered_obs=obs_filt_traces[i],
                    filtered_syn=syn_filt_traces[i],
                    taper=taper))

        if reset_flag:
            self._geodetic_flag = True

        return results

    def assemble_geodetic_results(self, point):
        """
        Assemble geodetic data for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters

        Returns
        -------
        List with :class:`heart.GeodeticResult`
        """
        assert self._geodetic_flag

        logger.debug('Assembling geodetic data ...')

        if self._seismic_flag:
            self._seismic_flag = False
            reset_flag = True
        else:
            reset_flag = False

        processed_synts = self.get_synthetics(point)['geodetic']

        results = []
        for i, target in enumerate(self.gtargets):
            res = target.displacement - processed_synts[i]

            results.append(heart.GeodeticResult(
                processed_obs=target.displacement,
                processed_syn=processed_synts[i],
                processed_res=res))

        if reset_flag:
            self._seismic_flag = True

        return results

    def update_llks(self, point):
        """
        Calculate likelihood with respect to given point in the solution space.
        """

        if self._seismic_flag:
            sresults = self.assemble_seismic_results(point)

            self._seis_llks = []
            for k, result in enumerate(sresults):
                icov = self.stargets[k].covariance.inverse
                self._seis_llks.append(shared(
                    result.processed_res.ydata.dot(
                        icov).dot(result.processed_res.ydata.T)))

        if self._geodetic_flag:
            gresults = self.assemble_geodetic_results(point)

            self._geo_llks = []
            for k, result in enumerate(gresults):
                icov = self.gtargets[k].covariance.inverse
                self._geo_llks.append(shared(
                    result.processed_res.dot(
                        icov).dot(result.processed_res.T)))


def sample(step, problem):
    """
    Sample solution space with the previously initalised algorithm.

    Parameters
    ----------

    step : :class:`ATMCMC` or :class:`pymc3.metropolis.Metropolis`
        from problem.init_sampler()
    problem : :class:`Problem` with characteristics of problem to solve
    """

    sc = problem.config.sampler_config
    pa = sc.parameters

    if pa.update_covariances:
        update = problem
    else:
        update = None

    if sc.name == 'Metropolis':
        logger.info('... Starting Metropolis ...\n')

        name = problem.outfolder
        util.ensuredir(name)

        pm.sample(
            draws=pa.n_steps,
            step=step,
            trace=pm.backends.Text(
                name=name,
                model=problem.model),
            model=problem.model,
            n_jobs=pa.n_jobs,
            update=update)

    elif sc.name == 'ATMCMC':
        logger.info('... Starting ATMIP ...\n')

        atmcmc.ATMIP_sample(
            pa.n_steps,
            step=step,
            progressbar=False,
            model=problem.model,
            n_jobs=pa.n_jobs,
            stage=pa.stage,
            update=update,
            trace=problem.outfolder,
            rm_flag=pa.rm_flag)


def estimate_hypers(step, problem):
    """
    Get initial estimates of the hyperparameters
    """
    logger.info('... Estimating hyperparameters ...')

    pc = problem.config.problem_config
    sc = problem.config.hyper_sampler_config
    pa = sc.parameters

    name = problem.outfolder
    util.ensuredir(name)

    mtraces = []
    for stage in range(pa.n_stages):
        logger.info('Metropolis stage %i' % stage)
        point = {param.name: param.random() for param in pc.priors}
        print point
        problem.outfolder = os.path.join(name, 'stage_%i' % stage)

        for g in problem._geo_llks:
            print g.get_value()

        if not os.path.exists(problem.outfolder):
            logger.debug('Sampling ...')
            problem.update_llks(point)
            logger
            with problem.model as model:
                mtraces.append(pm.sample(
                    draws=pa.n_steps,
                    step=step,
                    trace=pm.backends.Text(
                        name=problem.outfolder,
                        model=problem.model),
                    model=model,
                    chain=stage * pa.n_jobs,
                    njobs=pa.n_jobs,
                    ))

        else:
            logger.debug('Loading existing results!')
            mtraces.append(pm.backends.text.load(
                name=problem.outfolder, model=problem.model))

    mtrace = pm.backends.base.merge_traces(mtraces)
    outname = os.path.join(name, 'stage_final')

    if not os.path.exists(outname):
        util.ensuredir(outname)
        pm.backends.text.dump(name=outname, trace=mtrace)

    n_steps = pa.n_steps
    varnames = pc.hyperparameters.keys()

    def burn_sample(x):
        return x[(n_steps / 2):n_steps:2]

    for v in varnames:
        d = burn_sample(mtrace.get_values(v, combine=True, squeeze=True))
        lower = d.min(axis=0)
        upper = d.max(axis=0)
        pc.hyperparameters[v].lower = lower
        pc.hyperparameters[v].upper = upper
        pc.hyperparameters[v].testvalue = (upper + lower) / 2.

    config_file_name = 'config_' + pc.mode + '.yaml'
    conf_out = os.path.join(problem.config.project_dir, config_file_name)

    problem.config.problem_config = pc
    bconfig.dump(problem.config, filename=conf_out)


def choose_proposal(proposal_dist):
    """
    Initialises and selects proposal distribution.

    Parameters
    ----------
    proposal_dist : string
        Name of the proposal distribution to initialise

    Returns
    -------
    class:'pymc3.Proposal' Object
    """

    if proposal_dist == 'Cauchy':
        distribution = pm.CauchyProposal

    elif proposal_dist == 'Poisson':
        distribution = pm.PoissonProposal

    elif proposal_dist == 'Normal':
        distribution = pm.NormalProposal

    elif proposal_dist == 'Laplace':
        distribution = pm.LaplaceProposal

    elif proposal_dist == 'MultivariateNormal':
        distribution = pm.MultivariateNormalProposal

    return distribution


def load_model(project_dir, mode, hypers=False):
    """
    Load config from project directory and return BEAT problem including model.

    Parameters
    ----------
    project_dir : string
        path to beat model directory
    mode : string
        problem name to be loaded
    hypers : boolean
        flag to return hyper parameter estimation model instead of main model.

    Returns
    -------
    :class:`Problem`
    """

    config = bconfig.load_config(project_dir, mode)

    pc = config.problem_config

    if pc.mode == 'geometry':
        problem = GeometryOptimizer(config)
    else:
        logger.error('Modeling problem %s not supported' % pc.mode)
        raise Exception('Model not supported')

    if hypers:
        problem.built_hyper_model()
    else:
        problem.built_model()
    return problem


class ATMCMCStage(object):
    """
    ATMCMC stage, containing sampling results and intermediate optimizer
    parameters.
    """

    def __init__(self, number='final', path='./', step=None, updates=None,
                 mtrace=None):
        self.number = number
        self.path = path
        self.step = step
        self.updates = updates
        self.mtrace = mtrace


def load_stage(problem, stage_number=None, load='trace'):
    """
    Load stage results from sampling.

    Parameters
    ----------
    problem : :class:`Problem`
    stage_number : str
        Number of stage to load
    load : str
        what to load and return 'full', 'trace', 'params'

    Returns
    -------
    dict
    """

    project_dir = problem.config.project_dir
    mode = problem.config.problem_config.mode

    if stage_number is None:
        stage_number = 'final'

    homepath = problem.outfolder
    stagepath = os.path.join(homepath, 'stage_%s' % stage_number)

    if os.path.exists(stagepath):
        logger.info('Loading sampling results from: %s' % stagepath)
    else:
        stage_number = backend.get_highest_sampled_stage(
            homepath, return_final=True)

        if isinstance(stage_number, int):
            stage_number -= 1

        stage_number = str(stage_number)

        logger.info(
            'Stage results %s do not exist! Loading last completed'
            ' stage %s' % (stagepath, stage_number))
        stagepath = os.path.join(homepath, 'stage_%s' % stage_number)

    if load == 'full':
        to_load = ['params', 'trace']
    else:
        to_load = [load]

    stage = ATMCMCStage(path=stagepath, number=stage_number)

    if 'trace' in to_load:
        stage.mtrace = backend.load(stagepath, model=problem.model)

    if 'params' in to_load:
        stage.step, stage.updates = utility.load_atmip_params(
            project_dir, stage_number, mode)

    return stage
