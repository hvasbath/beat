import os
import time

import pymc3 as pm

from pymc3 import Metropolis

from pyrocko import gf, util, model
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


class Problem(Object):

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

    def init_sampler(self):
        '''
        Initialise the Sampling algorithm as defined in the configuration file.
        '''
        sc = self.config.sampler_config

        if self.model is None:
            raise Exception('Model has to be built before initialising the sampler.')

        with self.model:
            if sc.name == 'Metropolis':
                logger.info(
                    '... Initiate Metropolis ... \n'
                    'proposal_distribution %s, tune_interval=%i\n' % (
                    sc.parameters.proposal_dist, sc.parameters.tune_interval))

                t1 = time.time()
                step = Metropolis(
                    n_steps=sc.parameters.n_steps,
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
                    data_weighting=sc.parameters.data_weighting,
                    likelihood_name=self._like_name)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

        if self._seismic_flag:
            self.engine.close_cashed_stores()

        return step


class GeometryOptimizer(Problem):
    '''
    Defines the model setup to solve the non-linear fault geometry and
    returns the model object.

    Input: :py:class: 'BEATconfig'
    '''
    def __init__(self, config):
        logger.info('... Initialising Geometry Optimizer ... \n')

        self.outfolder = os.path.join(config.project_dir, 'geometry')
        util.ensuredir(self.outfolder)

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
            logger.info('Setting up seismic structure ...\n')
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

            logger.info('Getting seismic data-covariances ...\n')
            cov_ds_seismic = cov.get_seismic_data_covariances(
                data_traces=self.data_traces,
                filterer=sc.filterer,
                sample_rate=sc.gf_config.sample_rate,
                arrival_taper=sc.arrival_taper,
                engine=self.engine,
                event=self.event,
                targets=self.stargets)

            self.sweights = []
            for s_t in range(self.ns_t):
                self.stargets[s_t].covariance.data = cov_ds_seismic[s_t]
                icov = self.stargets[s_t].covariance.get_inverse()
                self.sweights.append(shared(icov))

            # syntetics generation
            logger.info('Initialising synthetics functions ... \n')
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
            logger.info('Setting up geodetic structure ...\n')
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
                icov = self.gtargets[g_t].covariance.get_inverse()
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
            logger.info('Initialising synthetics functions ... \n')
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
        if self._seismic_flag and self._geodetic_flag:
            self.config, self.sources,
            self.sweights,
            self.stargets,
            self.stations,
            self.engine,
            self.gweights,
            self.gtargets = state

        elif self._seismic_flag and not self._geodetic_flag:
            self.config, self.sources,
            self.sweights,
            self.stargets,
            self.stations,
            self.engine = state

        if not self._seismic_flag and self._geodetic_flag:
            self.config, self.sources,
            self.gweights,
            self.gtargets = state

    def built_model(self):
        logger.info('... Building model ...\n')

        with pm.Model() as self.model:

            logger.info('Optimization for %i sources', len(self.sources))

            input_rvs = []
            for param in self.config.problem_config.priors:
                input_rvs.append(pm.Uniform(param.name,
                                       shape=param.dimension,
                                       lower=param.lower,
                                       upper=param.upper,
                                       testval=param.testvalue,
                                       transform=None))

            total_llk = tt.zeros((1), tconfig.floatX)

            if self._seismic_flag:
                self.seis_input_rvs = utility.weed_input_rvs(
                    input_rvs, dataset='seismic')
                # seis
                seis_names = [param.name for param in self.seis_input_rvs]
                logger.info(
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

                for k in range(self.ns_t):
                    ssz = seis_res[k, :].shape[0]
                    sfactor = ssz * tt.log(2 * num.pi) + \
                                  self.stargets[k].covariance.log_determinant
                    logpts_s = tt.set_subtensor(logpts_s[k:k + 1],
                        (-0.5) * (sfactor + seis_res[k, :].dot(
                              self.sweights[k]).dot(seis_res[k, :].T)))

                seis_llk = pm.Deterministic(self._seis_like_name, logpts_s)

                total_llk = total_llk + seis_llk.sum()

            if self._geodetic_flag:
                self.geo_input_rvs = utility.weed_input_rvs(
                    input_rvs, dataset='geodetic')

                ## calc residuals
                # geo
                geo_names = [param.name for param in self.geo_input_rvs]
                logger.info(
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

                for l in range(self.ng_t):
                    gfactor = self.gtargets[l].covariance.log_norm_factor

                    logpts_g = tt.set_subtensor(logpts_g[l:l + 1],
                         (-0.5) * (gfactor + geo_res[l].dot(
                              self.gweights[l]).dot(geo_res[l].T)))

                geo_llk = pm.Deterministic(self._geo_like_name, logpts_g)

                total_llk = total_llk + geo_llk.sum()

            like = pm.Deterministic(
                self._like_name, total_llk)

            llk = pm.Potential(self._like_name, like)
            logger.info('Model building was successful!')

    def update_weights(self, point, n_jobs=1, plot=False):
        '''
        Calculate and update model prediction uncertainty covariances
        due to uncertainty in the velocity model with respect to one point
        in the solution space.
        Input: Point dictionary from pymc3
        '''
        # update sources
        point = utility.adjust_point_units(point)

        if self._seismic_flag:
            point['time'] += self.event.time

        source_points = utility.split_point(point)

        for i, source in enumerate(self.sources):
            source.update(**source_points[i])

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
                        crust_inds=sc.gf_config.crust_inds)

                    cov_velocity_model = cov.get_seis_cov_velocity_models(
                        engine=self.engine,
                        sources=dsources['seismic'],
                        targets=crust_targets,
                        arrival_taper=sc.arrival_taper,
                        filterer=sc.filterer,
                        plot=plot, n_jobs=n_jobs)

                self.engine.close_cashed_stores()

                index = j * len(self.stations) + i

                self.stargets[index].covariance.pred_v = cov_velocity_model
                icov = self.stargets[index].covariance.get_inverse()
                self.sweights[index].set_value(icov)

        # geodetic
        if self._geodetic_flag:
            gc = self.config.geodetic_config

            for i, gtarget in enumerate(self.gtargets):
                logger.debug('Track %s' % gtarget.track)
                gtarget.covariance.pred_v = cov.get_geo_cov_velocity_models(
                    store_superdir=gc.gf_config.store_superdir,
                    crust_inds=gc.gf_config.crust_inds,
                    dataset=gtarget,
                    sources=dsources['geodetic'])

            icov = gtarget.covariance.get_inverse()
            self.gweights[i].set_value(icov)

    def get_synthetics(self, point):
        '''
        Get synthetics for given point in solution space.
        '''
        point = utility.adjust_point_units(point)

        d = dict()

        if self._seismic_flag:
            point['time'] += self.event.time

        source_points = utility.split_point(point)

        for i, source in enumerate(self.sources):
            source.update(**source_points[i])

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
                filterer=sc.filterer, outmode='traces')

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


def sample(step, problem):
    '''
    Sample solution space with the previously initalised algorithm.

    Inputs:
    step - Object from init_sampler
    problem - Object with characteristics of problem to solve
    '''

    sc = problem.config.sampler_config.parameters

    if sc.update_covariances:
        update = problem
    else:
        update = None

    logger.info('... Starting ATMIP ...\n')
    atmcmc.ATMIP_sample(
        sc.n_steps,
        step=step,
        progressbar=True,
        model=problem.model,
        n_jobs=sc.n_jobs,
        stage=sc.stage,
        update=update,
        trace=problem.outfolder,
        rm_flag=sc.rm_flag,
        plot_flag=sc.plot_flag)


def choose_proposal(proposal_dist):
    '''
    Initialises and selects proposal distribution.
    Returns:
    Function
    '''

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


def load_model(project_dir, mode):
    '''
    Load config from project directory and return model.
    '''
    config = bconfig.load_config(project_dir, mode)

    pc = config.problem_config

    if pc.mode == 'geometry':
        problem = GeometryOptimizer(config)
    else:
        logger.error('Modeling problem %s not supported' % pc.mode)
        raise Exception('Model not supported')

    problem.built_model()
    return problem


def load_stage(project_dir, stage_number, mode):
    '''
    Load stage results from ATMIP sampling.
    '''

    problem = load_model(project_dir, mode)
    params = utility.load_atmip_params(project_dir, stage_number, mode)
    tracepath = os.path.join(project_dir, mode, 'stage_%i' % stage_number)
    mtrace = backend.load(tracepath, model=problem.model)
    return problem, params, mtrace
