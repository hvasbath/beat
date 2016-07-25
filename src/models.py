import os
import time

import pymc3 as pm

from pyrocko import gf, util, model
from pyrocko.guts import Object

import numpy as num
import theano.tensor as tt
from theano import config as tconfig
from theano import shared

from beat import theanof, heart, utility, atmcmc, inputf
from beat import covariance as cov

import logging

logger = logging.getLogger('beat')


class Project(Object):

    step = None
    model = None
    _seis_like_name = 'seis_like'
    _geo_like_name = 'geo_like'
    _like_name = 'like'

    def update_target_weights(self, mtrace, stage, n_steps):
        '''
        Update target weights based on distribution of misfits per target.
        Input: MultiTrace Object.
        '''
        if stage > 0:
            # get target likelihoods
            seis_likelihoods = mtrace.get_values(
                varname=self._seis_like_name,
                burn=n_steps - 1,
                combine=True)
            geo_likelihoods = mtrace.get_values(
                varname=self._geo_like_name,
                burn=n_steps - 1,
                combine=True)
        else:
            # for initial stage only one trace that contains all points for
            # each chain
            seis_likelihoods = mtrace.get_values(self._seis_like_name)
            geo_likelihoods = mtrace.get_values(self._geo_like_name)

        seis_target_cov = num.cov(seis_likelihoods, bias=False, rowvar=0)
        geo_target_cov = num.cov(geo_likelihoods, bias=False, rowvar=0)

        self.seis_misfit_icov.set_value(num.linalg.inv(seis_target_cov))
        self.geo_misfit_icov.set_value(num.linalg.inv(geo_target_cov))

    def init_atmip(self, n_chains=100, tune_interval=10):
        '''
        Initialise the (C)ATMIP algorithm.

        n_chains - number of independent Metropolis chains
        tune_interval - number of samples after which Metropolis is being
                        scaled according to the acceptance ratio.
        '''
        with self.model:
            logger.info('... Initiate Adaptive Transitional Metropolis ... \n'
                  ' n_chains=%i, tune_interval=%i\n' % (n_chains,
                                                           tune_interval))
            t1 = time.time()
            self.step = atmcmc.ATMCMC(
                n_chains=n_chains,
                tune_interval=tune_interval,
                likelihood_name=self._like_name)
            t2 = time.time()
            logger.info('Compilation time: %f' % (t2 - t1))

    def sample(self, n_steps=100, njobs=1):
        '''
        Sample solution space with the (C)ATMIP algorithm.

        Inputs:
        n_steps - number of samples within each chain
        n_jobs - number of parallel chains
        '''

        if not self.step:
            raise Exception('Sampler needs to be initialised first!'
                            'with: "init_atmip" ')

        logger.info('... Starting ATMIP ...\n')
        trace = atmcmc.ATMIP_sample(
            n_steps,
            step=self.step,
            progressbar=True,
            model=self.model,
            njobs=njobs,
            update=self,
            trace=self.geometry_outfolder)
        return trace


class GeometryOptimizer(Project):
    '''
    Defines the model setup to solve the non-linear fault geometry and
    returns the model object.

    Input: :py:class: 'BEATconfig'
    '''
    def __init__(self, config):
        logger.info('... Initialising Geometry Optimizer ... \n')

        self.geometry_outfolder = os.path.join(config.project_dir, 'geometry')
        util.ensuredir(self.geometry_outfolder)

        self.engine = gf.LocalEngine(store_superdirs=[config.store_superdir])

        # load data still not general enopugh
        logger.info('Loading waveforms ...\n')

        self.event = model.load_one_event(config.seismic_datadir + 'event.txt')

        stations = inputf.load_and_blacklist_stations(
            config.seismic_datadir, blacklist=config.blacklist)

        self.stations = utility.weed_stations(
            stations, self.event, distances=config.distances)

        self.data_traces = inputf.load_data_traces(
            datadir=config.seismic_datadir,
            stations=self.stations,
            channels=config.channels)

        target_deltat = 1. / config.sample_rate

        if self.data_traces[0].deltat != target_deltat:
            utility.downsample_traces(self.data_traces, deltat=target_deltat)

        self.stargets = heart.init_targets(
            self.stations,
            channels=config.channels,
            sample_rate=config.sample_rate,
            crust_inds=[0],  # always reference model
            interpolation='multilinear')

        logger.info('Loading SAR data ...\n')
        self.gtargets = inputf.load_SAR_data(
                config.geodetic_datadir, config.tracks)

        # Init sources
        self.sources = []

        for i in range(config.bounds[0].dimension):
            source = heart.RectangularSource.from_pyrocko_event(self.event)
            source.stf.anchor = -1.  # hardcoded inversion for hypocentral time
            self.sources.append(source)

        seismic_sources, geodetic_sources = utility.transform_sources(
                                                                self.sources)

        # targets
        self.ns_t = len(self.stargets)
        self.ng_t = len(self.gtargets)

        # geodetic data
        ordering = utility.ArrayOrdering(self.gtargets)

        disp_list = [self.gtargets[i].displacement for i in range(self.ng_t)]
        lons_list = [self.gtargets[i].lons for i in range(self.ng_t)]
        lats_list = [self.gtargets[i].lats for i in range(self.ng_t)]
        odws_list = [self.gtargets[i].odw for i in range(self.ng_t)]
        lv_list = [self.gtargets[i].look_vector() for i in range(self.ng_t)]

        ## Data and model covariances
        logger.info('Getting data-covariances ...\n')
        cov_ds_seismic = cov.get_seismic_data_covariances(
            data_traces=self.data_traces,
            config=config,
            engine=self.engine,
            event=self.event,
            targets=self.stargets)

        self.gweights = []
        for g_t in range(self.ng_t):
            self.gtargets[g_t].covariance.set_inverse()
            self.gweights.append(shared(self.gtargets[g_t].covariance.icov))

        self.sweights = []
        for s_t in range(self.ns_t):
            self.stargets[s_t].covariance.data = cov_ds_seismic[s_t]
            self.stargets[s_t].covariance.set_inverse()
            self.sweights.append(shared(self.stargets[s_t].covariance.icov))

        # Target weights, initially identity matrix = equal weights
        self.seis_misfit_icov = shared(num.eye(self.ns_t))
        self.geo_misfit_icov = shared(num.eye(self.ng_t))

        # merge geodetic data to call pscmp only once each forward model
        self.Bij = utility.ListToArrayBijection(ordering, disp_list)

        odws = self.Bij.fmap(odws_list)
        lons = self.Bij.fmap(lons_list)
        lats = self.Bij.fmap(lats_list)

        self.wdata = shared(self.Bij.fmap(disp_list) * odws)
        self.lv = shared(self.Bij.f3map(lv_list))
        self.odws = shared(odws)

        # syntetics generation
        logger.info('Initialising theano synthetics functions ... \n')
        self.get_geo_synths = theanof.GeoLayerSynthesizerStatic(
                            lats=lats,
                            lons=lons,
                            store_superdir=config.store_superdir,
                            crust_ind=0,    # always reference model
                            sources=geodetic_sources)

        self.get_seis_synths = theanof.SeisSynthesizer(
                            engine=self.engine,
                            sources=seismic_sources,
                            targets=self.stargets,
                            event=self.event,
                            arrival_taper=config.arrival_taper,
                            filterer=config.filterer)

        self.chop_traces = theanof.SeisDataChopper(
                            sample_rate=config.sample_rate,
                            traces=self.data_traces,
                            arrival_taper=config.arrival_taper,
                            filterer=config.filterer)

        self.config = config

    def built_model(self):
        logger.info('... Building model ...\n')

        with pm.Model() as self.model:
            logger.info('Optimization for %i sources', len(self.sources))
            ## instanciate random vars
            input_rvs = []
            for param in self.config.bounds:
                input_rvs.append(pm.Uniform(param.name,
                                       shape=param.dimension,
                                       lower=param.lower,
                                       upper=param.upper,
                                       testval=param.testvalue,
                                       transform=None))

            self.geo_input_rvs = utility.weed_input_rvs(input_rvs, mode='geo')
            self.seis_input_rvs = utility.weed_input_rvs(input_rvs, mode='seis')

            ## calc residuals
            # geo
            geo_names = [param.name for param in self.geo_input_rvs]
            logger.info(
            'Geodetic optimization on: \n \n %s' % ', '.join(geo_names))

            t0 = time.time()
            disp = self.get_geo_synths(*self.geo_input_rvs)
            t1 = time.time()
            logger.info('Geodetic forward model takes: %f' % (t1 - t0))

            los = (disp[:, 0] * self.lv[:, 0] + \
                   disp[:, 1] * self.lv[:, 1] + \
                   disp[:, 2] * self.lv[:, 2]) * self.odws
            geo_res = self.Bij.srmap(
                tt.cast((self.wdata - los), tconfig.floatX))

            # seis
            seis_names = [param.name for param in self.seis_input_rvs]
            logger.info(
            'Teleseismic optimization on: \n \n %s' % ', '.join(seis_names))

            t2 = time.time()
            synths, tmins = self.get_seis_synths(*self.seis_input_rvs)
            t3 = time.time()
            logger.info('Teleseismic forward model takes: %f' % (t3 - t2))

            data_trcs = self.chop_traces(tmins)

            seis_res = data_trcs - synths

            ## calc likelihoods
            logpts_g = tt.zeros((self.ng_t), tconfig.floatX)
            logpts_s = tt.zeros((self.ns_t), tconfig.floatX)

            for k in range(self.ns_t):
                logpts_s = tt.set_subtensor(logpts_s[k:k + 1],
                    (-0.5) * seis_res[k, :].dot(
                          self.sweights[k]).dot(seis_res[k, :].T))

            for l in range(self.ng_t):
                logpts_g = tt.set_subtensor(logpts_g[l:l + 1],
                    (-0.5) * geo_res[l].dot(
                          self.gweights[l]).dot(geo_res[l].T))

            # adding dataset missfits to traces
            seis_llk = pm.Deterministic(self._seis_like_name, logpts_s)
            geo_llk = pm.Deterministic(self._geo_like_name, logpts_g)

            # sum up geodetic and seismic likelihood
            like = pm.Deterministic(
                self._like_name,
                seis_llk.T.dot(self.seis_misfit_icov).sum() + \
                geo_llk.T.dot(self.geo_misfit_icov).sum())
            llk = pm.Potential(self._like_name, like)
            logger.info('Model building was successful!')

        def update_weights(self, point):
            '''
            Calculate and update model prediction uncertainty covariances
            due to uncertainty in the velocity model.
            Input: Point dictionary from pymc3
            '''
            # update sources
            for s, source in enumerate(self.sources):
                for param, value in point.iteritems():
                    source.update(param=value[s])

            seismic_sources, geodetic_sources = utility.transform_sources(
                                                                self.sources)

            # seismic
            for channel in self.config.channels:
                for i, station in enumerate(self.stations):
                    crust_targets = heart.init_targets(
                                  stations=[station],
                                  channels=channel,
                                  sample_rate=self.config.sample_rate,
                                  crust_inds=self.config.crust_inds)

                    self.stargets[i].covariance.pred_v = \
                        cov.get_seis_cov_velocity_models(
                                 engine=self.engine,
                                 sources=seismic_sources,
                                 crust_inds=self.config.crust_inds,
                                 targets=crust_targets,
                                 sample_rate=self.config.sample_rate,
                                 arrival_taper=self.config.arrival_taper,
                                 corner_fs=self.config.corner_fs)

                    self.sweights[i].set_value(
                        self.stargets[i].covariance.inverse())

            # geodetic
            for i, gtarget in enumerate(self.gtargets):
                gtarget.covariance.pred_v = cov.get_geo_cov_velocity_models(
                         store_superdir=self.config.store_superdir,
                         crust_inds=self.config.crust_inds,
                         dataset=gtarget,
                         sources=geodetic_sources)

                self.gweights[i].set_value(gtarget.covariance.inverse())
