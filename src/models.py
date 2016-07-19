import pymc3 as pm
import atmcmc
from heart import Project
from pyrocko import gf
import theano
from theano import config as tconfig
from theano import shared
import theanof
import utility
import time
import covariance as cov


class Project(Object):

    def sample(self, n_chains, n_steps, njobs, tune_interval):
        '''
        Sample solution space with the (C)ATMIP algorithm, where required input
        parameters are:
        n_chains - number of independent Metropolis chains
        n_steps - number of samples within each chain
        n_jobs - number of parallel chains
        tune_interval - number of samples after which Metropolis is being scaled
                        according to the acceptance ratio.
        '''
        with self.model:
            print 'Initiate Adaptive Transitional Metropolis ... '
            t1 = time.time()
            step = atmcmc.ATMCMC(
                    n_chains=n_chains,
                    tune_interval=tune_interval,
                    likelihood_name=self.model.deterministics[0].name)
            t2 = time.time()
            print 'Compilation time:', t2 - t1

        print 'Starting ATMIP ...'
        trace = atmcmc.ATMIP_sample(self, n_steps,
                    step=step,
                    progressbar=True,
                    njobs=njobs,
                    trace=self.config.geometry_outfolder)
            return trace
            

class GeometryOptimizer(Project):
    '''
    Defines the model setup to solve the non-linear fault geometry and 
    returns the model object.

    Input: :py:class: 'BEATconfig'
    Output: :py:class: 'Model'
    '''
    def __init__(self, config):

        
        self.config = config
        self.engine = gf.LocalEngine(store_superdirs=[config.store_superdir])

        # sources
        self.sources = [config.main_source] + config.sub_sources
        s_sources = []
        g_sources = []
        for source in self.sources:
            s_sources.append(source.patches(1, 1, 'seis')
            g_sources.append(source.patches(1, 1, 'geo')

        # targets
        self.ns_t = len(config.stargets)
        self.ng_t = len(config.gtargets)

        # geodetic data
        ordering = utility.ArrayOrdering(config.gtargets)
        
        disp_list = [config.gtargets[i].displacement for i in range(self.ng_t)]
        lons_list = [config.gtargets[i].lons for i in range(self.ng_t)]
        lats_list = [config.gtargets[i].lats for i in range(self.ng_t)]
        odws_list = [config.gtargets[i].odw for i in range(self.ng_t)]
        
        self.gweights = [shared(
            config.gtargets[i].covariance.icov) for i in range(self.ng_t)]
        self.sweights = [shared(
            config.stargets[i].covariance.icov) for i in range(self.ns_t)]
        
        self.lv = num.vstack(
                    [config.gtargets[i].look_vector() for i in range(ng_t)])
        
        self.Bij = utility.ListToArrayBijection(ordering, disp_list)

        self.odws = Bij.fmap(odws_list)
        self.wdata = Bij.fmap(disp_list) * odws
        self.lons = Bij.fmap(lons_list)
        self.lats = Bij.fmap(lats_list)

        # seismic data
        data_traces =

        # syntetics generation
        self.get_geo_synths = theanof.GeoLayerSynthesizer(
                            superdir=config.store_superdir,
                            crust_ind=0,    # always use reference model
                            sources=g_sources)
        self.chop_traces = theanof.SeisDataChopper(
                            sample_rate=config.sample_rate,
                            traces=data_traces,
                            arrival_taper=config.arrival_taper,
                            filterer=config.filterer)
        self.get_seis_synths = theanof.SeisSynthesizer(
                            engine=self.engine,
                            sources=s_sources,
                            targets=config.stargets,
                            event=config.event,
                            arrival_taper=config.arrival_taper,
                            filterer=config.filterer)

    def built_model(self):
        with pm.Model() as self.model:
            # instanciate random vars
            input_rvs = []
            for param in self.config.bounds:
                input_rvs.append(pm.Uniform(param.name,
                                       shape=len(param.dimension),
                                       lower=param.lower,
                                       upper=param.upper,
                                       testval=param.testvalue,
                                       transform=None))

            geo_input_rvs = copy.deepcopy(input_rvs)
            if 'time' in input_rvs:
                geo_input_rvs.pop('time')

            seis_input_rvs = copy.deepcopy(input_rvs)
            if 'opening' in input_rvs:
                seis_input_rvs.pop('opening')

            # calc residuals
            disp = self.get_geo_synths(self.lons, self.lats, *geo_input_rvs)
            los = (disp[:,0]*lv[:,0] + \
                   disp[:,1]*lv[:, 1] + \
                   disp[:,2]*lv[:,2]) * self.odws
                   
            geo_res = self.Bij.srmap((self.wdata - los))

            synths, tmins = self.get_seis_synths(*seis_input_rvs)
            data_trcs = self.chop_traces(tmins)
            seis_res = data_trcs - synths

            # calc likelihoods
            logpts_g = tt.zeros((ng_t), tconfig.floatX)
            logpts_s = tt.zeros((ns_t), tconfig.floatX)

            for k in range(self.ns_t)
                logpts_s = tt.set_subtensor(logpts_s[k:k + 1],
                        (-0.5) * seis_res[k, :].dot(
                              shared(self.sweights[k])).dot(seis_res[k, :].T))

            for l in range(self.ng_t)
                logpts_g = tt.set_subtensor(logpts_g[l:l + 1],
                        (-0.5) * geo_res[l, :].dot(
                              shared(self.gweights[l])).dot(geo_res[l, :].T))

            # adding dataset missfits to traces
            seis_llk = pm.Deterministic('seis_like', logpts_s)
            geo_llk = pm.Deterministic('geo_like', logpts_g)
                           
            # sum up geodetic and seismic likelihood
            like = pm.Deterministic('like',
                    seis_llk.T.dot(seis_mf_cov).dot(seis_llk) + \
                    geo_llk.T.dot(geo_mf_cov).dot(geo_llk))
            llk = pm.Potential('llk', like)

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
                          
            s_sources = []
            g_sources = []
            for source in self.sources:
                s_sources.append(source.patches(1, 1, 'seis')
                g_sources.append(source.patches(1, 1, 'geo')

            # seismic
            for channel in self.config.channels:
                for i, station in enumerate(self.config.stations):
                    crust_targets = heart.init_targets(
                                  stations=[station],
                                  channels=channel,
                                  sample_rate=self.config.sample_rate,
                                  crust_inds=self.config.crust_inds)
                            
                    self.stargets[i].covariance.pred_v = \
                        cov.calc_seis_cov_velocity_models(
                                 engine=self.engine,
                                 sources=s_sources,
                                 crust_inds=self.config.crust_inds,
                                 targets=crust_targets,
                                 sample_rate=self.config.sample_rate,
                                 arrival_taper=self.config.arrival_taper,
                                 corner_fs=self.config.corner_fs)
        
                    self.sweights[i].set_value(
                        self.stargets[i].covariance.inverse())

            # geodetic
            for i, gtarget in enumerate(self.config.gtargets):
                gtarget.covariance.pred_v = cov.calc_geo_cov_velocity_models(
                         store_superdir=self.config.store_superdir,
                         crust_inds=self.config.crust_inds,
                         dataset=gtarget,
                         sources=g_sources)
                         
                self.gweights[i].set_value(gtarget.covariance.inverse())
            
