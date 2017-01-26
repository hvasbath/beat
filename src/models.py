import os
import time
import copy
import shutil

import pymc3 as pm
from pymc3 import Metropolis

from pyrocko import gf, util, trace
from pyrocko.guts import Object

import numpy as num

import theano.tensor as tt
from theano import config as tconfig
from theano import shared

from beat import theanof, heart, utility, atmcmc, backend, metropolis
from beat import covariance as cov
from beat import config as bconfig

import logging

logger = logging.getLogger('models')

__all__ = ['GeometryOptimizer', 'sample', 'load_model', 'load_stage']


def multivariate_normal(targets, weights, hyperparams, residuals):
    """
    Calculate posterior Likelihood of a Multivariate Normal distribution.
    Can only be executed in a `with model context`

    Parameters
    ----------
    targets : list
        of :class:`heart.TeleseismicTarget` or :class:`heart.GeodeticTarget`
    weights : list
        of :class:`theano.shared`
        Square matrix of the inverse of the covariance matrix as weights
    hyperparams : dict
        of :class:`theano.`
    residual : list or array of model residuals

    Returns
    -------
    array_like
    """
    n_t = len(targets)

    logpts = tt.zeros((n_t), tconfig.floatX)

    for l, target in enumerate(targets):
        M = tt.cast(shared(target.samples, borrow=True), 'int16')
        factor = tt.cast(shared(
            target.covariance.log_norm_factor, borrow=True), tconfig.floatX)
        hp_name = bconfig.hyper_pars[target.typ]

        logpts = tt.set_subtensor(logpts[l:l + 1],
            (-0.5) * (factor - \
            (M * 2 * hyperparams[hp_name]) + \
            tt.exp(hyperparams[hp_name] * 2) * \
            (residuals[l].dot(weights[l]).dot(residuals[l].T))
                     )
                                 )

    return logpts


def hyper_normal(targets, hyperparams, llks):
    """
    Calculate posterior Likelihood only dependent on hyperparameters
    """
    n_t = len(targets)

    logpts = tt.zeros((n_t), tconfig.floatX)

    for k, target in enumerate(targets):
        M = target.samples
        factor = target.covariance.log_norm_factor
        hp_name = bconfig.hyper_pars[target.typ]

        logpts = tt.set_subtensor(logpts[k:k + 1],
            (-0.5) * (factor - \
            (M * 2 * hyperparams[hp_name]) + \
            tt.exp(hyperparams[hp_name] * 2) * \
                llks[k]
                     )
                                 )

    return logpts


class Composite(Object):
    """
    Class that comprises the rules to formulate the problem. Has to be
    used by an overarching problem object.

    Parameters
    ----------
    hypers : boolean
        determines whether to initialise Composites with hyper parameter model
    """

    name = None
    _like_name = None
    config = None
    weights = None

    def __init__(self, hypers=False):

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(shared(num.array([1.])))

    def get_hyper_formula(self, hyperparams):
        """
        Get likelihood formula for the hyper model built. Has to be called
        within a with model context.
        """
        logpts = hyper_normal(self.targets, hyperparams, self._llks)
        llk = pm.Deterministic(self._like_name, logpts)
        return llk.sum()

    def apply(self, composite):
        """
        Update composite weight matrixes (in place) with weights in given
        composite.

        Parameters
        ----------
        composite : :class:`Composite`
            containing weight matrixes to use for updates
        """

        for i, weight in enumerate(composite.weights):
            A = weight.get_value()
            self.weights[i].set_value(A)


class GeodeticComposite(Composite):
    """
    Comprises data structure of the geodetic composite.

    Parameters
    ----------
    gc : :class:`config.GeodeticConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, gc, project_dir, hypers=False):

        logger.debug('Setting up geodetic structure ...\n')
        self.name = 'geodetic'
        self._like_name = 'geo_like'

        geodetic_data_path = os.path.join(
            project_dir, bconfig.geodetic_data_name)
        self.targets = utility.load_objects(geodetic_data_path)

        self.n_t = len(self.targets)
        logger.info('Number of geodetic datasets: %i ' % self.n_t)

        if gc.calc_data_cov:
            logger.info('Using data covariance!')
        else:
            logger.info('No data-covariance estimation ...\n')
            for t in self.targets:
                t.covariance.data = num.zeros(t.lats.size)
                t.covariance.pred_v = num.eye(t.lats.size)

        self.weights = []
        for target in self.targets:
            icov = target.covariance.inverse
            self.weights.append(shared(icov))

        self.config = gc

        super(GeodeticComposite, self).__init__(hypers=hypers)

    def assemble_results(self, point):
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

        logger.debug('Assembling geodetic data ...')

        processed_synts = self.get_synthetics(point)

        results = []
        for i, target in enumerate(self.targets):
            res = target.displacement - processed_synts[i]

            results.append(heart.GeodeticResult(
                processed_obs=target.displacement,
                processed_syn=processed_synts[i],
                processed_res=res))

        return results

    def update_llks(self, point):
        """
        Update posterior likelihoods (in place) of the composite w.r.t.
        one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        results = self.assemble_results(point)
        for l, result in enumerate(results):
            icov = self.targets[l].covariance.inverse
            llk = num.array(result.processed_res.dot(
                icov).dot(result.processed_res.T)).flatten()
            self._llks[l].set_value(llk)


class GeodeticGeometryComposite(GeodeticComposite):
    """
    Comprises how to solve the non-linear geodetic forward model.

    Parameters
    ----------
    gc : :class:`config.GeodeticConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    sources : list
        of :class:`pyrocko.gf.seismosizer.Source`
    event : :class:`pyrocko.model.Event`
        contains information of reference event, coordinates of reference
        point and source time
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, gc, project_dir, sources, event, hypers=False):

        super(GeodeticGeometryComposite, self).__init__(
            gc, project_dir, hypers=hypers)

        self.event = event
        self.sources = sources

        _disp_list = [self.targets[i].displacement for i in range(self.n_t)]
        _lons_list = [self.targets[i].lons for i in range(self.n_t)]
        _lats_list = [self.targets[i].lats for i in range(self.n_t)]
        _odws_list = [self.targets[i].odw for i in range(self.n_t)]
        _lv_list = [self.targets[i].update_los_vector()
                        for i in range(self.n_t)]

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
        self.get_synths = theanof.GeoLayerSynthesizerStatic(
            lats=lats,
            lons=lons,
            store_superdir=gc.gf_config.store_superdir,
            crust_ind=0,    # always reference model
            sources=sources)

    def __getstate__(self):
        outstate = (
            self.config,
            self.sources,
            self.weights,
            self.targets)

        return outstate

    def __setstate__(self, state):
            self.config, \
            self.sources, \
            self.weights, \
            self.targets = state

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.
        """
        tpoint = copy.deepcopy(point)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = bconfig.hyper_pars.values()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source = self.sources[0]
        source_params = source.keys()

        for param in tpoint.keys():
            if param not in source_params:
                tpoint.pop(param)

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            heart.adjust_fault_reference(source, input_depth='top')

    def get_formula(self, input_rvs, hyperparams):
        """
        Get geodetic likelihood formula for the model built. Has to be called
        within a with model context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        self.input_rvs = input_rvs

        logger.debug(
            'Geodetic optimization on: \n '
            '%s' % ', '.join(self.input_rvs.keys()))

        t0 = time.time()
        disp = self.get_synths(self.input_rvs)
        t1 = time.time()
        logger.debug(
            'Geodetic forward model on test model takes: %f' % \
                (t1 - t0))

        los = (disp[:, 0] * self.lv[:, 0] + \
               disp[:, 1] * self.lv[:, 1] + \
               disp[:, 2] * self.lv[:, 2]) * self.odws
        residuals = self.Bij.srmap(
            tt.cast((self.wdata - los), tconfig.floatX))

        logpts = multivariate_normal(
            self.targets, self.weights, hyperparams, residuals)

        llk = pm.Deterministic(self._like_name, logpts)
        return llk.sum()

    def get_synthetics(self, point, **kwargs):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of the forward model

        Returns
        -------
        list with :class:`numpy.ndarray` synthetics for each target
        """
        self.point2sources(point)

        gc = self.config
        crust_inds = [0]

        synths = []
        for crust_ind in crust_inds:
            for target in self.targets:
                disp = heart.geo_layer_synthetics(
                    gc.gf_config.store_superdir,
                    crust_ind,
                    lons=target.lons,
                    lats=target.lats,
                    sources=self.sources, **kwargs)
                synths.append((
                    disp[:, 0] * target.los_vector[:, 0] + \
                    disp[:, 1] * target.los_vector[:, 1] + \
                    disp[:, 2] * target.los_vector[:, 2]))

        return synths

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        gc = self.config

        self.point2sources(point)

        for i, target in enumerate(self.targets):
            logger.debug('Track %s' % target.track)
            cov_pv = cov.get_geo_cov_velocity_models(
                store_superdir=gc.gf_config.store_superdir,
                crust_inds=range(gc.gf_config.n_variations + 1),
                target=target,
                sources=self.sources)

            cov_pv = utility.ensure_cov_psd(cov_pv)

            target.covariance.pred_v = cov_pv
            icov = target.covariance.inverse
            self.weights[i].set_value(icov)


class SeismicComposite(Composite):
    """
    Comprises how to solve the non-linear seismic forward model.

    Parameters
    ----------
    sc : :class:`config.SeismicConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, sc, project_dir, hypers=False):

        logger.debug('Setting up seismic structure ...\n')
        self.name = 'seismic'
        self._like_name = 'seis_like'

        self.engine = gf.LocalEngine(
            store_superdirs=[sc.gf_config.store_superdir])

        seismic_data_path = os.path.join(
            project_dir, bconfig.seismic_data_name)
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

        self.targets = heart.init_targets(
            self.stations,
            channels=sc.channels,
            sample_rate=sc.gf_config.sample_rate,
            crust_inds=[0],  # always reference model
            interpolation='multilinear')

        self.n_t = len(self.targets)
        logger.info('Number of seismic datasets: %i ' % self.n_t)

        if sc.calc_data_cov:
            logger.info('Estimating seismic data-covariances ...\n')
            cov_ds_seismic = cov.get_seismic_data_covariances(
                data_traces=self.data_traces,
                filterer=sc.filterer,
                sample_rate=sc.gf_config.sample_rate,
                arrival_taper=sc.arrival_taper,
                engine=self.engine,
                event=self.event,
                targets=self.targets)
        else:
            logger.info('No data-covariance estimation ...\n')
            cov_ds_seismic = []
            at = sc.arrival_taper
            n_samples = int(num.ceil(
                (num.abs(at.a) + at.d) * sc.gf_config.sample_rate))

            for tr in self.data_traces:
                cov_ds_seismic.append(num.eye(n_samples))

        self.weights = []
        for t, target in enumerate(self.targets):
            if target.covariance.data is None:
                logger.debug(
                    'No data covariance given. Setting default: zero')
                target.covariance.data = num.zeros_like(
                    cov_ds_seismic[t])
                target.covariance.pred_v = cov_ds_seismic[t]

            icov = target.covariance.inverse
            self.weights.append(shared(icov))

        super(SeismicComposite, self).__init__(hypers=hypers)

    def assemble_results(self, point):
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

        logger.debug('Assembling seismic waveforms ...')

        syn_proc_traces = self.get_synthetics(point, outmode='stacked_traces')

        tmins = [tr.tmin for tr in syn_proc_traces]

        at = copy.deepcopy(self.config.arrival_taper)

        obs_proc_traces = heart.taper_filter_traces(
            self.data_traces,
            arrival_taper=at,
            filterer=self.config.filterer,
            tmins=tmins,
            outmode='traces')

        self.config.arrival_taper = None

        syn_filt_traces = self.get_synthetics(point, outmode='data')

        obs_filt_traces = heart.taper_filter_traces(
            self.data_traces,
            filterer=self.config.filterer,
            outmode='traces')

        factor = 2.
        for i, (trs, tro) in enumerate(zip(syn_filt_traces, obs_filt_traces)):

            trs.chop(tmin=tmins[i] - factor * at.fade,
                     tmax=tmins[i] + factor * at.fade + at.duration)
            tro.chop(tmin=tmins[i] - factor * at.fade,
                     tmax=tmins[i] + factor * at.fade + at.duration)

        self.config.arrival_taper = at

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

        return results

    def update_llks(self, point):
        """
        Update posterior likelihoods of the composite with respect to one point
        in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        results = self.assemble_results(point)
        for k, result in enumerate(results):
            icov = self.targets[k].covariance.inverse
            _llk = num.array(result.processed_res.ydata.dot(
                icov).dot(result.processed_res.ydata.T)).flatten()
            self._llks[k].set_value(_llk)


class SeismicGeometryComposite(SeismicComposite):
    """
    Comprises how to solve the non-linear seismic forward model.

    Parameters
    ----------
    sc : :class:`config.SeismicConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    sources : list
        of :class:`pyrocko.gf.seismosizer.Source`
    event : :class:`pyrocko.model.Event`
        contains information of reference event, coordinates of reference
        point and source time
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, sc, project_dir, sources, event, hypers=False):

        super(SeismicGeometryComposite, self).__init__(
            sc, project_dir, hypers=hypers)

        self.event = event
        self.sources = sources

        # syntetics generation
        logger.debug('Initialising synthetics functions ... \n')
        self.get_synths = theanof.SeisSynthesizer(
            engine=self.engine,
            sources=self.sources,
            targets=self.targets,
            event=self.event,
            arrival_taper=sc.arrival_taper,
            filterer=sc.filterer)

        self.chop_traces = theanof.SeisDataChopper(
            sample_rate=sc.gf_config.sample_rate,
            traces=self.data_traces,
            arrival_taper=sc.arrival_taper,
            filterer=sc.filterer)

        self.config = sc

    def __getstate__(self):
        outstate = (
            self.config,
            self.sources,
            self.weights,
            self.targets,
            self.stations,
            self.engine)

        return outstate

    def __setstate__(self, state):
            self.config, \
            self.sources, \
            self.weights, \
            self.targets, \
            self.stations, \
            self.engine = state

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.
        """
        tpoint = copy.deepcopy(point)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = bconfig.hyper_pars.values()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source = self.sources[0]
        source_params = source.stf.keys() + source.keys()

        for param in tpoint.keys():
            if param not in source_params:
                tpoint.pop(param)

        tpoint['time'] += self.event.time

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            heart.adjust_fault_reference(source, input_depth='top')

    def get_formula(self, input_rvs, hyperparams):
        """
        Get seismic likelihood formula for the model built. Has to be called
        within a with model context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        self.input_rvs = input_rvs

        logger.debug(
            'Teleseismic optimization on: \n '
            ' %s' % ', '.join(self.input_rvs.keys()))

        t2 = time.time()
        synths, tmins = self.get_synths(self.input_rvs)
        t3 = time.time()
        logger.debug(
            'Teleseismic forward model on test model takes: %f' % \
                (t3 - t2))

        data_trcs = self.chop_traces(tmins)

        residuals = data_trcs - synths

        logpts = multivariate_normal(
            self.targets, self.weights, hyperparams, residuals)

        llk = pm.Deterministic(self._like_name, logpts)
        return llk.sum()

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
        list of synthetics for each target
        """
        self.point2sources(point)

        sc = self.config
        synths, _ = heart.seis_synthetics(
            engine=self.engine,
            sources=self.sources,
            targets=self.targets,
            arrival_taper=sc.arrival_taper,
            filterer=sc.filterer, **kwargs)

        return synths

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        sc = self.config

        self.point2sources(point)

        for j, channel in enumerate(sc.channels):
            for i, station in enumerate(self.stations):
                logger.debug('Channel %s of Station %s ' % (
                    channel, station.station))
                crust_targets = heart.init_targets(
                    stations=[station],
                    channels=channel,
                    sample_rate=sc.gf_config.sample_rate,
                    crust_inds=range(sc.gf_config.n_variations + 1))

                cov_pv = cov.get_seis_cov_velocity_models(
                    engine=self.engine,
                    sources=self.sources,
                    targets=crust_targets,
                    arrival_taper=sc.arrival_taper,
                    filterer=sc.filterer,
                    plot=plot, n_jobs=n_jobs)

                cov_pv = utility.ensure_cov_psd(cov_pv)

                self.engine.close_cashed_stores()

                index = j * len(self.stations) + i

                self.targets[index].covariance.pred_v = cov_pv
                icov = self.targets[index].covariance.inverse
                self.weights[index].set_value(icov)


class GeodeticDistributorComposite(GeodeticComposite):
    """
    Comprises how to solve the geodetic (static) linear forward model.
    Distributed slip
    """

    gfs = {}
    sgfs = {}
    gf_names = {}

    def __init__(self, gc, project_dir, hypers=False):

        super(GeodeticDistributorComposite, self).__init__(
            gc, project_dir, hypers=hypers)

        self._mode = 'static'
        self.gfpath = os.path.join(project_dir, self._mode,
                         bconfig.linear_gf_dir_name)

        self.data = [
            shared(target.displacement.astype(tconfig.floatX), borrow=True) \
            for target in self.targets]
        self.odws = [
            shared(target.odw.astype(tconfig.floatX), borrow=True) \
            for target in self.targets]

    def load_gfs(self, crust_inds=None, make_shared=True):
        """
        Load Greens Function matrixes for each variable to be inverted for.
        Updates gfs and gf_names attributes.

        Parameters
        ----------
        crust_inds : list
            of int to indexes of Green's Functions
        make_shared : bool
            if True transforms gfs to :class:`theano.shared` variables
        """

        if crust_inds is None:
            crust_inds = range(self.gc.gf_config.n_variations + 1)

        for crust_ind in crust_inds:
            gfpath = os.path.join(self.gfpath,
                str(crust_ind) + '_' + bconfig.geodetic_linear_gf_name)

            self.gf_names[crust_ind] = gfpath
            gfs = utility.load_objects(gfpath)[0]

            if make_shared:
                self.sgfs[crust_ind] = {param: [
                    shared(gf.astype(tconfig.floatX), borrow=True) \
                        for gf in gfs[param]] \
                            for param in gfs.keys()}
            else:
                self.gfs[crust_ind] = gfs

    def get_formula(self, input_rvs, hyperparams):

        residuals = [None for i in range(self.n_t)]
        for t in range(self.n_t):

            mu = tt.zeros_like(self.data[t], tconfig.floatX)
            for var, rv in input_rvs.iteritems():
                mu += tt.dot(self.sgfs[0][var][t], rv)

            residuals[t] = self.odws[t] * (self.data[t] - mu)

        logpts = multivariate_normal(
            self.targets, self.weights, hyperparams, residuals)

        llk = pm.Deterministic(self._like_name, logpts)

        return llk.sum()

    def get_synthetics(self, point, outmode='data'):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of the forward model

        Returns
        -------
        list with :class:`numpy.ndarray` synthetics for each target
        """
        if len(self.gfs.keys()) == 0:
            self.load_gfs(crust_inds=[0], make_shared=False)

        tpoint = copy.deepcopy(point)

        hps = bconfig.hyper_pars.values()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        gf_params = self.gfs[0].keys()

        for param in tpoint.keys():
            if param not in gf_params:
                tpoint.pop(param)

        synthetics = []
        for i, target in enumerate(self.targets):

            mu = num.zeros_like(target.displacement)
            for var, rv in tpoint.iteritems():
                mu += num.dot(self.gfs[0][var][i], rv)
                synthetics.append(mu)

        return synthetics


geometry_composite_catalog = {
    'seismic': SeismicGeometryComposite,
    'geodetic': GeodeticGeometryComposite}


distributor_composite_catalog = {
    'geodetic': GeodeticDistributorComposite,
    }


class Problem(object):
    """
    Overarching class for the optimization problems to be solved.

    Parameters
    ----------
    config : :class:`beat.BEATConfig`
        Configuration object that contains the problem definition.
    """

    event = None
    model = None
    _like_name = 'like'
    composites = {}
    hyperparams = {}

    def __init__(self, config):

        logger.info('Analysing problem ...')
        logger.info('---------------------\n')

        self.config = config

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
                    ' proposal_distribution %s, tune_interval=%i,'
                    ' n_jobs=%i \n' % (
                    sc.parameters.proposal_dist, sc.parameters.tune_interval,
                    sc.parameters.n_jobs))

                t1 = time.time()
                if hypers:
                    step = Metropolis(
                        tune_interval=sc.parameters.tune_interval,
                        proposal_dist=atmcmc.proposal_dists[
                            sc.parameters.proposal_dist])
                else:
                    step = atmcmc.ATMCMC(
                        n_chains=sc.parameters.n_jobs,
                        tune_interval=sc.parameters.tune_interval,
                        likelihood_name=self._like_name,
                        proposal_name=sc.parameters.proposal_dist)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

            elif sc.name == 'ATMCMC':
                logger.info(
                    '... Initiate Adaptive Transitional Metropolis ... \n'
                    ' n_chains=%i, tune_interval=%i, n_jobs=%i \n' % (
                        sc.parameters.n_chains, sc.parameters.tune_interval,
                        sc.parameters.n_jobs))

                t1 = time.time()
                step = atmcmc.ATMCMC(
                    n_chains=sc.parameters.n_chains,
                    tune_interval=sc.parameters.tune_interval,
                    coef_variation=sc.parameters.coef_variation,
                    proposal_dist=sc.parameters.proposal_dist,
                    likelihood_name=self._like_name)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

        if 'seismic' in self.composites.keys():
            composite = self.composites['seismic']
            composite.engine.close_cashed_stores()

        return step

    def built_model(self):
        """
        Initialise :class:`pymc3.Model` depending on problem composites,
        geodetic and/or seismic data are included. Composites also determine
        the problem to be solved.
        """

        logger.info('... Building model ...\n')

        mode = self.config.problem_config.mode

        self.outfolder = os.path.join(self.config.project_dir, mode)
        util.ensuredir(self.outfolder)

        with pm.Model() as self.model:

            pc = self.config.problem_config

            logger.debug('Optimization for %i sources', pc.n_sources)

            rvs = dict()
            for param in pc.priors.itervalues():
                rvs[param.name] = pm.Uniform(
                    param.name,
                    shape=param.dimension,
                    lower=param.lower,
                    upper=param.upper,
                    testval=param.testvalue,
                    transform=None,
                    dtype=tconfig.floatX)

            self.hyperparams = self.get_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for dataset, composite in self.composites.iteritems():
                input_rvs = utility.weed_input_rvs(rvs, mode, dataset=dataset)
                total_llk += composite.get_formula(input_rvs, self.hyperparams)

            like = pm.Deterministic(self._like_name, total_llk)
            llk = pm.Potential(self._like_name, like)
            logger.info('Model building was successful!')

    def built_hyper_model(self):
        """
        Initialise :class:`pymc3.Model` depending on configuration file,
        geodetic and/or seismic data are included. Estimates initial parameter
        bounds for hyperparameters.
        """

        logger.info('... Building Hyper model ...\n')

        pc = self.config.problem_config

        self.outfolder = os.path.join(
            self.config.project_dir, pc.mode, 'hypers')
        util.ensuredir(self.outfolder)

        point = {}
        for param in pc.priors.values():
            point[param.name] = param.testvalue

        self.update_llks(point)

        with pm.Model() as self.model:

            self.hyperparams = self.get_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for composite in self.composites.itervalues():
                total_llk += composite.get_hyper_formula(self.hyperparams)

            like = pm.Deterministic(self._like_name, total_llk)
            llk = pm.Potential(self._like_name, like)
            logger.info('Hyper model building was successful!')

    def get_random_point(self):
        """
        Get random point in solution space.
        """
        pc = self.config.problem_config

        point = {param.name: param.random() for param in pc.priors.values()}
        hps = {param.name: param.random() \
            for param in pc.hyperparameters.values()}

        for k, v in hps.iteritems():
            point[k] = v

        return point

    def get_hyperparams(self):
        """
        Evaluate problem setup and return hyperparameter dictionary.
        Has to be executed in a "with model context"!
        """
        pc = self.config.problem_config

        hyperparams = {}
        n_hyp = len(pc.hyperparameters.keys())

        logger.debug('Optimization for %i hyperparemeters', n_hyp)

        for hp_name in bconfig.hyper_pars.values():
            if hp_name in pc.hyperparameters:
                hyperpar = pc.hyperparameters[hp_name]
                hyperparams[hp_name] = pm.Uniform(
                    hyperpar.name,
                    shape=hyperpar.dimension,
                    lower=hyperpar.lower,
                    upper=hyperpar.upper,
                    testval=hyperpar.testvalue,
                    transform=None)
            else:
                hyperparams[hp_name] = 0.

        return hyperparams

    def update_llks(self, point):
        """
        Update posterior likelihoods of each composite of the problem with
        respect to one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        for composite in self.composites.itervalues():
            composite.update_llks(point)

    def apply(self, problem):
        """
        Update composites in problem object with given composites.
        """
        for composite in problem.composites.values():
            self.composites[composite.name].apply(composite)

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Calculate and update model prediction uncertainty covariances of
        composites due to uncertainty in the velocity model with respect to
        one point in the solution space. Shared variables are updated in place.

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
        for composite in self.composites.itervalues():
            composite.update_weights(point, n_jobs=n_jobs)

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
        Dictionary with keys according to composites containing the synthetics
        as lists.
        """

        d = dict()

        for composite in self.composites.itervalues():
            d[composite.name] = composite.get_synthetics(point, outmode='data')

        return d


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

    def __init__(self, config, hypers=False):
        logger.info('... Initialising Geometry Optimizer ... \n')

        super(GeometryOptimizer, self).__init__(config)

        # Load event
        if config.event is None:
            logger.warn('Found no event information!')
        else:
            self.event = config.event

        # Init sources
        self.sources = []
        for i in range(config.problem_config.n_sources):
            if self.event:
                source = heart.RectangularSource.from_pyrocko_event(self.event)
                # hardcoded inversion for hypocentral time
                source.stf.anchor = -1.
            else:
                source = heart.RectangularSource()

            self.sources.append(source)

        dsources = utility.transform_sources(
            self.sources,
            config.problem_config.datasets)

        for dataset in config.problem_config.datasets:
            self.composites[dataset] = geometry_composite_catalog[dataset](
                config[dataset + '_config'],
                config.project_dir,
                dsources[dataset],
                self.event,
                hypers)

        self.config = config


class DistributionOptimizer(Problem):
    """
    Defines the model setup to solve the linear slip-distribution and
    returns the model object.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):
        logger.info('... Initialising Distribution Optimizer ... \n')

        super(DistributionOptimizer, self).__init__(config)

        for dataset in config.problem_config.datasets:
            composite = distributor_composite_catalog[dataset](
                config[dataset + '_config'],
                config.project_dir,
                hypers)

            # do the optimization only on the reference velocity model
            logger.info("Loading %s Green's Functions" % dataset)
            composite.load_gfs(crust_inds=[0])
            self.composites[dataset] = composite

        self.config = config


problem_catalog = {
    bconfig.modes_catalog.keys()[0]: GeometryOptimizer,
    bconfig.modes_catalog.keys()[1]: DistributionOptimizer}


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

        metropolis.Metropolis_sample(
            n_stages=pa.n_stages,
            n_steps=pa.n_steps,
            stage=pa.stage,
            step=step,
            progressbar=True,
            trace=problem.outfolder,
            burn=pa.burn,
            thin=pa.thin,
            model=problem.model,
            n_jobs=pa.n_jobs,
            update=update,
            rm_flag=pa.rm_flag)

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
    sc0 = problem.config.sampler_config
    sc = problem.config.hyper_sampler_config
    pa = sc.parameters

    name = problem.outfolder
    util.ensuredir(name)

    mtraces = []
    for stage in range(pa.n_stages):
        logger.info('Metropolis stage %i' % stage)

        if stage == 0:
            point = {param.name: param.testvalue \
                for param in pc.priors.values()}
        else:
            point = {param.name: param.random() \
                for param in pc.priors.values()}

        problem.outfolder = os.path.join(name, 'stage_%i' % stage)
        start = {param.name: param.random() for param in \
                                            pc.hyperparameters.itervalues()}

        if pa.rm_flag:
            shutil.rmtree(problem.outfolder, ignore_errors=True)

        if not os.path.exists(problem.outfolder):
            logger.debug('Sampling ...')
            if sc0.parameters.update_covariances:
                problem.update_weights(point)

            problem.update_llks(point)
            with problem.model as hmodel:
                mtraces.append(pm.sample(
                    draws=pa.n_steps,
                    step=step,
                    trace=pm.backends.Text(
                        name=problem.outfolder,
                        model=hmodel),
                    start=start,
                    model=hmodel,
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

    for v, i in pc.hyperparameters.iteritems():
        d = mtrace.get_values(
            v, combine=True, burn=int(n_steps * pa.burn),
            thin=pa.thin, squeeze=True)
        lower = num.floor(d.min(axis=0)) - 1.
        upper = num.ceil(d.max(axis=0)) + 1.
        logger.info('Updating hyperparameter %s from %f, %f to %f, %f' % (
            v, i.lower, i.upper, lower, upper))
        pc.hyperparameters[v].lower = lower
        pc.hyperparameters[v].upper = upper
        pc.hyperparameters[v].testvalue = (upper + lower) / 2.

    config_file_name = 'config_' + pc.mode + '.yaml'
    conf_out = os.path.join(problem.config.project_dir, config_file_name)

    problem.config.problem_config = pc
    bconfig.dump(problem.config, filename=conf_out)


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
    problem : :class:`Problem`
    """

    config = bconfig.load_config(project_dir, mode)

    pc = config.problem_config

    if hypers and len(pc.hyperparameters) == 0:
        raise Exception('No hyperparameters specified!'
        ' option --hypers not applicable')

    if pc.mode in problem_catalog.keys():
        problem = problem_catalog[pc.mode](config, hypers)
    else:
        logger.error('Modeling problem %s not supported' % pc.mode)
        raise Exception('Model not supported')

    if hypers:
        problem.built_hyper_model()
    else:
        problem.built_model()
    return problem


class Stage(object):
    """
    Stage, containing sampling results and intermediate sampler
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

    stage = Stage(path=stagepath, number=stage_number)

    if 'trace' in to_load:
        stage.mtrace = backend.load(stagepath, model=problem.model)

    if 'params' in to_load:
        stage.step, stage.updates = backend.load_sampler_params(
            project_dir, stage_number, mode)

    return stage
