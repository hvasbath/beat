from logging import getLogger
import os
import copy
from time import time

import numpy as num

from theano import shared
from theano import config as tconfig
import theano.tensor as tt
from theano.printing import Print

from pyrocko.gf import LocalEngine
from pyrocko.trace import Trace

from beat import theanof, utility
from beat.ffi import load_gf_library, get_gf_prefix
from beat import config as bconfig
from beat import heart, covariance as cov
from beat.models.base import ConfigInconsistentError, Composite
from beat.models.distributions import multivariate_normal_chol

from pymc3 import Uniform, Deterministic


logger = getLogger('seismic')


__all__ = [
    'SeismicGeometryComposite',
    'SeismicDistributerComposite']


class SeismicComposite(Composite):
    """
    Comprises how to solve the non-linear seismic forward model.

    Parameters
    ----------
    sc : :class:`config.SeismicConfig`
        configuration object containing seismic setup parameters
    event: :class:`pyrocko.model.Event`
    project_dir : str
        directory of the model project, where to find the data
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """
    _datasets = None
    _weights = None
    _targets = None

    def __init__(self, sc, event, project_dir, hypers=False):

        super(SeismicComposite, self).__init__()

        logger.debug('Setting up seismic structure ...\n')
        self.name = 'seismic'
        self._like_name = 'seis_like'
        self.correction_name = 'time_shift'

        self.event = event
        self.engine = LocalEngine(
            store_superdirs=[sc.gf_config.store_superdir])

        seismic_data_path = os.path.join(
            project_dir, bconfig.seismic_data_name)

        self.datahandler = heart.init_datahandler(
            seismic_config=sc, seismic_data_path=seismic_data_path)

        self.wavemaps = []
        for wc in sc.waveforms:
            if wc.include:
                wmap = heart.init_wavemap(
                    waveformfit_config=wc,
                    datahandler=self.datahandler,
                    event=event)

                if sc.calc_data_cov:
                    logger.info(
                        'Estimating seismic data-covariances '
                        'for %s ...\n' % wmap.name)

                    cov_ds_seismic = cov.seismic_data_covariance(
                        data_traces=wmap.datasets,
                        filterer=wc.filterer,
                        sample_rate=sc.gf_config.sample_rate,
                        arrival_taper=wc.arrival_taper,
                        engine=self.engine,
                        event=self.event,
                        targets=wmap.targets)
                else:
                    logger.info('No data-covariance estimation, using imported'
                                ' covariances...\n')

                    cov_ds_seismic = []
                    at = wc.arrival_taper
                    n_samples = int(num.ceil(
                        at.duration * sc.gf_config.sample_rate))

                    for trc in wmap.datasets:
                        if trc.covariance is None:
                            logger.warn(
                                'No data covariance given/estimated! '
                                'Setting default: eye')
                            cov_ds_seismic.append(num.eye(n_samples))
                        else:
                            data_cov = trc.covariance.data
                            if data_cov.shape[0] != n_samples:
                                raise ValueError(
                                    'Imported covariance %i does not agree '
                                    ' with taper duration %i!' % (
                                        data_cov.shape[0], n_samples))
                            cov_ds_seismic.append(data_cov)

                weights = []
                for t, trc in enumerate(wmap.datasets):
                    trc.covariance = heart.Covariance(data=cov_ds_seismic[t])
                    if int(trc.covariance.data.sum()) == trc.data_len():
                        logger.warn('Data covariance is identity matrix!'
                                    ' Please double check!!!')
                    icov = trc.covariance.chol_inverse
                    weights.append(
                        shared(
                            icov,
                            name='seis_%s_weight_%i' % (wc.name, t),
                            borrow=True))

                wmap.add_weights(weights)

                self.wavemaps.append(wmap)
            else:
                logger.info(
                    'The waveform defined in "%s" config is not '
                    'included in the optimization!' % wc.name)

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(
                    shared(
                        num.array([1.]), name='seis_llk_%i' % t, borrow=True))

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__.copy()

    def init_hierarchicals(self, problem_config):
        """
        Initialise random variables for temporal station corrections.
        """
        if not self.config.station_corrections and \
                self.correction_name in problem_config.hierarchicals:
                raise ConfigInconsistentError(
                    'Station corrections disabled, but they are defined'
                    ' in the problem configuration!')

        if self.config.station_corrections and \
                self.correction_name not in problem_config.hierarchicals:
                raise ConfigInconsistentError(
                    'Station corrections enabled, but they are not defined'
                    ' in the problem configuration!')

        if self.correction_name in problem_config.hierarchicals:
            nhierarchs = len(self.get_unique_stations())
            param = problem_config.hierarchicals[self.correction_name]
            logger.info(
                'Estimating time shift for each station...')
            kwargs = dict(
                name=self.correction_name,
                shape=nhierarchs,
                lower=num.repeat(param.lower, nhierarchs),
                upper=num.repeat(param.upper, nhierarchs),
                testval=num.repeat(param.testvalue, nhierarchs),
                transform=None,
                dtype=tconfig.floatX)

            try:
                station_corrs_rv = Uniform(**kwargs)

            except TypeError:
                kwargs.pop('name')
                station_corrs_rv = Uniform.dist(**kwargs)

            self.hierarchicals[self.correction_name] = station_corrs_rv
        else:
            nhierarchs = 0

    def get_unique_stations(self):
        sl = [wmap.stations for wmap in self.wavemaps]
        us = []
        for s in sl:
            us.extend(s)
        return list(set(us))

    @property
    def n_t(self):
        return sum(wmap.n_t for wmap in self.wavemaps)

    @property
    def datasets(self):
        if self._datasets is None:
            ds = []
            for wmap in self.wavemaps:
                ds.extend(wmap.datasets)

            self._datasets = ds
        return self._datasets

    @property
    def weights(self):
        if self._weights is None:
            ws = []
            for wmap in self.wavemaps:
                ws.extend(wmap.weights)

            self._weights = ws
        return self._weights

    @property
    def targets(self):
        if self._targets is None:
            ts = []
            for wmap in self.wavemaps:
                ts.extend(wmap.targets)

            self._targets = ts
        return self._targets

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

        syn_proc_traces, obs_proc_traces = self.get_synthetics(
            point, outmode='stacked_traces')

        syn_filt_traces, obs_filt_traces = self.get_synthetics(
            point, outmode='stacked_traces', taper_tolerance_factor=2.)

        ats = []
        for wmap in self.wavemaps:
            wc = wmap.config
            ats.extend(wmap.n_t * [wc.arrival_taper])

        results = []
        for i, (obs_tr, at) in enumerate(zip(obs_proc_traces, ats)):

            dtrace = obs_tr.copy()
            dtrace.set_ydata(
                (obs_tr.get_ydata() - syn_proc_traces[i].get_ydata()))

            taper = at.get_pyrocko_taper(
                float(obs_tr.tmin + num.abs(at.a)))

            results.append(heart.SeismicResult(
                processed_obs=obs_tr,
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
            choli = self.datasets[k].covariance.chol_inverse
            tmp = choli.dot(result.processed_res.ydata)
            _llk = num.asarray([num.dot(tmp, tmp)])
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
            sc, event, project_dir, hypers=hypers)

        self.synthesizers = {}
        self.choppers = {}

        self.sources = sources

        if sc.station_corrections:
            self.correction_name = 'time_shift'

        if not hypers:
            # syntetics generation
            logger.debug('Initialising synthetics functions ... \n')
            for wmap in self.wavemaps:
                wc = wmap.config

                self.synthesizers[wc.name] = theanof.SeisSynthesizer(
                    engine=self.engine,
                    sources=self.sources,
                    targets=wmap.targets,
                    event=self.event,
                    arrival_taper=wc.arrival_taper,
                    wavename=wmap.name,
                    filterer=wc.filterer,
                    pre_stack_cut=sc.pre_stack_cut)

                self.choppers[wc.name] = theanof.SeisDataChopper(
                    sample_rate=sc.gf_config.sample_rate,
                    traces=wmap.datasets,
                    arrival_taper=wc.arrival_taper,
                    filterer=wc.filterer)

        self.config = sc

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.
        """
        tpoint = copy.deepcopy(point)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source = self.sources[0]
        source_params = source.keys() + source.stf.keys()

        for param in tpoint.keys():
            if param not in source_params:
                tpoint.pop(param)

        tpoint['time'] += self.event.time

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])

    def get_formula(
            self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Get seismic likelihood formula for the model built. Has to be called
        within a with model context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution`
        fixed_rvs : dict
            of :class:`numpy.array`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`
        problem_config : :class:`config.ProblemConfig`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        hp_specific = problem_config.dataset_specific_residual_noise_estimation

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            'Seismic optimization on: \n '
            ' %s' % ', '.join(self.input_rvs.keys()))

        t2 = time()
        wlogpts = []

        self.init_hierarchicals(problem_config)
        if self.config.station_corrections:
            logger.info(
                'Initialized %i hierarchical parameters for '
                'station corrections.' % len(self.get_unique_stations()))

        for wmap in self.wavemaps:
            synths, tmins = self.synthesizers[wmap.name](self.input_rvs)

            if len(self.hierarchicals) > 0:
                tmins += self.hierarchicals[
                    self.correction_name][wmap.station_correction_idxs]

            data_trcs = self.choppers[wmap.name](tmins)
            residuals = data_trcs - synths

            logpts = multivariate_normal_chol(
                wmap.datasets, wmap.weights, hyperparams, residuals,
                hp_specific=hp_specific)

            wlogpts.append(logpts)

        t3 = time()
        logger.debug(
            'Teleseismic forward model on test model takes: %f' %
            (t3 - t2))

        llk = Deterministic(self._like_name, tt.concatenate((wlogpts)))
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
        default: array of synthetics for all targets
        """

        self.point2sources(point)

        sc = self.config
        synths = []
        obs = []
        for wmap in self.wavemaps:
            wc = wmap.config

            synthetics, tmins = heart.seis_synthetics(
                engine=self.engine,
                sources=self.sources,
                targets=wmap.targets,
                arrival_taper=wc.arrival_taper,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=sc.pre_stack_cut,
                **kwargs)

            if self.config.station_corrections:
                sh = point[
                    self.correction_name][wmap.station_correction_idxs]

                for i, tr in enumerate(synthetics):
                    tr.tmin += sh[i]
                    tr.tmax += sh[i]

            synths.extend(synthetics)

            obs_tr = heart.taper_filter_traces(
                wmap.datasets,
                arrival_taper=wc.arrival_taper,
                filterer=wc.filterer,
                tmins=tmins,
                **kwargs)

            obs.extend(obs_tr)

        return synths, obs

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

        for wmap in self.wavemaps:
            wc = wmap.config

            for channel in wmap.channels:
                datasets = wmap.get_datasets([channel])
                weights = wmap.get_weights([channel])

                for station, dataset, weight in zip(
                        wmap.stations, datasets, weights):

                    logger.debug('Channel %s of Station %s ' % (
                        channel, station.station))

                    crust_targets = heart.init_seismic_targets(
                        stations=[station],
                        earth_model_name=sc.gf_config.earth_model_name,
                        channels=channel,
                        sample_rate=sc.gf_config.sample_rate,
                        crust_inds=range(*sc.gf_config.n_variations),
                        reference_location=sc.gf_config.reference_location)

                    cov_pv = cov.seismic_cov_velocity_models(
                        engine=self.engine,
                        sources=self.sources,
                        targets=crust_targets,
                        wavename=wmap.name,
                        arrival_taper=wc.arrival_taper,
                        filterer=wc.filterer,
                        plot=plot, n_jobs=n_jobs)
                    cov_pv = utility.ensure_cov_psd(cov_pv)

                    self.engine.close_cashed_stores()

                    dataset.covariance.pred_v = cov_pv

                    t0 = time()
                    choli = dataset.covariance.chol_inverse
                    t1 = time()
                    logger.debug('Calculate weight time %f' % (t1 - t0))
                    weight.set_value(choli)
                    dataset.covariance.update_slog_pdet()


class SeismicDistributerComposite(SeismicComposite):
    """
    Comprises how to solve the seismic (kinematic) linear forward model.
    Distributed slip
    """

    def __init__(self, sc, project_dir, event, hypers=False):

        super(SeismicDistributerComposite, self).__init__(
            sc, event, project_dir, hypers=hypers)

        self.gfs = {}
        self.gf_names = {}
        self.choppers = {}
        self.sweep_implementation = 'c'

        self.slip_varnames = bconfig.static_dist_vars
        self._mode = 'ffi'
        self.gfpath = os.path.join(
            project_dir, self._mode, bconfig.linear_gf_dir_name)

        self.config = sc
        sgfc = sc.gf_config

        if sgfc.patch_width != sgfc.patch_length:
            raise ValueError(
                'So far only square patches supported in kinematic'
                ' model! - fast_sweeping issues')

        if len(sgfc.reference_sources) > 1:
            raise ValueError(
                'So far only one reference plane supported! - '
                'fast_sweeping issues')

        self.fault = self.load_fault_geometry()
        n_p_dip, n_p_strike = self.fault.get_subfault_discretization(0)

        logger.info('Fault discretized to %s [km]'
                    ' patches.' % sgfc.patch_length)
        if not hypers:
            self.sweeper = theanof.Sweeper(
                sgfc.patch_length,
                n_p_dip,
                n_p_strike,
                self.sweep_implementation)

            for wmap in self.wavemaps:
                self.choppers[wmap.name] = theanof.SeisDataChopper(
                    sample_rate=sc.gf_config.sample_rate,
                    traces=wmap.datasets,
                    arrival_taper=wmap.config.arrival_taper,
                    filterer=wmap.config.filterer)

    def load_fault_geometry(self):
        """
        Load fault-geometry, i.e. discretized patches.

        Returns
        -------
        :class:`heart.FaultGeometry`
        """
        return utility.load_objects(
            os.path.join(self.gfpath, bconfig.fault_geometry_name))[0]

    def get_gflibrary_key(self, crust_ind, wavename, component):
        return '%i_%s_%s' % (crust_ind, wavename, component)

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
        if not isinstance(crust_inds, list):
            raise TypeError('crust_inds need to be a list!')

        if crust_inds is None:
            crust_inds = range(*self.config.gf_config.n_variations)

        for wmap in self.wavemaps:
            for crust_ind in crust_inds:
                gfs = {}
                for var in self.slip_varnames:
                    gflib_name = get_gf_prefix(
                        datatype=self.name, component=var,
                        wavename=wmap.config.name, crust_ind=crust_ind)
                    gfpath = os.path.join(
                        self.gfpath, gflib_name)

                    gfs = load_gf_library(
                        directory=self.gfpath, filename=gflib_name)

                    if make_shared:
                        gfs.init_optimization()

                    key = self.get_gflibrary_key(
                        crust_ind=crust_ind,
                        wavename=wmap.config.name,
                        component=var)

                    self.gf_names[key] = gfpath
                    self.gfs[key] = gfs

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):

        hp_specific = problem_config.dataset_specific_residual_noise_estimation

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            'Seismic optimization on: \n '
            ' %s' % ', '.join(self.input_rvs.keys()))

        t2 = time()
        wlogpts = []

        self.init_hierarchicals(problem_config)
        if self.config.station_corrections:
            logger.info(
                'Initialized %i hierarchical parameters for '
                'station corrections.' % len(self.get_unique_stations()))

        self.input_rvs.update(fixed_rvs)

        ref_idx = self.config.gf_config.reference_model_idx

        nuc_strike = input_rvs['nucleation_strike']
        nuc_dip = input_rvs['nucleation_dip']

        t2 = time()
        # convert velocities to rupture onset
        logger.debug('Fast sweeping ...')

        nuc_dip_idx, nuc_strike_idx = self.fault.fault_locations2idxs(
            positions_dip=nuc_dip,
            positions_strike=nuc_strike,
            backend='theano')

        starttimes = self.sweeper(
            (1. / input_rvs['velocities']), nuc_dip_idx, nuc_strike_idx)

        wlogpts = []
        for wmap in self.wavemaps:
            logger.debug('Stacking %s phase ...' % wmap.config.name)
            synthetics = tt.zeros(
                (wmap.n_t, wmap.config.arrival_taper.nsamples(
                    self.config.gf_config.sample_rate)),
                dtype=tconfig.floatX)

            for var in self.slip_varnames:
                logger.debug('Stacking %s variable' % var)
                key = self.get_gflibrary_key(
                    crust_ind=ref_idx, wavename=wmap.name, component=var)
                synthetics += self.gfs[key].stack_all(
                    starttimes=starttimes,
                    durations=input_rvs['durations'],
                    slips=input_rvs[var],
                    interpolation=wmap.config.interpolation)

            logger.debug('Get hypocenter location ...')
            patchidx = self.fault.spatchmap(
                0, dipidx=nuc_dip_idx, strikeidx=nuc_strike_idx)

            # cut data according to wavemaps
            logger.debug('Cut data accordingly')

            tmins = self.gfs[key].get_all_tmins(
                patchidx).ravel() + input_rvs['nucleation_time']

            # add station corrections
            if len(self.hierarchicals) > 0:
                tmins += self.hierarchicals[
                    self.correction_name][wmap.station_correction_idxs]

            data_traces = self.choppers[wmap.name](tmins)

            residuals = data_traces - synthetics

            logger.debug('Calculating likelihoods ...')
            logpts = multivariate_normal_chol(
                wmap.datasets, wmap.weights, hyperparams, residuals,
                hp_specific=hp_specific)

            wlogpts.append(logpts)

        t3 = time()
        logger.debug(
            'Seismic formula on test model takes: %f' % (t3 - t2))

        llk = Deterministic(self._like_name, tt.concatenate((wlogpts)))
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
        list with :class:`heart.SeismicDataset` synthetics for each target
        """
        ref_idx = self.config.gf_config.reference_model_idx
        if len(self.gfs) == 0:
            self.load_gfs(
                crust_inds=[ref_idx],
                make_shared=False)

        tpoint = copy.deepcopy(point)

        hps = self.config.get_hypernames()
        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        nuc_dip_idx, nuc_strike_idx = self.fault.fault_locations2idxs(
            positions_dip=tpoint['nucleation_dip'],
            positions_strike=tpoint['nucleation_strike'],
            backend='numpy')

        starttimes = self.fault.get_subfault_starttimes(
            index=0,
            rupture_velocities=tpoint['velocities'],
            nuc_dip_idx=nuc_dip_idx,
            nuc_strike_idx=nuc_strike_idx).flatten()

        patchidx = self.fault.patchmap(
            index=0, dipidx=nuc_dip_idx, strikeidx=nuc_strike_idx)

        synth_traces = []
        obs_traces = []
        for wmap in self.wavemaps:
            synthetics = num.zeros(
                (wmap.n_t, wmap.config.arrival_taper.nsamples(
                    self.config.gf_config.sample_rate)))
            for var in self.slip_varnames:
                key = self.get_gflibrary_key(
                    crust_ind=ref_idx, wavename=wmap.name, component=var)

                try:
                    gflibrary = self.gfs[key]
                except KeyError:
                    raise KeyError(
                        'GF library %s not loaded! Loaded GFs:'
                        ' %s' % (key, utility.list2string(self.gfs.keys())))

                gflibrary.set_stack_mode('numpy')
                synthetics += gflibrary.stack_all(
                    starttimes=starttimes,
                    durations=tpoint['durations'],
                    slips=tpoint[var],
                    interpolation=wmap.config.interpolation)

            for i, target in enumerate(wmap.targets):
                tr = Trace(
                    ydata=synthetics[i, :],
                    tmin=float(
                        gflibrary.reference_times[i] +
                        tpoint['nucleation_time']),
                    deltat=gflibrary.deltat)

                tr.set_codes(*target.codes)
                synth_traces.append(tr)

            if self.config.station_corrections:
                sh = point[
                    self.correction_name][wmap.station_correction_idxs]

                for i, tr in enumerate(synth_traces):
                    tr.tmin += sh[i]
                    tr.tmax += sh[i]

            obs_traces.extend(heart.taper_filter_traces(
                wmap.datasets,
                arrival_taper=wmap.config.arrival_taper,
                filterer=wmap.config.filterer,
                tmins=(gflibrary.get_all_tmins(patchidx)),
                **kwargs))

        return synth_traces, obs_traces

    def update_weights(self, point, n_jobs=1, plot=False):
        logger.warning('Not implemented yet!')
        raise NotImplementedError('Not implemented yet!')
