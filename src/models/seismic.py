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
from beat.ffo import load_gf_library, get_gf_prefix
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

        self.noise_analyser = cov.SeismicNoiseAnalyser(
            structure=sc.noise_estimator.structure,
            pre_arrival_time=sc.noise_estimator.pre_arrival_time,
            engine=self.engine,
            event=self.event,
            chop_bounds=['b', 'c'])

        self.wavemaps = []
        for i, wc in enumerate(sc.waveforms):
            if wc.include:
                wmap = heart.init_wavemap(
                    waveformfit_config=wc,
                    datahandler=self.datahandler,
                    event=event,
                    mapnumber=i)

                self.wavemaps.append(wmap)
            else:
                logger.info(
                    'The waveform defined in "%s %i" config is not '
                    'included in the optimization!' % (wc.name, i))

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(
                    shared(
                        num.array([1.]), name='seis_llk_%i' % t, borrow=True))

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__.copy()

    def analyse_noise(self, tpoint=None):
        """
        Analyse seismic noise in datatraces and set
        data-covariance matrixes accordingly.
        """
        if self.config.noise_estimator.structure == 'non-toeplitz':
            results = self.assemble_results(
                tpoint, order='wmap', chop_bounds=['b', 'c'])
        else:
            results = [None] * len(self.wavemaps)

        for wmap, wmap_results in zip(self.wavemaps, results):
            logger.info(
                'Retrieving seismic data-covariances with structure "%s" '
                'for %s ...' % (
                    self.config.noise_estimator.structure, wmap._mapid))

            cov_ds_seismic = self.noise_analyser.get_data_covariances(
                wmap=wmap, results=wmap_results,
                sample_rate=self.config.gf_config.sample_rate)

            for j, trc in enumerate(wmap.datasets):
                if trc.covariance is None:
                    trc.covariance = heart.Covariance(data=cov_ds_seismic[j])
                else:
                    trc.covariance.data = cov_ds_seismic[j]

                if int(trc.covariance.data.sum()) == trc.data_len():
                    logger.warn('Data covariance is identity matrix!'
                                ' Please double check!!!')

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

    def init_weights(self):
        """
        Initialise shared weights in wavemaps.
        """
        for wmap in self.wavemaps:
            weights = []
            for j, trc in enumerate(wmap.datasets):
                icov = trc.covariance.chol_inverse
                weights.append(
                    shared(
                        icov,
                        name='seis_%s_weight_%i' % (wmap._mapid, j),
                        borrow=True))

            wmap.add_weights(weights)

    def get_unique_stations(self):
        us = []
        for wmap in self.wavemaps:
            us.extend(wmap.get_station_names())
        return utility.unique_list(us)

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

    def assemble_results(
            self, point, chop_bounds=['a', 'd'], order='list',
            outmode='stacked_traces'):
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
        if point is None:
            raise ValueError('A point has to be provided!')

        logger.debug('Assembling seismic waveforms ...')

        syn_proc_traces, obs_proc_traces = self.get_synthetics(
            point, outmode=outmode,
            chop_bounds=chop_bounds, order='wmap')

        # will yield exactly the same as previous call needs wmap.prepare data
        # to be aware of taper_tolerance_factor
        syn_filt_traces, obs_filt_traces = self.get_synthetics(
            point, outmode=outmode, taper_tolerance_factor=0.,
            chop_bounds=chop_bounds, order='wmap')

        # from pyrocko import trace
        # trace.snuffle(syn_filt_traces+ obs_filt_traces)
        results = []
        for i, wmap in enumerate(self.wavemaps):
            wc = wmap.config
            at = wc.arrival_taper

            wmap_results = []
            for j, obs_tr in enumerate(obs_proc_traces[i]):

                dtrace_proc = obs_tr.copy()

                dtrace_proc.set_ydata(
                    (obs_tr.get_ydata() - syn_proc_traces[i][j].get_ydata()))

                dtrace_filt = obs_filt_traces[i][j].copy()
                dtrace_filt.set_ydata(
                    (obs_filt_traces[i][j].get_ydata() -
                        syn_filt_traces[i][j].get_ydata()))

                taper = at.get_pyrocko_taper(
                    float(obs_tr.tmin - at.a))

                wmap_results.append(heart.SeismicResult(
                    processed_obs=obs_tr,
                    processed_syn=syn_proc_traces[i][j],
                    processed_res=dtrace_proc,
                    filtered_obs=obs_filt_traces[i][j],
                    filtered_syn=syn_filt_traces[i][j],
                    filtered_res=dtrace_filt,
                    taper=taper))

            if order == 'list':
                results.extend(wmap_results)

            elif order == 'wmap':
                results.append(wmap_results)

            else:
                raise ValueError('Order "%s" is not supported' % order)

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
        results = self.assemble_results(point, chop_bounds=['b', 'c'])
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

        self._mode = 'geometry'
        self.synthesizers = {}
        self.choppers = {}

        self.sources = sources

        self.correction_name = 'time_shift'

        self.config = sc

    def point2sources(self, point, input_depth='top'):
        """
        Updates the composite source(s) (in place) with the point values.

        Parameters
        ----------
        point : dict
            with random variables from solution space
        input_depth : string
            may be either 'top'- input coordinates are transformed to center
            'center' - input coordinates are not transformed
        """
        tpoint = copy.deepcopy(point)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source = self.sources[0]
        source_params = list(source.keys()) + list(source.stf.keys())

        for param in list(tpoint.keys()):
            if param not in source_params:
                tpoint.pop(param)

        tpoint['time'] += self.event.time

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(
                source, input_depth=input_depth, **source_points[i])

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
        hp_specific = self.config.dataset_specific_residual_noise_estimation
        tpoint = problem_config.get_test_point()

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            'Seismic optimization on: \n '
            ' %s' % ', '.join(self.input_rvs.keys()))

        t2 = time()
        wlogpts = []

        self.init_hierarchicals(problem_config)
        self.analyse_noise(tpoint)
        self.init_weights()
        if self.config.station_corrections:
            logger.info(
                'Initialized %i hierarchical parameters for '
                'station corrections.' % len(self.get_unique_stations()))

        for wmap in self.wavemaps:
            if len(self.hierarchicals) > 0:
                time_shifts = self.hierarchicals[
                    self.correction_name][wmap.station_correction_idxs]
                self.input_rvs[self.correction_name] = time_shifts

            wc = wmap.config

            logger.info(
                'Preparing data of "%s" for optimization' % wmap._mapid)
            wmap.prepare_data(
                source=self.event, engine=self.engine, outmode='array')

            logger.info(
                'Initializing synthesizer for "%s"' % wmap._mapid)
            self.synthesizers[wmap._mapid] = theanof.SeisSynthesizer(
                engine=self.engine,
                sources=self.sources,
                targets=wmap.targets,
                event=self.event,
                arrival_taper=wc.arrival_taper,
                arrival_times=wmap._arrival_times,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=self.config.pre_stack_cut,
                station_corrections=self.config.station_corrections)

            synths, _ = self.synthesizers[wmap._mapid](self.input_rvs)

            residuals = wmap.shared_data_array - synths

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
        outmode = kwargs.pop('outmode', 'stacked_traces')
        chop_bounds = kwargs.pop('chop_bounds', ['a', 'd'])
        order = kwargs.pop('order', 'list')

        self.point2sources(point)

        sc = self.config
        synths = []
        obs = []
        for wmap in self.wavemaps:
            wc = wmap.config

            wmap.prepare_data(
                source=self.event,
                engine=self.engine,
                outmode=outmode,
                chop_bounds=chop_bounds)

            arrival_times = wmap._arrival_times
            if self.config.station_corrections:
                try:
                    arrival_times += point[
                        self.correction_name][wmap.station_correction_idxs]
                except IndexError:  # got reference point from config
                    arrival_times += float(point[self.correction_name]) * \
                        num.ones(wmap.n_t)

            synthetics, _ = heart.seis_synthetics(
                engine=self.engine,
                sources=self.sources,
                targets=wmap.targets,
                arrival_taper=wc.arrival_taper,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=sc.pre_stack_cut,
                arrival_times=arrival_times,
                outmode=outmode,
                chop_bounds=chop_bounds,
                **kwargs)

            if order == 'list':
                synths.extend(synthetics)
                obs.extend(wmap._prepared_data)

            elif order == 'wmap':
                synths.append(synthetics)
                obs.append(wmap._prepared_data)

            else:
                raise ValueError('Order "%s" is not supported' % order)

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

        # update data covariances in case model dependend non-toeplitz
        if self.config.noise_estimator.structure == 'non-toeplitz':
            logger.info('Updating data-covariances ...')
            self.analyse_noise(point)

        crust_inds = range(*sc.gf_config.n_variations)
        thresh = 5
        if len(crust_inds) > thresh:
            logger.info('Updating seismic velocity model-covariances ...')
            if self.config.noise_estimator.structure == 'non-toeplitz':
                logger.warning(
                    'Non-toeplitz estimation in combination with model '
                    'prediction covariances is still EXPERIMENTAL and results'
                    ' should be interpreted with care!!')

            for wmap in self.wavemaps:
                wc = wmap.config

                arrival_times = wmap._arrival_times
                if self.config.station_corrections:
                    arrival_times += point[
                        self.correction_name][wmap.station_correction_idxs]

                for channel in wmap.channels:
                    tidxs = wmap.get_target_idxs([channel])
                    for station, tidx in zip(wmap.stations, tidxs):

                        logger.debug('Channel %s of Station %s ' % (
                            channel, station.station))

                        crust_targets = heart.init_seismic_targets(
                            stations=[station],
                            earth_model_name=sc.gf_config.earth_model_name,
                            channels=channel,
                            sample_rate=sc.gf_config.sample_rate,
                            crust_inds=crust_inds,
                            reference_location=sc.gf_config.reference_location)

                        cov_pv = cov.seismic_cov_velocity_models(
                            engine=self.engine,
                            sources=self.sources,
                            targets=crust_targets,
                            wavename=wmap.name,
                            arrival_taper=wc.arrival_taper,
                            arrival_time=arrival_times[tidx],
                            filterer=wc.filterer,
                            plot=plot, n_jobs=n_jobs)
                        cov_pv = utility.ensure_cov_psd(cov_pv)

                        self.engine.close_cashed_stores()

                        dataset = wmap.datasets[tidx]
                        dataset.covariance.pred_v = cov_pv

                        t0 = time()
                        choli = dataset.covariance.chol_inverse
                        t1 = time()
                        logger.debug('Calculate weight time %f' % (t1 - t0))
                        wmap.weights[tidx].set_value(choli)
                        dataset.covariance.update_slog_pdet()
        else:
            logger.info(
                'Not updating seismic velocity model-covariances because '
                'number of model variations is too low! < %i' % thresh)


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

        self._mode = 'ffo'
        self.gfpath = os.path.join(
            project_dir, self._mode, bconfig.linear_gf_dir_name)

        self.config = sc
        sgfc = sc.gf_config

        for pw, pl in zip(sgfc.patch_widths, sgfc.patch_lengths):
            if pw != pl:
                raise ValueError(
                    'So far only square patches supported in kinematic'
                    ' model! - fast_sweeping issues')

        if len(sgfc.reference_sources) > 1:
            raise ValueError(
                'So far only one reference plane supported! - '
                'fast_sweeping issues')

        self.fault = self.load_fault_geometry()
        # TODO: n_subfaultssupport
        n_p_dip, n_p_strike = self.fault.get_subfault_discretization(0)

        logger.info('Fault(s) discretized to %s [km]'
                    ' patches.' % utility.list2string(sgfc.patch_lengths))

        if not hypers:
            self.sweeper = theanof.Sweeper(
                sgfc.patch_lengths[0],
                n_p_dip,
                n_p_strike,
                self.sweep_implementation)

            for wmap in self.wavemaps:

                logger.info(
                    'Preparing data of "%s" for optimization' % wmap.name)
                wmap.prepare_data(
                    source=self.event, engine=self.engine, outmode='array')

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

        logger.info("Loading %s Green's Functions" % self.name)
        self.load_gfs(
            crust_inds=[self.config.gf_config.reference_model_idx],
            make_shared=False)

        hp_specific = self.config.dataset_specific_residual_noise_estimation
        tpoint = problem_config.get_test_point()

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            'Seismic optimization on: \n '
            ' %s' % ', '.join(self.input_rvs.keys()))

        t2 = time()
        wlogpts = []

        self.analyse_noise(tpoint)
        for gfs in self.gfs.values():
            gfs.init_optimization()

        self.init_weights()
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
        # TODO make nsubfaults ready
        nuc_dip_idx, nuc_strike_idx = self.fault.fault_locations2idxs(
            index=0,
            positions_dip=nuc_dip,
            positions_strike=nuc_strike,
            backend='theano')

        starttimes0 = self.sweeper(
            (1. / input_rvs['velocities']), nuc_dip_idx, nuc_strike_idx)

        starttimes0 += input_rvs['nucleation_time']
        wlogpts = []
        for wmap in self.wavemaps:
            # TODO: for subfault in range(self.fault.nsubfaults):

            # station corrections
            if len(self.hierarchicals) > 0:
                raise NotImplementedError(
                    'Station corrections not fully implemented! for FFO!')
                starttimes = (
                    tt.tile(starttimes0, wmap.n_t) +
                    tt.repeat(self.hierarchicals[self.correction_name][
                        wmap.station_correction_idxs],
                        self.fault.npatches)).reshape(
                            wmap.n_t, self.fault.npatches)

                targetidxs = shared(
                    num.atleast_2d(num.arange(wmap.n_t)).T, borrow=True)
            else:
                starttimes = starttimes0
                targetidxs = num.lib.index_tricks.s_[:]

            logger.debug('Stacking %s phase ...' % wmap.config.name)
            synthetics = tt.zeros(
                (wmap.n_t, wmap.config.arrival_taper.nsamples(
                    self.config.gf_config.sample_rate)),
                dtype=tconfig.floatX)

            # make sure data is init as array, if non-toeplitz above-traces!
            wmap.prepare_data(
                source=self.event, engine=self.engine, outmode='array')

            for var in self.slip_varnames:
                logger.debug('Stacking %s variable' % var)
                key = self.get_gflibrary_key(
                    crust_ind=ref_idx, wavename=wmap.name, component=var)
                synthetics += self.gfs[key].stack_all(
                    targetidxs=targetidxs,
                    starttimes=starttimes,
                    durations=input_rvs['durations'],
                    slips=input_rvs[var],
                    interpolation=wmap.config.interpolation)

            residuals = wmap.shared_data_array - synthetics

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

        outmode = kwargs.pop('outmode', 'stacked_traces')

        # GF library cut in between [b, c] no [a,d] possible
        chop_bounds = ['b', 'c']
        order = kwargs.pop('order', 'list')

        ref_idx = self.config.gf_config.reference_model_idx
        if len(self.gfs) == 0:
            self.load_gfs(
                crust_inds=[ref_idx],
                make_shared=False)

        for gfs in self.gfs.values():
            gfs.set_stack_mode('numpy')

        tpoint = copy.deepcopy(point)

        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        # TODO make nsubfaults ready
        nuc_dip_idx, nuc_strike_idx = self.fault.fault_locations2idxs(
            index=0,
            positions_dip=tpoint['nucleation_dip'],
            positions_strike=tpoint['nucleation_strike'],
            backend='numpy')

        starttimes0 = self.fault.get_subfault_starttimes(
            index=0,
            rupture_velocities=tpoint['velocities'],
            nuc_dip_idx=nuc_dip_idx,
            nuc_strike_idx=nuc_strike_idx).flatten()

        starttimes0 += point['nucleation_time']

        # station corrections
        if len(self.hierarchicals) > 0:
            raise NotImplementedError(
                'Station corrections not fully implemented! for FFO!')
            # starttimes = (
            #    num.tile(starttimes0, wmap.n_t) +
            #    num.repeat(self.hierarchicals[self.correction_name][
            #        wmap.station_correction_idxs],
            #        self.fault.npatches)).reshape(
            #            wmap.n_t, self.fault.npatches)
            #
            # targetidxs = num.atleast_2d(num.arange(wmap.n_t)).T
        else:
            starttimes = starttimes0
            targetidxs = num.lib.index_tricks.s_[:]

        # obsolete from variable obs data, patchidx = self.fault.patchmap(
        #    index=0, dipidx=nuc_dip_idx, strikeidx=nuc_strike_idx)

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
                    targetidxs=targetidxs,
                    starttimes=starttimes,
                    durations=tpoint['durations'],
                    slips=tpoint[var],
                    interpolation=wmap.config.interpolation)

            wmap_synthetics = []
            for i, target in enumerate(wmap.targets):
                tr = Trace(
                    ydata=synthetics[i, :],
                    tmin=float(
                        gflibrary.reference_times[i]),
                    deltat=gflibrary.deltat)

                tr.set_codes(*target.codes)
                wmap_synthetics.append(tr)

            wmap.prepare_data(
                source=self.event,
                engine=self.engine,
                outmode=outmode,
                chop_bounds=chop_bounds)

            if order == 'list':
                synth_traces.extend(wmap_synthetics)
                obs_traces.extend(wmap._prepared_data)

            elif order == 'wmap':
                synth_traces.append(wmap_synthetics)
                obs_traces.append(wmap._prepared_data)

            else:
                raise ValueError('Order "%s" is not supported' % order)

        return synth_traces, obs_traces

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """

        # update data covariances in case model dependend non-toeplitz
        if self.config.noise_estimator.structure == 'non-toeplitz':
            logger.info('Updating data-covariances ...')
            self.analyse_noise(point)
