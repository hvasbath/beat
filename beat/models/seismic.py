import copy
import os
from collections import OrderedDict
from logging import getLogger
from time import time

import numpy as num
import theano.tensor as tt
from pymc3 import Deterministic, Uniform
from pyrocko.gf import LocalEngine
from pyrocko.trace import Trace
from theano import config as tconfig
from theano import shared
from theano.printing import Print
from theano.tensor import fft

from beat import config as bconfig
from beat import covariance as cov
from beat import heart, theanof, utility
from beat.ffi import get_gf_prefix, load_gf_library
from beat.models.base import (
    Composite,
    ConfigInconsistentError,
    FaultGeometryNotFoundError,
    get_hypervalue_from_point,
)
from beat.models.distributions import get_hyper_name, multivariate_normal_chol

logger = getLogger("seismic")


__all__ = ["SeismicGeometryComposite", "SeismicDistributerComposite"]


class SeismicComposite(Composite):
    """
    Comprises how to solve the non-linear seismic forward model.

    Parameters
    ----------
    sc : :class:`config.SeismicConfig`
        configuration object containing seismic setup parameters
    events: list
        of :class:`pyrocko.model.Event`
    project_dir : str
        directory of the model project, where to find the data
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    _datasets = None
    _weights = None
    _targets = None
    _hierarchicalnames = None

    def __init__(self, sc, events, project_dir, hypers=False):

        super(SeismicComposite, self).__init__(events)

        logger.debug("Setting up seismic structure ...\n")
        self.name = "seismic"
        self._like_name = "seis_like"
        self.correction_name = "time_shift"

        self.engine = LocalEngine(store_superdirs=[sc.gf_config.store_superdir])

        if sc.responses_path is not None:
            responses_path = os.path.join(sc.responses_path, bconfig.response_file_name)
        else:
            responses_path = sc.responses_path

        # load data
        self.datahandlers = []
        for i in range(self.nevents):
            seismic_data_path = os.path.join(
                project_dir, bconfig.multi_event_seismic_data_name(i)
            )

            logger.info(
                "Loading seismic data for event %i"
                " from: %s " % (i, seismic_data_path)
            )
            self.datahandlers.append(
                heart.init_datahandler(
                    seismic_config=sc,
                    seismic_data_path=seismic_data_path,
                    responses_path=responses_path,
                )
            )

        self.noise_analyser = cov.SeismicNoiseAnalyser(
            structure=sc.noise_estimator.structure,
            pre_arrival_time=sc.noise_estimator.pre_arrival_time,
            engine=self.engine,
            events=self.events,
            chop_bounds=["b", "c"],
        )

        self.wavemaps = []
        for i, wc in enumerate(sc.waveforms):
            logger.info('Initialising seismic wavemap for "%s" ...' % wc.name)
            if wc.include:
                wmap = heart.init_wavemap(
                    waveformfit_config=wc,
                    datahandler=self.datahandlers[wc.event_idx],
                    event=self.events[wc.event_idx],
                    mapnumber=i,
                )

                self.wavemaps.append(wmap)
            else:
                logger.info(
                    'The waveform defined in "%s %i" config is not '
                    "included in the optimization!" % (wc.name, i)
                )

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(
                    shared(num.array([1.0]), name="seis_llk_%i" % t, borrow=True)
                )

    def _hyper2wavemap(self, hypername):

        dummy = "_".join(hypername.split("_")[1:-1])
        for wmap in self.wavemaps:
            if wmap._mapid == dummy:
                return wmap

        raise ValueError("No waveform mapping found for hyperparameter! %s" % hypername)

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
        if self.config.dataset_specific_residual_noise_estimation:
            wmap = self._hyper2wavemap(hp_name)
            return wmap.hypersize
        else:
            return 1

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__.copy()

    def analyse_noise(self, tpoint=None, chop_bounds=["b", "c"]):
        """
        Analyse seismic noise in datatraces and set
        data-covariance matrixes accordingly.
        """
        if self.config.noise_estimator.structure == "non-toeplitz":
            results = self.assemble_results(
                tpoint, order="wmap", chop_bounds=chop_bounds
            )
        else:
            results = [None] * len(self.wavemaps)

        for wmap, wmap_results in zip(self.wavemaps, results):
            logger.info(
                'Retrieving seismic data-covariances with structure "%s" '
                "for %s ..." % (self.config.noise_estimator.structure, wmap._mapid)
            )

            cov_ds_seismic = self.noise_analyser.get_data_covariances(
                wmap=wmap,
                results=wmap_results,
                sample_rate=self.config.gf_config.sample_rate,
                chop_bounds=chop_bounds,
            )

            for j, trc in enumerate(wmap.datasets):
                if trc.covariance is None:
                    trc.covariance = heart.Covariance(data=cov_ds_seismic[j])
                else:
                    trc.covariance.data = cov_ds_seismic[j]

                if int(trc.covariance.data.sum()) == trc.data_len():
                    logger.warning(
                        "Data covariance is identity matrix!" " Please double check!!!"
                    )

    def init_hierarchicals(self, problem_config):
        """
        Initialise random variables for temporal station corrections.
        """
        hierarchicals = problem_config.hierarchicals
        self._hierarchicalnames = []
        nwmaps = len(self.wavemaps)
        nspecwmaps = num.sum(
            [1 for wmap in self.wavemaps if wmap.config.domain == "spectrum"]
        )
        if (
            not self.config.station_corrections
            and self.correction_name in hierarchicals
        ):
            raise ConfigInconsistentError(
                "Station corrections disabled, but they are defined"
                " in the problem configuration!"
            )

        if (
            self.config.station_corrections
            and self.correction_name not in hierarchicals
        ):
            raise ConfigInconsistentError(
                "Station corrections enabled, but they are not defined"
                " in the problem configuration!"
            )

        if (
            self.config.station_corrections
            and self.correction_name in hierarchicals
            and nwmaps == nspecwmaps
        ):
            raise ConfigInconsistentError(
                "Station corrections enabled, and they are defined"
                " in the problem configuration, but they are not required"
                " it's only spectra!"
            )

        if self.correction_name in hierarchicals:
            logger.info("Estimating time shift for each station and waveform map...")
            for wmap in self.wavemaps:
                hierarchical_name = wmap.time_shifts_id
                nhierarchs = len(wmap.get_station_names())
                if wmap.config.domain == "spectrum":
                    logger.info(
                        '%s got fixed at "0.0" for spectra' % (hierarchical_name)
                    )
                    self.hierarchicals[hierarchical_name] = num.zeros(
                        (nhierarchs), dtype="int16"
                    )
                else:
                    logger.info(
                        "For %s with %i shifts" % (hierarchical_name, nhierarchs)
                    )

                    if hierarchical_name in hierarchicals:
                        logger.info(
                            "Using wavemap specific imported:"
                            " %s " % hierarchical_name
                        )
                        param = hierarchicals[hierarchical_name]
                    else:
                        logger.info("Using global %s" % self.correction_name)
                        param = copy.deepcopy(
                            problem_config.hierarchicals[self.correction_name]
                        )
                        param.lower = num.repeat(param.lower, nhierarchs)
                        param.upper = num.repeat(param.upper, nhierarchs)
                        param.testvalue = num.repeat(param.testvalue, nhierarchs)

                    if hierarchical_name not in self.hierarchicals:
                        if not num.array_equal(param.lower, param.upper):
                            kwargs = dict(
                                name=hierarchical_name,
                                shape=param.dimension,
                                lower=param.lower,
                                upper=param.upper,
                                testval=param.testvalue,
                                transform=None,
                                dtype=tconfig.floatX,
                            )

                            try:
                                self.hierarchicals[hierarchical_name] = Uniform(
                                    **kwargs
                                )
                            except TypeError:
                                kwargs.pop("name")
                                self.hierarchicals[hierarchical_name] = Uniform.dist(
                                    **kwargs
                                )

                            self._hierarchicalnames.append(hierarchical_name)
                        else:
                            logger.info(
                                "not solving for %s, got fixed at %s"
                                % (
                                    param.name,
                                    utility.list2string(param.lower.flatten()),
                                )
                            )
                            self.hierarchicals[hierarchical_name] = param.lower

    def export(
        self,
        point,
        results_path,
        stage_number,
        fix_output=False,
        force=False,
        update=False,
        chop_bounds=["b", "c"],
    ):
        """
        Save results for given point to result path.
        """

        def save_covs(wmap, cov_mat="pred_v"):
            """
            Save covariance matrixes of given attribute
            """

            covs = {
                dataset.nslcd_id_str: getattr(dataset.covariance, cov_mat)
                for dataset in wmap.datasets
            }

            outname = os.path.join(
                results_path, "%s_C_%s_%s" % ("seismic", cov_mat, wmap._mapid)
            )
            logger.info('"%s" to: %s' % (wmap._mapid, outname))
            num.savez(outname, **covs)

        from pyrocko import io

        # synthetics and data
        results = self.assemble_results(point, chop_bounds=chop_bounds)
        for traces, attribute in heart.results_for_export(
            results=results, datatype="seismic"
        ):

            filename = "%s_%i.mseed" % (attribute, stage_number)
            outpath = os.path.join(results_path, filename)
            try:
                io.save(traces, outpath, overwrite=force)
            except io.mseed.CodeTooLong:
                if fix_output:
                    for tr in traces:
                        tr.set_station(tr.station[-5::])
                        tr.set_location(str(self.config.gf_config.reference_model_idx))

                    io.save(traces, outpath, overwrite=force)
                else:
                    raise ValueError(
                        "Some station codes are too long! "
                        "(the --fix_output option will truncate to "
                        "last 5 characters!)"
                    )

        # export stdz residuals
        self.analyse_noise(point, chop_bounds=chop_bounds)
        if update:
            logger.info("Saving velocity model covariance matrixes...")
            self.update_weights(point, chop_bounds=chop_bounds)
            for wmap in self.wavemaps:
                save_covs(wmap, "pred_v")

        logger.info("Saving data covariance matrixes...")
        for wmap in self.wavemaps:
            save_covs(wmap, "data")

    def init_weights(self):
        """
        Initialise shared weights in wavemaps.
        """
        logger.info("Initialising weights ...")
        for wmap in self.wavemaps:
            weights = []
            for j, trc in enumerate(wmap.datasets):
                icov = trc.covariance.chol_inverse
                weights.append(
                    shared(
                        icov, name="seis_%s_weight_%i" % (wmap._mapid, j), borrow=True
                    )
                )

            wmap.add_weights(weights=weights)

    def get_all_station_names(self):
        """
        Returns list of station names in the order of wavemaps.
        """
        us = []
        for wmap in self.wavemaps:
            us.extend(wmap.get_station_names())

        return us

    def get_unique_time_shifts_ids(self):
        """
        Return unique time_shifts ids from wavemaps, which are keys to
        hierarchical RVs of station corrections
        """
        ts = []
        for wmap in self.wavemaps:
            ts.append(wmap.time_shifts_id)

        return utility.unique_list(ts)

    def get_unique_station_names(self):
        """
        Return unique station names from all wavemaps
        """
        return utility.unique_list(self.get_all_station_names())

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
        if self._weights is None or len(self._weights) == 0:
            ws = []
            for wmap in self.wavemaps:
                if wmap.weights:
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
        self,
        point,
        chop_bounds=["a", "d"],
        order="list",
        outmode="stacked_traces",
        force=True,
    ):
        """
        Assemble seismic traces for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        force : bool
            force preparation of data with input params otherwise cached is
            used

        Returns
        -------
        List with :class:`heart.SeismicResult`
        """
        if point is None:
            raise ValueError("A point has to be provided!")

        logger.debug("Assembling seismic waveforms ...")

        syn_proc_traces, obs_proc_traces = self.get_synthetics(
            point, outmode=outmode, chop_bounds=chop_bounds, order="wmap", force=force
        )

        results = []
        for i, wmap in enumerate(self.wavemaps):
            wc = wmap.config
            at = wc.arrival_taper

            wmap_results = []
            for j, obs_tr in enumerate(obs_proc_traces[i]):

                taper = at.get_pyrocko_taper(float(obs_tr.tmin - at.a))

                if outmode != "tapered_data":
                    source_contributions = [syn_proc_traces[i][j]]
                else:
                    source_contributions = syn_proc_traces[i][j]

                wmap_results.append(
                    heart.SeismicResult(
                        point=point,
                        processed_obs=obs_tr,
                        source_contributions=source_contributions,
                        taper=taper,
                        filterer=wmap.config.filterer,
                        domain=wc.domain,
                    )
                )

            if order == "list":
                results.extend(wmap_results)

            elif order == "wmap":
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
        results = self.assemble_results(point, chop_bounds=["b", "c"])
        for k, result in enumerate(results):
            choli = self.datasets[k].covariance.chol_inverse
            tmp = choli.dot(result.processed_res.ydata)
            _llk = num.asarray([num.dot(tmp, tmp)])
            self._llks[k].set_value(_llk)

    def get_standardized_residuals(
        self, point, chop_bounds=["b", "c"], results=None, weights=None
    ):
        """
        Parameters
        ----------
        point : dict
            with parameters to point in solution space to calculate
            standardized residuals

        Returns
        -------
        dict of arrays of standardized residuals,
            keys are nslc_ids
        """

        if results is None:
            results = self.assemble_results(
                point, order="list", chop_bounds=chop_bounds
            )

        if weights is None:
            self.update_weights(point, chop_bounds=chop_bounds)

        counter = utility.Counter()
        hp_specific = self.config.dataset_specific_residual_noise_estimation

        stdz_residuals = OrderedDict()
        for dataset, result, target in zip(self.datasets, results, self.targets):

            hp = get_hypervalue_from_point(
                point, dataset, counter, hp_specific=hp_specific
            )
            ydata = result.processed_res.get_ydata()
            choli = num.linalg.inv(dataset.covariance.chol(num.exp(hp * 2.0)))
            stdz_residuals[target.nslcd_id_str] = choli.dot(ydata)
        return stdz_residuals

    def get_variance_reductions(
        self, point, results=None, weights=None, chop_bounds=["a", "d"]
    ):
        """
        Parameters
        ----------
        point : dict
            with parameters to point in solution space to calculate
            variance reductions

        Returns
        -------
        dict of floats,
            keys are nslc_ids
        """

        if results is None:
            results = self.assemble_results(
                point, order="list", chop_bounds=chop_bounds
            )

        ndatasets = len(self.datasets)

        assert len(results) == ndatasets

        if weights is None:
            self.analyse_noise(point, chop_bounds=chop_bounds)
            self.update_weights(point, chop_bounds=chop_bounds)
            weights = self.weights

        nweights = len(weights)

        assert nweights == ndatasets

        logger.debug("n weights %i , n datasets %i" % (nweights, ndatasets))

        logger.debug("Calculating variance reduction for solution ...")

        counter = utility.Counter()
        hp_specific = self.config.dataset_specific_residual_noise_estimation

        var_reds = OrderedDict()
        for result, tr in zip(results, self.datasets):
            nslcd_id_str = result.processed_obs.nslcd_id_str

            hp = get_hypervalue_from_point(point, tr, counter, hp_specific=hp_specific)
            icov = tr.covariance.inverse(num.exp(hp * 2.0))

            data = result.processed_obs.get_ydata()
            residual = result.processed_res.get_ydata()

            nom = residual.T.dot(icov).dot(residual)
            denom = data.T.dot(icov).dot(data)

            logger.debug("nom %f, denom %f" % (float(nom), float(denom)))

            var_reds[nslcd_id_str] = float(1 - (nom / denom))

            logger.debug(
                "Variance reduction for %s is %f"
                % (nslcd_id_str, var_reds[nslcd_id_str])
            )

            if 0:
                from matplotlib import pyplot as plt

                fig, ax = plt.subplots(1, 1)
                im = ax.imshow(tr.covariance.data)
                plt.colorbar(im)
                plt.show()

        return var_reds


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
    events : list
        of :class:`pyrocko.model.Event`
        contains information of reference event(s), coordinates of reference
        point(s) and source time(s)
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, sc, project_dir, sources, events, hypers=False):

        super(SeismicGeometryComposite, self).__init__(
            sc, events, project_dir, hypers=hypers
        )

        self._mode = "geometry"
        self.synthesizers = {}
        self.choppers = {}

        self.sources = sources

        self.correction_name = "time_shift"

        self.config = sc

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.

        Parameters
        ----------
        point : dict
            with random variables from solution space
        """
        tpoint = copy.deepcopy(point)
        tpoint.update(self.fixed_rvs)
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

        # update source times
        if "time" in tpoint:
            if self.nevents == 1:
                tpoint["time"] += self.event.time  # single event
            else:
                for i, event in enumerate(self.events):  # multi event
                    tpoint["time"][i] += event.time

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Get seismic likelihood formula for the model built. Has to be called
        within a with model context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution` of source parameters
        fixed_rvs : dict
            of :class:`numpy.array`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`
        problem_config : :class:`config.ProblemConfig`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        chop_bounds = ["b", "c"]  # we want llk calculation only between b c

        hp_specific = self.config.dataset_specific_residual_noise_estimation
        tpoint = problem_config.get_test_point()

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            "Seismic optimization on: \n " " %s" % ", ".join(self.input_rvs.keys())
        )

        self.input_rvs.update(fixed_rvs)

        t2 = time()
        wlogpts = []

        self.init_hierarchicals(problem_config)
        self.analyse_noise(tpoint, chop_bounds=chop_bounds)
        self.init_weights()
        if self.config.station_corrections:
            logger.info(
                "Initialized %i hierarchical parameters for "
                "station corrections." % len(self.get_all_station_names())
            )

        for wmap in self.wavemaps:
            if len(self.hierarchicals) > 0:
                time_shifts = self.hierarchicals[wmap.time_shifts_id][
                    wmap.station_correction_idxs
                ]
                self.input_rvs[self.correction_name] = time_shifts

            wc = wmap.config

            logger.info('Preparing data of "%s" for optimization' % wmap._mapid)
            wmap.prepare_data(
                source=self.events[wc.event_idx],
                engine=self.engine,
                outmode="array",
                chop_bounds=chop_bounds,
            )

            logger.info('Initializing synthesizer for "%s"' % wmap._mapid)

            if self.nevents == 1:
                logger.info("Using all sources for wavemap %s !" % wmap._mapid)
                sources = self.sources
            else:
                logger.info(
                    "Using source based on event %i for wavemap %s!"
                    % (wc.event_idx, wmap._mapid)
                )
                sources = [self.sources[wc.event_idx]]

            self.synthesizers[wmap._mapid] = theanof.SeisSynthesizer(
                engine=self.engine,
                sources=sources,
                targets=wmap.targets,
                event=self.events[wc.event_idx],
                arrival_taper=wc.arrival_taper,
                arrival_times=wmap._arrival_times,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=self.config.pre_stack_cut,
                station_corrections=self.config.station_corrections,
                domain=wc.domain,
            )

            synths, _ = self.synthesizers[wmap._mapid](self.input_rvs)
            residuals = wmap.shared_data_array - synths

            logpts = multivariate_normal_chol(
                wmap.datasets,
                wmap.weights,
                hyperparams,
                residuals,
                hp_specific=hp_specific,
            )

            wlogpts.append(logpts)

        t3 = time()
        logger.debug("Teleseismic forward model on test model takes: %f" % (t3 - t2))

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
        outmode = kwargs.pop("outmode", "stacked_traces")
        chop_bounds = kwargs.pop("chop_bounds", ["a", "d"])
        order = kwargs.pop("order", "list")
        nprocs = kwargs.pop("nprocs", 4)
        force = kwargs.pop("force", False)

        self.point2sources(point)

        sc = self.config
        synths = []
        obs = []
        for wmap in self.wavemaps:
            wc = wmap.config
            if not wmap.is_prepared or force:
                wmap.prepare_data(
                    source=self.events[wc.event_idx],
                    engine=self.engine,
                    outmode="stacked_traces",  # no source individual contribs
                    chop_bounds=chop_bounds,
                )

            arrival_times = copy.deepcopy(wmap._arrival_times)
            if self.config.station_corrections and wc.domain == "time":
                try:
                    arrival_times += point[wmap.time_shifts_id][
                        wmap.station_correction_idxs
                    ]
                except KeyError:  # got reference point from config
                    if self.correction_name in point:
                        arrival_times += float(point[self.correction_name]) * num.ones(
                            wmap.n_t
                        )
                    else:  # fixed individual station corrections
                        arrival_times += self.hierarchicals[wmap.time_shifts_id][
                            wmap.station_correction_idxs
                        ]

            if self.nevents == 1:
                logger.debug("Using all sources for each wavemap!")
                sources = self.sources
            else:
                logger.debug(
                    "Using individual sources based on event index " "for each wavemap!"
                )
                sources = [self.sources[wc.event_idx]]

            synthetics, _ = heart.seis_synthetics(
                engine=self.engine,
                sources=sources,
                targets=wmap.targets,
                arrival_taper=wc.arrival_taper,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=sc.pre_stack_cut,
                arrival_times=arrival_times,
                outmode=outmode,
                chop_bounds=chop_bounds,
                nprocs=nprocs,
                # plot=True,
                **kwargs
            )

            if self.config.station_corrections and wc.domain == "time":
                # set tmin to data tmin
                for tr, dtr in zip(synthetics, wmap._prepared_data):
                    if isinstance(tr, list):
                        for t in tr:
                            t.tmin = dtr.tmin
                            t.tmax = dtr.tmax
                    else:
                        tr.tmin = dtr.tmin
                        tr.tmax = dtr.tmax

            if wc.domain == "spectrum":

                valid_spectrum_indices = wmap.get_valid_spectrum_indices(
                    chop_bounds=chop_bounds, pad_to_pow2=True
                )

                synthetics = heart.fft_transforms(
                    synthetics,
                    valid_spectrum_indices=valid_spectrum_indices,
                    outmode=outmode,
                    pad_to_pow2=True,
                )

            if order == "list":
                synths.extend(synthetics)
                obs.extend(wmap._prepared_data)

            elif order == "wmap":
                synths.append(synthetics)
                obs.append(wmap._prepared_data)

            else:
                raise ValueError('Order "%s" is not supported' % order)

        return synths, obs

    def update_weights(self, point, n_jobs=1, plot=False, chop_bounds=["b", "c"]):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        if not self.weights:
            self.init_weights()

        sc = self.config

        self.point2sources(point)

        # update data covariances in case model dependent non-toeplitz
        if self.config.noise_estimator.structure == "non-toeplitz":
            logger.info("Updating data-covariances ...")
            self.analyse_noise(point, chop_bounds=chop_bounds)

        crust_inds = range(*sc.gf_config.n_variations)
        thresh = 5
        if len(crust_inds) > thresh:
            logger.info("Updating seismic velocity model-covariances ...")
            if self.config.noise_estimator.structure == "non-toeplitz":
                logger.warning(
                    "Non-toeplitz estimation in combination with model "
                    "prediction covariances is still EXPERIMENTAL and results"
                    " should be interpreted with care!!"
                )

            for wmap in self.wavemaps:
                wc = wmap.config

                arrival_times = wmap._arrival_times
                if self.config.station_corrections:
                    arrival_times += point[wmap.time_shifts_id][
                        wmap.station_correction_idxs
                    ]

                for channel in wmap.channels:
                    tidxs = wmap.get_target_idxs([channel])
                    for station, tidx in zip(wmap.stations, tidxs):

                        logger.debug(
                            "Channel %s of Station %s " % (channel, station.station)
                        )

                        crust_targets = heart.init_seismic_targets(
                            stations=[station],
                            earth_model_name=sc.gf_config.earth_model_name,
                            channels=channel,
                            sample_rate=sc.gf_config.sample_rate,
                            crust_inds=crust_inds,
                            reference_location=sc.gf_config.reference_location,
                        )

                        t0 = time()
                        cov_pv = cov.seismic_cov_velocity_models(
                            engine=self.engine,
                            sources=self.sources,
                            targets=crust_targets,
                            wavename=wmap.name,
                            arrival_taper=wc.arrival_taper,
                            arrival_time=arrival_times[tidx],
                            filterer=wc.filterer,
                            chop_bounds=chop_bounds,
                            plot=plot,
                            n_jobs=n_jobs,
                        )
                        t1 = time()
                        logger.debug(
                            "%s: Calculate weight time %f"
                            % (station.station, (t1 - t0))
                        )
                        cov_pv = utility.ensure_cov_psd(cov_pv)

                        self.engine.close_cashed_stores()

                        dataset = wmap.datasets[tidx]
                        dataset.covariance.pred_v = cov_pv

        else:
            logger.info(
                "Not updating seismic velocity model-covariances because "
                "number of model variations is too low! < %i" % thresh
            )

        for wmap in self.wavemaps:
            logger.info("Updating weights of wavemap %s" % wmap._mapid)
            for i, dataset in enumerate(wmap.datasets):
                choli = dataset.covariance.chol_inverse

                # update shared variables
                dataset.covariance.update_slog_pdet()
                wmap.weights[i].set_value(choli)


class SeismicDistributerComposite(SeismicComposite):
    """
    Comprises how to solve the seismic (kinematic) linear forward model.
    Distributed slip
    """

    def __init__(self, sc, project_dir, events, hypers=False):

        super(SeismicDistributerComposite, self).__init__(
            sc, events, project_dir, hypers=hypers
        )

        self.gfs = {}
        self.gf_names = {}
        self.choppers = {}
        self.sweep_implementation = "c"

        self._mode = "ffi"
        self.gfpath = os.path.join(project_dir, self._mode, bconfig.linear_gf_dir_name)

        self.config = sc
        dgc = sc.gf_config.discretization_config

        for pw, pl in zip(dgc.patch_widths, dgc.patch_lengths):
            if pw != pl:
                raise ValueError(
                    "So far only square patches supported in kinematic"
                    " model! - fast_sweeping issues"
                )

        if len(sc.gf_config.reference_sources) > 1:
            logger.warning(
                "So far only rupture propagation on each subfault individually"
            )

        self.fault = self.load_fault_geometry()

        logger.info(
            "Fault(s) discretized to %s [km]"
            " patches." % utility.list2string(dgc.patch_lengths)
        )

        if not hypers:
            self.sweepers = []
            for idx in range(self.fault.nsubfaults):
                n_p_dip, n_p_strike = self.fault.ordering.get_subfault_discretization(
                    idx
                )

                self.sweepers.append(
                    theanof.Sweeper(
                        dgc.patch_lengths[idx],
                        n_p_dip,
                        n_p_strike,
                        self.sweep_implementation,
                    )
                )

    def load_fault_geometry(self):
        """
        Load fault-geometry, i.e. discretized patches.

        Returns
        -------
        :class:`heart.FaultGeometry`
        """
        try:
            return utility.load_objects(
                os.path.join(self.gfpath, bconfig.fault_geometry_name)
            )[0]
        except Exception:
            raise FaultGeometryNotFoundError()

    def point2sources(self, point):
        """
        Returns the fault source patche(s) with the point values updated.

        Parameters
        ----------
        point : dict
            with random variables from solution space
        """
        tpoint = copy.deepcopy(point)
        tpoint.update(self.fixed_rvs)

        if self.nevents == 1:
            events = [self.event]  # single event
        else:
            events = self.events  # multi event

        return self.fault.point2sources(tpoint, events=events)

    def get_gflibrary_key(self, crust_ind, wavename, component):
        return "%i_%s_%s" % (crust_ind, wavename, component)

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
            raise TypeError("crust_inds need to be a list!")

        if crust_inds is None:
            crust_inds = range(*self.config.gf_config.n_variations)

        for wmap in self.wavemaps:
            for crust_ind in crust_inds:
                gfs = {}
                for var in self.slip_varnames:
                    gflib_name = get_gf_prefix(
                        datatype=self.name,
                        component=var,
                        wavename=wmap._mapid,
                        crust_ind=crust_ind,
                    )
                    gfpath = os.path.join(self.gfpath, gflib_name + ".yaml")

                    if not os.path.exists(gfpath):
                        filename = get_gf_prefix(
                            datatype=self.name,
                            component=var,
                            wavename=wmap.config.name,
                            crust_ind=crust_ind,
                        )
                        logger.warning(
                            "Seismic GFLibrary %s does not exist, "
                            "trying to load with old naming: %s"
                            % (gflib_name, filename)
                        )
                        gfpath = os.path.join(self.gfpath, filename + ".yaml")

                    else:
                        logger.info("Loading SeismicGFLibrary %s " % gflib_name)
                        filename = gflib_name

                    gfs = load_gf_library(directory=self.gfpath, filename=filename)

                    if make_shared:
                        gfs.init_optimization()

                    key = self.get_gflibrary_key(
                        crust_ind=crust_ind, wavename=wmap._mapid, component=var
                    )

                    self.gf_names[key] = gfpath
                    self.gfs[key] = gfs

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):

        # no a, d taper bounds as GF library saved between b c
        chop_bounds = ["b", "c"]

        logger.info("Loading %s Green's Functions" % self.name)
        self.load_gfs(
            crust_inds=[self.config.gf_config.reference_model_idx], make_shared=False
        )

        hp_specific = self.config.dataset_specific_residual_noise_estimation
        tpoint = problem_config.get_test_point()

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            "Seismic optimization on: \n " " %s" % ", ".join(self.input_rvs.keys())
        )

        t2 = time()
        wlogpts = []

        self.analyse_noise(tpoint, chop_bounds=chop_bounds)
        for gfs in self.gfs.values():
            gfs.init_optimization()

        self.init_weights()
        self.init_hierarchicals(problem_config)
        if self.config.station_corrections:
            logger.info(
                "Initialized %i hierarchical parameters for "
                "station corrections." % len(self.get_all_station_names())
            )

        self.input_rvs.update(fixed_rvs)

        ref_idx = self.config.gf_config.reference_model_idx

        nuc_strike = input_rvs["nucleation_strike"]
        nuc_dip = input_rvs["nucleation_dip"]

        t2 = time()
        # convert velocities to rupture onset
        logger.debug("Fast sweeping ...")
        starttimes0 = tt.zeros((self.fault.npatches), dtype=tconfig.floatX)
        for index in range(self.fault.nsubfaults):
            nuc_dip_idx, nuc_strike_idx = self.fault.fault_locations2idxs(
                index=index,
                positions_dip=nuc_dip[index],
                positions_strike=nuc_strike[index],
                backend="theano",
            )

            sf_patch_indexs = self.fault.cum_subfault_npatches[index : index + 2]
            starttimes_tmp = self.sweepers[index](
                (1.0 / self.fault.vector2subfault(index, input_rvs["velocities"])),
                nuc_dip_idx,
                nuc_strike_idx,
            )

            starttimes_tmp += input_rvs["time"][index]
            starttimes0 = tt.set_subtensor(
                starttimes0[sf_patch_indexs[0] : sf_patch_indexs[1]], starttimes_tmp
            )

        wlogpts = []
        for wmap in self.wavemaps:
            wc = wmap.config
            if wc.domain == "spectrum":
                raise TypeError("FFI is currently only supported for time-domain!")

            # station corrections
            if len(self.hierarchicals) > 0:
                logger.info("Applying station corrections ...")
                starttimes = (
                    tt.tile(starttimes0, wmap.n_t)
                    - tt.repeat(
                        self.hierarchicals[wmap.time_shifts_id][
                            wmap.station_correction_idxs
                        ],
                        self.fault.npatches,
                    )
                ).reshape((wmap.n_t, self.fault.npatches))
            else:
                logger.info("No station corrections ...")
                starttimes = tt.tile(starttimes0, wmap.n_t).reshape(
                    (wmap.n_t, self.fault.npatches)
                )

            targetidxs = shared(num.atleast_2d(num.arange(wmap.n_t)).T, borrow=True)

            logger.debug("Stacking %s phase ..." % wc.name)
            synthetics = tt.zeros(
                (
                    wmap.n_t,
                    wc.arrival_taper.nsamples(self.config.gf_config.sample_rate),
                ),
                dtype=tconfig.floatX,
            )

            # make sure data is init as array, if non-toeplitz above-traces!
            wmap.prepare_data(
                source=self.events[wc.event_idx],
                engine=self.engine,
                outmode="array",
                chop_bounds=chop_bounds,
            )

            for var in self.slip_varnames:
                logger.debug("Stacking %s variable" % var)
                key = self.get_gflibrary_key(
                    crust_ind=ref_idx, wavename=wmap._mapid, component=var
                )
                logger.debug("GF Library key %s" % key)

                synthetics += self.gfs[key].stack_all(
                    targetidxs=targetidxs,
                    starttimes=starttimes,
                    durations=input_rvs["durations"],
                    slips=input_rvs[var],
                    interpolation=wc.interpolation,
                )

            residuals = wmap.shared_data_array - synthetics

            logger.debug("Calculating likelihoods ...")
            logpts = multivariate_normal_chol(
                wmap.datasets,
                wmap.weights,
                hyperparams,
                residuals,
                hp_specific=hp_specific,
            )

            wlogpts.append(logpts)

        t3 = time()
        logger.debug("Seismic formula on test model takes: %f" % (t3 - t2))

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
            outmode: stacked_traces/ tapered_data/ array

        Returns
        -------
        list with :class:`heart.SeismicDataset` synthetics for each target
        """

        outmode = kwargs.pop("outmode", "stacked_traces")
        patchidxs = kwargs.pop("patchidxs", None)

        if patchidxs is None:
            patchidxs = num.arange(self.fault.npatches, dtype="int")

        # GF library cut in between [b, c] no [a,d] possible
        chop_bounds = ["b", "c"]
        order = kwargs.pop("order", "list")

        ref_idx = self.config.gf_config.reference_model_idx
        if len(self.gfs) == 0:
            self.load_gfs(crust_inds=[ref_idx], make_shared=False)

        for gfs in self.gfs.values():
            gfs.set_stack_mode("numpy")

        tpoint = copy.deepcopy(point)

        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        starttimes0 = num.zeros((self.fault.npatches), dtype=tconfig.floatX)
        for index in range(self.fault.nsubfaults):
            starttimes_tmp = self.fault.point2starttimes(tpoint, index=index).ravel()

            sf_patch_indexs = self.fault.cum_subfault_npatches[index : index + 2]
            starttimes0[sf_patch_indexs[0] : sf_patch_indexs[1]] = starttimes_tmp

        synth_traces = []
        obs_traces = []
        for wmap in self.wavemaps:
            wc = wmap.config

            starttimes = num.tile(starttimes0, wmap.n_t).reshape(
                wmap.n_t, self.fault.npatches
            )

            # station corrections
            if self.config.station_corrections:
                logger.debug(
                    "Applying station corrections " "for wmap {}".format(wmap._mapid)
                )
                try:
                    corrections = point[wmap.time_shifts_id]
                except KeyError:  # got reference point from config
                    if self.correction_name in point:
                        corrections = float(point[self.correction_name]) * num.ones(
                            wmap.n_t
                        )
                    else:  # fixed individual station corrections
                        corrections = self.hierarchicals[wmap.time_shifts_id][
                            wmap.station_correction_idxs
                        ]

                starttimes -= num.repeat(
                    corrections[wmap.station_correction_idxs], self.fault.npatches
                ).reshape(wmap.n_t, self.fault.npatches)

            # TODO check targetidxs if station blacklisted!?
            targetidxs = num.atleast_2d(num.arange(wmap.n_t)).T

            synthetics = num.zeros(
                (wmap.n_t, wc.arrival_taper.nsamples(self.config.gf_config.sample_rate))
            )
            for var in self.slip_varnames:
                key = self.get_gflibrary_key(
                    crust_ind=ref_idx, wavename=wmap._mapid, component=var
                )
                try:
                    logger.debug("Accessing GF Library key %s" % key)
                    gflibrary = self.gfs[key]
                except KeyError:
                    raise KeyError(
                        "GF library %s not loaded! Loaded GFs:"
                        " %s" % (key, utility.list2string(self.gfs.keys()))
                    )
                from time import time

                gflibrary.set_stack_mode("numpy")

                t0 = time()
                synthetics += gflibrary.stack_all(
                    targetidxs=targetidxs,
                    starttimes=starttimes[:, patchidxs],
                    durations=tpoint["durations"][patchidxs],
                    slips=tpoint[var][patchidxs],
                    patchidxs=patchidxs,
                    interpolation=wc.interpolation,
                )
                t1 = time()
                logger.debug("{} seconds to stack {}".format((t1 - t0), wmap._mapid))

            wmap_synthetics = []
            if outmode != "array":
                for i, target in enumerate(wmap.targets):
                    tr = Trace(
                        ydata=synthetics[i, :],
                        tmin=float(gflibrary.reference_times[i]),
                        deltat=gflibrary.deltat,
                    )

                    tr.set_codes(*target.codes)

                    if outmode == "tapered_data":
                        # TODO subfault individual synthetics (use patchidxs arg)
                        tr = [tr]

                    wmap_synthetics.append(tr)

            elif outmode == "array":
                wmap_synthetics.extend(synthetics)
            else:
                raise ValueError(
                    "Supported outmodes: stacked_traces, tapered_data, array! "
                    "Given outmode: %s !" % outmode
                )

            if not wmap.is_prepared:
                wmap.prepare_data(
                    source=self.events[wc.event_idx],
                    engine=self.engine,
                    outmode=outmode,
                    chop_bounds=chop_bounds,
                )

            if order == "list":
                synth_traces.extend(wmap_synthetics)
                obs_traces.extend(wmap._prepared_data)

            elif order == "wmap":
                synth_traces.append(wmap_synthetics)
                obs_traces.append(wmap._prepared_data)

            else:
                raise ValueError('Order "%s" is not supported' % order)

        return synth_traces, obs_traces

    def update_weights(self, point, n_jobs=1, plot=False, chop_bounds=["b", "c"]):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        if not self.weights:
            self.init_weights()

        # update data covariances in case model dependent non-toeplitz
        if self.config.noise_estimator.structure == "non-toeplitz":
            logger.info("Updating data-covariances ...")
            self.analyse_noise(point, chop_bounds=chop_bounds)

        for wmap in self.wavemaps:
            logger.info("Updating weights of wavemap %s" % wmap._mapid)
            for i, dataset in enumerate(wmap.datasets):
                choli = dataset.covariance.chol_inverse

                # update shared variables
                dataset.covariance.update_slog_pdet()
                wmap.weights[i].set_value(choli)
