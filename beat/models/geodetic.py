import copy
import os
from collections import OrderedDict
from logging import getLogger
from time import time

import numpy as num
import theano.tensor as tt
from pymc3 import Deterministic, Uniform
from pyrocko.gf import LocalEngine, RectangularSource
from theano import config as tconfig
from theano import shared
from theano.printing import Print

from beat import config as bconfig
from beat import covariance as cov
from beat import heart, theanof, utility
from beat.ffi import get_gf_prefix, load_gf_library
from beat.interseismic import geo_backslip_synthetics, seperate_point
from beat.models.base import (
    Composite,
    ConfigInconsistentError,
    FaultGeometryNotFoundError,
    get_hypervalue_from_point,
)
from beat.models.distributions import multivariate_normal_chol

logger = getLogger("geodetic")


km = 1000.0


__all__ = [
    "GeodeticGeometryComposite",
    "GeodeticInterseismicComposite",
    "GeodeticDistributerComposite",
]


class GeodeticComposite(Composite):
    """
    Comprises data structure of the geodetic composite.

    Parameters
    ----------
    gc : :class:`config.GeodeticConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    events : list
        of :class:`pyrocko.model.Event`
        contains information of reference event(s), coordinates of reference
        point(s) and source time(s)
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    _hierarchicalnames = None
    weights = None

    def __init__(self, gc, project_dir, events, hypers=False):

        super(GeodeticComposite, self).__init__(events)

        logger.debug("Setting up geodetic structure ...\n")
        self.name = "geodetic"
        self._like_name = "geo_like"

        geodetic_data_path = os.path.join(project_dir, bconfig.geodetic_data_name)

        self.datasets = utility.load_objects(geodetic_data_path)
        logger.info("Number of geodetic datasets: %i " % self.n_t)

        # init geodetic targets
        self.targets = heart.init_geodetic_targets(
            datasets=self.datasets,
            earth_model_name=gc.gf_config.earth_model_name,
            interpolation=gc.interpolation,
            crust_inds=[gc.gf_config.reference_model_idx],
            sample_rate=gc.gf_config.sample_rate,
        )

        # merge geodetic data to calculate residuals on single array
        datasets, los_vectors, odws, self.Bij = heart.concatenate_datasets(
            self.datasets
        )
        logger.info("Number of geodetic data points: %i " % self.Bij.ordering.size)

        self.sdata = shared(datasets, name="geodetic_data", borrow=True)
        self.slos_vectors = shared(los_vectors, name="los_vecs", borrow=True)
        self.sodws = shared(odws, name="odws", borrow=True)

        self.noise_analyser = cov.GeodeticNoiseAnalyser(
            config=gc.noise_estimator, events=self.events
        )

        if gc.corrections_config.has_enabled_corrections:
            correction_configs = gc.corrections_config.iter_corrections()
            logger.info("Initialising corrections ...")
            for data in self.datasets:
                data.setup_corrections(
                    event=self.event, correction_configs=correction_configs
                )

        self.config = gc

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(
                    shared(num.array([1.0]), name="geo_llk_%i" % t, borrow=True)
                )

    def init_weights(self):
        self.weights = []
        for i, data in enumerate(self.datasets):
            if int(data.covariance.data.sum()) == data.ncoords:
                logger.warning(
                    "Data covariance is identity matrix! Please double check!!!"
                )

            choli = data.covariance.chol_inverse
            self.weights.append(shared(choli, name="geo_weight_%i" % i, borrow=True))
            data.covariance.update_slog_pdet()

    @property
    def n_t(self):
        return len(self.datasets)

    def get_all_dataset_names(self, hp_name):
        """
        Return unique GNSS stations and radar acquisitions.
        """
        names = []
        for dataset in self.datasets:
            if dataset.typ == hp_name.split("_")[1]:
                if isinstance(dataset, heart.DiffIFG):
                    names.append(dataset.name)
                elif isinstance(dataset, heart.GNSSCompoundComponent):
                    names.append(dataset.component)
                else:
                    TypeError(
                        'Geodetic Dataset of class "%s" not '
                        "supported" % dataset.__class__.__name__
                    )

        return names

    def analyse_noise(self, tpoint=None):
        """
        Analyse geodetic noise in datasets and set
        data-covariance matrixes accordingly.
        """
        if self.config.noise_estimator.structure == "non-toeplitz":
            results = self.assemble_results(tpoint)
        else:
            results = [None] * len(self.datasets)

        if len(self.datasets) != len(results):
            raise ValueError("Number of datasets and results need to be equal!")

        for dataset, result in zip(self.datasets, results):
            logger.info(
                'Retrieving geodetic data-covariances with structure "%s" '
                "for %s ..." % (self.config.noise_estimator.structure, dataset.name)
            )

            cov_d_geodetic = self.noise_analyser.get_data_covariance(
                dataset, result=result
            )

            if dataset.covariance is None:
                dataset.covariance = heart.Covariance(data=cov_d_geodetic)
            else:
                dataset.covariance.data = cov_d_geodetic

            if int(dataset.covariance.data.sum()) == dataset.ncoords:
                logger.warning(
                    "Data covariance is identity matrix! Please double check!!!"
                )

    def get_hypersize(self, hp_name=""):
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
        n_datasets = len(self.get_all_dataset_names(hp_name))
        if n_datasets == 0:
            raise ConfigInconsistentError(
                'Found no data for hyperparameter "%s". Please either load'
                " the data or remove it from types dictionary!" % hp_name,
                params="hypers",
            )
        elif self.config.dataset_specific_residual_noise_estimation:
            return n_datasets
        else:
            return 1

    def assemble_results(self, point, **kwargs):
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

        logger.debug("Assembling geodetic data ...")

        processed_synts = self.get_synthetics(point, outmode="stacked_arrays")

        results = []
        for i, data in enumerate(self.datasets):
            res = data.displacement - processed_synts[i]

            results.append(
                heart.GeodeticResult(
                    point=point,
                    processed_obs=data.displacement,
                    processed_syn=processed_synts[i],
                    processed_res=res,
                )
            )

        return results

    def export(
        self,
        point,
        results_path,
        stage_number,
        fix_output=False,
        force=False,
        update=False,
    ):

        from kite.scene import Scene, UserIOWarning
        from pyrocko.guts import dump

        from beat.plotting import map_displacement_grid

        def save_covs(datasets, cov_mat="pred_v"):
            """
            Save covariance matrixes of given attribute
            """

            covs = {
                dataset.name: getattr(dataset.covariance, cov_mat)
                for dataset in datasets
            }

            outname = os.path.join(results_path, "%s_C_%s" % ("geodetic", cov_mat))
            logger.info('"%s" to: %s' % (outname))
            num.savez(outname, **covs)

        gc = self.config

        results = self.assemble_results(point)

        def get_filename(attr, ending="csv"):
            return os.path.join(
                results_path,
                "{}_{}_{}.{}".format(
                    os.path.splitext(dataset.name)[0], attr, stage_number, ending
                ),
            )

        # export for gnss
        for typ, config in gc.types.items():
            if "GNSS" == typ:
                from pyrocko.model import gnss

                logger.info("Exporting GNSS data ...")
                campaigns = config.load_data(campaign=True)

                for campaign in campaigns:
                    model_camp = gnss.GNSSCampaign(
                        stations=copy.deepcopy(campaign.stations),
                        name="%s_model" % campaign.name,
                    )

                    dataset_to_result = {}
                    for dataset, result in zip(self.datasets, results):
                        if dataset.typ == "GNSS":
                            dataset_to_result[dataset] = result

                    for dataset, result in dataset_to_result.items():
                        for ista, sta in enumerate(model_camp.stations):
                            comp = getattr(sta, dataset.component)
                            comp.shift = result.processed_syn[ista]
                            comp.sigma = 0.0

                    outname = os.path.join(
                        results_path, "gnss_synths_%i.yaml" % stage_number
                    )

                    dump(model_camp, filename=outname)

            elif "SAR" == typ:
                logger.info("Exporting SAR data ...")
                for dataset, result in zip(self.datasets, results):
                    if dataset.typ == "SAR":
                        try:
                            scene_path = os.path.join(config.datadir, dataset.name)
                            logger.info(
                                "Loading full resolution kite scene: %s" % scene_path
                            )
                            scene = Scene.load(scene_path)
                        except UserIOWarning:
                            logger.warning(
                                "Full resolution data could not be"
                                " loaded! Skipping ..."
                            )
                            continue

                        for attr in ["processed_obs", "processed_syn", "processed_res"]:

                            filename = get_filename(attr, ending="csv")
                            displacements = getattr(result, attr)
                            dataset.export_to_csv(filename, displacements)
                            logger.info("Stored CSV file to: %s" % filename)

                            filename = get_filename(attr, ending="yml")
                            vals = map_displacement_grid(displacements, scene)
                            scene.displacement = vals
                            scene.save(filename)
                            logger.info("Stored kite scene to: %s" % filename)

        # export stdz residuals
        self.analyse_noise(point)
        if update:
            logger.info("Saving velocity model covariance matrixes...")
            self.update_weights(point)
            save_covs(self.datasets, "pred_v")

        logger.info("Saving data covariance matrixes...")
        save_covs(self.datasets, "data")

    def init_hierarchicals(self, problem_config):
        """
        Initialize hierarchical parameters.
        Ramp estimation in azimuth and range direction of a radar scene and/or
        Rotation of GNSS stations around an Euler pole
        """
        hierarchicals = problem_config.hierarchicals
        self._hierarchicalnames = []
        for number, corr in enumerate(
            self.config.corrections_config.iter_corrections()
        ):
            logger.info(
                "Evaluating config for %s corrections " "for datasets..." % corr.feature
            )
            if corr.enabled:
                for data in self.datasets:
                    if data.name in corr.dataset_names:
                        hierarchical_names = corr.get_hierarchical_names(
                            name=data.name, number=number
                        )
                    else:
                        hierarchical_names = []

                    for hierarchical_name in hierarchical_names:
                        if not corr.enabled and hierarchical_name in hierarchicals:

                            raise ConfigInconsistentError(
                                "%s %s disabled, but they are defined"
                                " in the problem configuration"
                                " (hierarchicals)!" % (corr.feature, data.name)
                            )

                        if (
                            corr.enabled
                            and hierarchical_name not in hierarchicals
                            and data.name in corr.dataset_names
                        ):
                            raise ConfigInconsistentError(
                                "%s %s corrections enabled, but they are"
                                " not defined in the problem configuration!"
                                " (hierarchicals)" % (corr.feature, data.name)
                            )

                        param = hierarchicals[hierarchical_name]
                        if hierarchical_name not in self.hierarchicals:
                            if not num.array_equal(param.lower, param.upper):
                                kwargs = dict(
                                    name=param.name,
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
                                    self.hierarchicals[
                                        hierarchical_name
                                    ] = Uniform.dist(**kwargs)

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
            else:
                logger.info("No %s correction!" % corr.feature)

        logger.info("Initialized %i hierarchical parameters." % len(self.hierarchicals))

    def apply_corrections(self, residuals, point=None, operation="-"):
        """
        Apply all the configured correction terms e.g. SAR orbital ramps,
        GNSS Euler pole rotations etc...
        """
        for i, dataset in enumerate(self.datasets):
            if dataset.has_correction:
                for corr in dataset.corrections:
                    correction = corr.get_displacements(self.hierarchicals, point=point)
                    # correction = Print('corr')(correction)

                    if operation == "-":
                        residuals[i] -= correction  # needs explicit assignment!
                    elif operation == "+":
                        residuals[i] += correction

        return residuals

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
            choli = self.datasets[l].covariance.chol_inverse
            tmp = choli.dot(result.processed_res)
            _llk = num.asarray([num.dot(tmp, tmp)])
            self._llks[l].set_value(_llk)

    def get_variance_reductions(self, point, results=None, weights=None):
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
            results = self.assemble_results(point)

        ndatasets = len(self.datasets)

        assert len(results) == ndatasets

        if weights is None:
            self.analyse_noise(point)
            self.update_weights(point)
            weights = self.weights

        nweights = len(weights)
        assert nweights == ndatasets

        logger.debug("n weights %i , n datasets %i" % (nweights, ndatasets))

        assert nweights == ndatasets

        logger.debug("Calculating variance reduction for solution ...")

        counter = utility.Counter()
        hp_specific = self.config.dataset_specific_residual_noise_estimation

        var_reds = OrderedDict()
        for dataset, weight, result in zip(self.datasets, weights, results):

            hp = get_hypervalue_from_point(
                point, dataset, counter, hp_specific=hp_specific
            )
            icov = dataset.covariance.inverse(num.exp(hp * 2.0))

            data = result.processed_obs
            residual = result.processed_res

            nom = residual.T.dot(icov).dot(residual)
            denom = data.T.dot(icov).dot(data)

            logger.debug("nom %f, denom %f" % (float(nom), float(denom)))
            var_red = 1 - (nom / denom)

            logger.debug("Variance reduction for %s is %f" % (dataset.name, var_red))

            if 0:
                from matplotlib import pyplot as plt

                fig, ax = plt.subplots(1, 1)
                im = ax.imshow(dataset.covariance.data)
                plt.colorbar(im)
                plt.show()

            var_reds[dataset.name] = var_red

        return var_reds

    def get_standardized_residuals(self, point, results=None, weights=None):
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
            results = self.assemble_results(point)

        if weights is None:
            self.update_weights(point)

        counter = utility.Counter()
        hp_specific = self.config.dataset_specific_residual_noise_estimation

        stdz_residuals = OrderedDict()
        for dataset, result in zip(self.datasets, results):
            hp = get_hypervalue_from_point(
                point, dataset, counter, hp_specific=hp_specific
            )
            choli = num.linalg.inv(dataset.covariance.chol(num.exp(hp * 2.0)))
            stdz_residuals[dataset.name] = choli.dot(result.processed_res)

        return stdz_residuals


class GeodeticSourceComposite(GeodeticComposite):
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
    events : list
        of :class:`pyrocko.model.Event`
        contains information of reference event, coordinates of reference
        point and source time
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, gc, project_dir, sources, events, hypers=False):

        super(GeodeticSourceComposite, self).__init__(
            gc, project_dir, events, hypers=hypers
        )

        self.engine = LocalEngine(store_superdirs=[gc.gf_config.store_superdir])

        self.sources = sources

    def __getstate__(self):
        self.engine.close_cashed_stores()
        return self.__dict__.copy()

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.
        """
        tpoint = copy.deepcopy(point)
        tpoint.update(self.fixed_rvs)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source_params = list(self.sources[0].keys())
        for param in list(tpoint.keys()):
            if param not in source_params:
                tpoint.pop(param)

        source_points = utility.split_point(tpoint)
        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            # reset source time may result in store error otherwise
            source.time = 0.0

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Get geodetic likelihood formula for the model built. Has to be called
        within a with model context.
        Part of the pymc3 model.

        Parameters
        ----------
        input_rvs : dict
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
            "Geodetic optimization on: \n " "%s" % ", ".join(self.input_rvs.keys())
        )

        self.input_rvs.update(fixed_rvs)

        t0 = time()
        disp = self.get_synths(self.input_rvs)
        t1 = time()
        logger.debug("Geodetic forward model on test model takes: %f" % (t1 - t0))

        los_disp = (disp * self.slos_vectors).sum(axis=1)

        residuals = self.Bij.srmap(
            tt.cast((self.sdata - los_disp) * self.sodws, tconfig.floatX)
        )

        self.init_hierarchicals(problem_config)
        self.analyse_noise(tpoint)
        self.init_weights()
        if self.config.corrections_config.has_enabled_corrections:
            logger.info("Applying corrections! ...")
            residuals = self.apply_corrections(residuals, operation="-")

        logpts = multivariate_normal_chol(
            self.datasets, self.weights, hyperparams, residuals, hp_specific=hp_specific
        )

        llk = Deterministic(self._like_name, logpts)
        return llk.sum()


class GeodeticGeometryComposite(GeodeticSourceComposite):
    def __init__(self, gc, project_dir, sources, events, hypers=False):

        super(GeodeticGeometryComposite, self).__init__(
            gc, project_dir, sources, events, hypers=hypers
        )

        if not hypers:
            # synthetics generation
            logger.debug("Initialising synthetics functions ... \n")
            self.get_synths = theanof.GeoSynthesizer(
                engine=self.engine, sources=self.sources, targets=self.targets
            )

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

        displacements = heart.geo_synthetics(
            engine=self.engine, targets=self.targets, sources=self.sources, **kwargs
        )

        synths = []
        for disp, data in zip(displacements, self.datasets):
            los_d = (disp * data.los_vector).sum(axis=1)
            synths.append(los_d)

        if self.config.corrections_config.has_enabled_corrections:
            synths = self.apply_corrections(synths, point=point, operation="+")

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

        if not self.weights:
            self.init_weights()

        self.point2sources(point)

        # update data covariances in case model dependent non-toeplitz
        if self.config.noise_estimator.structure == "non-toeplitz":
            logger.info("Updating data-covariances ...")
            self.analyse_noise(point)

        crust_inds = range(*gc.gf_config.n_variations)
        thresh = 5
        if len(crust_inds) > thresh:
            logger.info("Updating geodetic velocity model-covariances ...")
            if self.config.noise_estimator.structure == "non-toeplitz":
                logger.warning(
                    "Non-toeplitz estimation in combination with model "
                    "prediction covariances is still EXPERIMENTAL and results"
                    " should be interpreted with care!!"
                )

            for i, data in enumerate(self.datasets):
                crust_targets = heart.init_geodetic_targets(
                    datasets=[data],
                    earth_model_name=gc.gf_config.earth_model_name,
                    interpolation=gc.interpolation,
                    crust_inds=crust_inds,
                    sample_rate=gc.gf_config.sample_rate,
                )

                logger.debug("Track %s" % data.name)
                cov_pv = cov.geodetic_cov_velocity_models(
                    engine=self.engine,
                    sources=self.sources,
                    targets=crust_targets,
                    dataset=data,
                    plot=plot,
                    event=self.event,
                    n_jobs=1,
                )

                cov_pv = utility.ensure_cov_psd(cov_pv)
                data.covariance.pred_v = cov_pv
                choli = data.covariance.chol_inverse

                self.weights[i].set_value(choli)
                data.covariance.update_slog_pdet()
        else:
            logger.info(
                "Not updating geodetic velocity model-covariances because "
                "number of model variations is too low! < %i" % thresh
            )


class GeodeticInterseismicComposite(GeodeticSourceComposite):
    def __init__(self, gc, project_dir, sources, events, hypers=False):

        super(GeodeticInterseismicComposite, self).__init__(
            gc, project_dir, sources, events, hypers=hypers
        )

        for source in sources:
            if not isinstance(source, RectangularSource):
                raise TypeError("Sources have to be RectangularSources!")

        if not hypers:
            self._lats = self.Bij.l2a([data.lats for data in self.datasets])
            self._lons = self.Bij.l2a([data.lons for data in self.datasets])

            self.get_synths = theanof.GeoInterseismicSynthesizer(
                lats=self._lats,
                lons=self._lons,
                engine=self.engine,
                targets=self.targets,
                sources=sources,
                reference=self.event,
            )

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
        tpoint = copy.deepcopy(point)
        tpoint.update(self.fixed_rvs)
        spoint, bpoint = seperate_point(tpoint)

        self.point2sources(spoint)

        synths = []
        for target, data in zip(self.targets, self.datasets):
            disp = geo_backslip_synthetics(
                engine=self.engine,
                sources=self.sources,
                targets=[target],
                lons=target.lons,
                lats=target.lats,
                reference=self.event,
                **bpoint
            )
            synths.append((disp * data.los_vector).sum(axis=1))

        return synths

    def update_weights(self, point, n_jobs=1, plot=False):

        if not self.weights:
            self.init_weights()

        logger.warning("Not implemented yet!")
        raise NotImplementedError("Not implemented yet!")


class GeodeticDistributerComposite(GeodeticComposite):
    """
    Comprises how to solve the geodetic (static) linear forward model.
    Distributed slip
    """

    def __init__(self, gc, project_dir, events, hypers=False):

        super(GeodeticDistributerComposite, self).__init__(
            gc, project_dir, events, hypers=hypers
        )

        self.gfs = {}
        self.gf_names = {}

        self._mode = "ffi"
        self.gfpath = os.path.join(project_dir, self._mode, bconfig.linear_gf_dir_name)

        self.fault = None

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

        if crust_inds is None:
            crust_inds = range(*self.config.gf_config.n_variations)

        if not isinstance(crust_inds, list):
            raise TypeError("crust_inds need to be a list!")

        for crust_ind in crust_inds:
            gfs = {}
            for var in self.slip_varnames:
                gflib_name = get_gf_prefix(
                    datatype=self.name,
                    component=var,
                    wavename="static",
                    crust_ind=crust_ind,
                )
                gfpath = os.path.join(self.gfpath, gflib_name)

                gfs = load_gf_library(directory=self.gfpath, filename=gflib_name)

                if make_shared:
                    gfs.init_optimization()

                key = self.get_gflibrary_key(
                    crust_ind=crust_ind, wavename="static", component=var
                )

                self.gf_names[key] = gfpath
                self.gfs[key] = gfs

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
        # tpoint.update(self.fixed_rvs)   if vars are fixed GFS are not calculated

        if self.nevents == 1:
            events = [self.event]  # single event
        else:
            events = self.events  # multi event

        if self.fault is None:
            self.fault = self.load_fault_geometry()

        return self.fault.point2sources(tpoint, events=events)

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Formulation of the distribution problem for the model built. Has to be
        called within a with-model-context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`

        Returns
        -------
        llk : :class:`theano.tensor.Tensor`
            log-likelihood for the distributed slip
        """
        logger.info("Loading %s Green's Functions" % self.name)
        self.load_gfs(
            crust_inds=[self.config.gf_config.reference_model_idx], make_shared=False
        )

        tpoint = problem_config.get_test_point()
        self.analyse_noise(tpoint)
        for gfs in self.gfs.values():
            gfs.init_optimization()

        self.init_weights()

        hp_specific = self.config.dataset_specific_residual_noise_estimation

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs
        ref_idx = self.config.gf_config.reference_model_idx

        mu = tt.zeros((self.Bij.ordering.size), tconfig.floatX)
        for var in self.slip_varnames:
            key = self.get_gflibrary_key(
                crust_ind=ref_idx, wavename="static", component=var
            )
            mu += self.gfs[key].stack_all(slips=input_rvs[var])

        residuals = self.Bij.srmap(
            tt.cast((self.sdata - mu) * self.sodws, tconfig.floatX)
        )

        self.init_hierarchicals(problem_config)
        if self.config.corrections_config.has_enabled_corrections:
            residuals = self.apply_corrections(residuals)

        logpts = multivariate_normal_chol(
            self.datasets, self.weights, hyperparams, residuals, hp_specific=hp_specific
        )

        llk = Deterministic(self._like_name, logpts)

        return llk.sum()

    def get_synthetics(self, point, outmode="data"):
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

        mu = num.zeros((self.Bij.ordering.size))
        for var in self.slip_varnames:
            key = self.get_gflibrary_key(
                crust_ind=ref_idx, wavename="static", component=var
            )
            mu += self.gfs[key].stack_all(slips=point[var])

        synths = self.Bij.a2l(mu)

        if self.config.corrections_config.has_enabled_corrections:
            synths = self.apply_corrections(synths, point=point, operation="+")

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

        if not self.weights:
            self.init_weights()

        # update data covariances in case model dependent non-toeplitz
        if self.config.noise_estimator.structure == "non-toeplitz":
            logger.info("Updating data-covariances ...")
            self.analyse_noise(point)

        crust_inds = range(*gc.gf_config.n_variations)
        thresh = 5
        if len(crust_inds) > thresh:
            logger.info("Updating geodetic velocity model-covariances ...")
            if self.config.noise_estimator.structure == "non-toeplitz":
                logger.warning(
                    "Non-toeplitz estimation in combination with model "
                    "prediction covariances is still EXPERIMENTAL and results"
                    " should be interpreted with care!!"
                )

            crust_inds = list(range(*self.config.gf_config.n_variations))
            n_variations = len(crust_inds)
            if len(self.gfs) != n_variations:
                logger.info("Loading geodetic linear GF matrixes ...")
                self.load_gfs(crust_inds=crust_inds, make_shared=False)

            crust_displacements = num.zeros((n_variations, self.Bij.ordering.size))
            for i, crust_ind in enumerate(crust_inds):
                mu = num.zeros((self.Bij.ordering.size))
                for var in self.slip_varnames:
                    key = self.get_gflibrary_key(
                        crust_ind=crust_ind, wavename="static", component=var
                    )
                    mu += self.gfs[key].stack_all(slips=point[var])

                crust_displacements[i, :] = mu

            crust_synths = self.Bij.a_nd2l(crust_displacements)
            if len(crust_synths) != self.n_t:
                raise ValueError(
                    "Number of datasets %i and number of synthetics %i "
                    "inconsistent!" % (self.n_t, len(crust_synths))
                )

            for i, data in enumerate(self.datasets):
                logger.debug("Track %s" % data.name)
                cov_pv = num.cov(crust_synths[i], rowvar=0)

                cov_pv = utility.ensure_cov_psd(cov_pv)
                data.covariance.pred_v = cov_pv
                choli = data.covariance.chol_inverse

                self.weights[i].set_value(choli)
                data.covariance.update_slog_pdet()
        else:
            logger.info(
                "Not updating geodetic velocity model-covariances because "
                "number of model variations is too low! < %i" % thresh
            )
