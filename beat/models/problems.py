import copy
import os
import time
from logging import getLogger

import numpy as num
import theano.tensor as tt
from pymc3 import Deterministic, Model, Potential, Uniform
from pyrocko import util
from theano import config as tconfig

from beat import config as bconfig
from beat.backend import ListArrayOrdering, ListToArrayBijection
from beat.models import geodetic, laplacian, polarity, seismic
from beat.utility import list2string, transform_sources, weed_input_rvs

# disable theano rounding warning
tconfig.warn.round = False

km = 1000.0

logger = getLogger("models")


__all__ = ["GeometryOptimizer", "DistributionOptimizer", "load_model"]


class InconsistentNumberHyperparametersError(Exception):

    context = (
        "Configuration file has to be updated!"
        + " Hyperparameters have to be re-estimated. \n"
        + ' Please run "beat update <project_dir>'
        + ' --parameters=hypers,hierarchicals"'
    )

    def __init__(self, errmess=""):
        self.errmess = errmess

    def __str__(self):
        return "\n%s\n%s" % (self.errmess, self.context)


geometry_composite_catalog = {
    "polarity": polarity.PolarityComposite,
    "seismic": seismic.SeismicGeometryComposite,
    "geodetic": geodetic.GeodeticGeometryComposite,
}


distributer_composite_catalog = {
    "seismic": seismic.SeismicDistributerComposite,
    "geodetic": geodetic.GeodeticDistributerComposite,
    "laplacian": laplacian.LaplacianDistributerComposite,
}


interseismic_composite_catalog = {"geodetic": geodetic.GeodeticInterseismicComposite}


class Problem(object):
    """
    Overarching class for the optimization problems to be solved.

    Parameters
    ----------
    config : :class:`beat.BEATConfig`
        Configuration object that contains the problem definition.
    """

    _varnames = None
    _hypernames = None
    _hierarchicalnames = None

    def __init__(self, config, hypers=False):

        self.model = None

        self._like_name = "like"

        self.fixed_params = {}
        self.composites = {}
        self.hyperparams = {}

        logger.info("Analysing problem ...")
        logger.info("---------------------\n")

        # Load events
        if config.event is None:
            logger.warning("Found no event information!")
            raise AttributeError("Problem config has no event information!")
        else:
            self.event = config.event
            nsubevents = len(config.subevents)
            self.subevents = config.subevents

            if nsubevents > 0:
                logger.info("Found %i subevents." % nsubevents)

        self.config = config

        mode = self.config.problem_config.mode

        outfolder = os.path.join(self.config.project_dir, mode)

        if hypers:
            outfolder = os.path.join(outfolder, "hypers")

        self.outfolder = outfolder
        util.ensuredir(self.outfolder)

    @property
    def events(self):
        return [self.event] + self.subevents

    @property
    def nevents(self):
        return len(self.events)

    def init_sampler(self, hypers=False):
        """
        Initialise the Sampling algorithm as defined in the configuration file.
        """
        from beat import sampler

        if hypers:
            sc = self.config.hyper_sampler_config
        else:
            sc = self.config.sampler_config

        logger.info('Using "%s" backend to store samples!' % sc.backend)

        if self.model is None:
            raise Exception("Model has to be built before initialising the sampler.")

        with self.model:
            if sc.name == "Metropolis":
                logger.info(
                    "... Initiate Metropolis ... \n"
                    " proposal_distribution: %s, tune_interval=%i,"
                    " n_jobs=%i \n"
                    % (
                        sc.parameters.proposal_dist,
                        sc.parameters.tune_interval,
                        sc.parameters.n_jobs,
                    )
                )

                t1 = time.time()
                step = sampler.Metropolis(
                    n_chains=sc.parameters.n_chains,
                    likelihood_name=self._like_name,
                    tune_interval=sc.parameters.tune_interval,
                    proposal_name=sc.parameters.proposal_dist,
                    backend=sc.backend,
                )
                t2 = time.time()
                logger.info("Compilation time: %f" % (t2 - t1))

            elif sc.name == "SMC":
                logger.info(
                    "... Initiate Sequential Monte Carlo ... \n"
                    " n_chains=%i, tune_interval=%i, n_jobs=%i,"
                    " proposal_distribution: %s, \n"
                    % (
                        sc.parameters.n_chains,
                        sc.parameters.tune_interval,
                        sc.parameters.n_jobs,
                        sc.parameters.proposal_dist,
                    )
                )

                t1 = time.time()
                step = sampler.SMC(
                    n_chains=sc.parameters.n_chains,
                    tune_interval=sc.parameters.tune_interval,
                    coef_variation=sc.parameters.coef_variation,
                    proposal_dist=sc.parameters.proposal_dist,
                    likelihood_name=self._like_name,
                    backend=sc.backend,
                )
                t2 = time.time()
                logger.info("Compilation time: %f" % (t2 - t1))

            elif sc.name == "PT":
                logger.info(
                    "... Initiate Metropolis for Parallel Tempering... \n"
                    " proposal_distribution: %s, tune_interval=%i,"
                    " n_chains=%i \n"
                    % (
                        sc.parameters.proposal_dist,
                        sc.parameters.tune_interval,
                        sc.parameters.n_chains,
                    )
                )
                step = sampler.Metropolis(
                    n_chains=sc.parameters.n_chains + 1,  # plus master
                    likelihood_name=self._like_name,
                    tune_interval=sc.parameters.tune_interval,
                    proposal_name=sc.parameters.proposal_dist,
                    backend=sc.backend,
                )

            else:
                raise ValueError(
                    'Sampler "%s" not supported! ' " Typo in config file?" % sc.name
                )

        return step

    def built_model(self):
        """
        Initialise :class:`pymc3.Model` depending on problem composites,
        geodetic and/or seismic data are included. Composites also determine
        the problem to be solved.
        """

        logger.info("... Building model ...\n")

        pc = self.config.problem_config

        with Model() as self.model:

            self.rvs, self.fixed_params = pc.get_random_variables()

            self.init_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for datatype, composite in self.composites.items():
                if datatype in bconfig.modes_catalog[pc.mode].keys():
                    input_rvs = weed_input_rvs(self.rvs, pc.mode, datatype=datatype)
                    fixed_rvs = weed_input_rvs(
                        self.fixed_params, pc.mode, datatype=datatype
                    )

                else:
                    input_rvs = self.rvs
                    fixed_rvs = self.fixed_params

                total_llk += composite.get_formula(
                    input_rvs, fixed_rvs, self.hyperparams, pc
                )

            # deterministic RV to write out llks to file
            like = Deterministic("tmp", total_llk)

            # will overwrite deterministic name ...
            llk = Potential(self._like_name, like)
            logger.info("Model building was successful! \n")

    def plant_lijection(self):
        """
        Add list to array bijection to model object by monkey-patching.
        """
        if self.model is not None:
            lordering = ListArrayOrdering(self.model.unobserved_RVs, intype="tensor")
            lpoint = [var.tag.test_value for var in self.model.unobserved_RVs]
            self.model.lijection = ListToArrayBijection(lordering, lpoint)
        else:
            raise AttributeError("Model needs to be built!")

    def built_hyper_model(self):
        """
        Initialise :class:`pymc3.Model` depending on configuration file,
        geodetic and/or seismic data are included. Estimates initial parameter
        bounds for hyperparameters.
        """

        logger.info("... Building Hyper model ...\n")

        pc = self.config.problem_config

        if len(self.hierarchicals) == 0:
            self.init_hierarchicals()

        point = self.get_random_point(include=["hierarchicals", "priors"])

        if self.config.problem_config.mode == bconfig.geometry_mode_str:
            for param in pc.priors.values():
                point[param.name] = param.testvalue

        with Model() as self.model:

            self.init_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for composite in self.composites.values():
                if hasattr(composite, "analyse_noise"):
                    composite.analyse_noise(point)
                    composite.init_weights()

                composite.update_llks(point)

                total_llk += composite.get_hyper_formula(self.hyperparams)

            like = Deterministic("tmp", total_llk)
            llk = Potential(self._like_name, like)
            logger.info("Hyper model building was successful!")

    def get_random_point(self, include=["priors", "hierarchicals", "hypers"]):
        """
        Get random point in solution space.
        """
        pc = self.config.problem_config

        point = {}
        if "hierarchicals" in include:
            for name, param in self.hierarchicals.items():
                if not isinstance(param, num.ndarray):
                    point[name] = param.random()

        if "priors" in include:
            for param in pc.priors.values():
                shape = pc.get_parameter_shape(param)
                point[param.name] = param.random(shape)

        if "hypers" in include:
            if len(self.hyperparams) == 0:
                self.init_hyperparams()

            hps = {
                hp_name: param.random()
                for hp_name, param in self.hyperparams.items()
                if not isinstance(param, num.ndarray)
            }

            point.update(hps)

        return point

    @property
    def varnames(self):
        """
        Sampled random variable names.

        Returns
        -------
        list of strings
        """
        if self._varnames is None:
            self._varnames = list(
                self.config.problem_config.get_random_variables()[0].keys()
            )
        return self._varnames

    @property
    def hypernames(self):
        """
        Sampled random variable names.

        Returns
        -------
        list of strings
        """
        if self._hypernames is None:
            self.init_hyperparams()
        return self._hypernames

    @property
    def hierarchicalnames(self):
        """
        Sampled random variable names.

        Returns
        -------
        list of strings
        """
        if self._hierarchicalnames is None:
            self.init_hierarchicals()
        return self._hierarchicalnames

    def init_hyperparams(self):
        """
        Evaluate problem setup and return hyperparameter dictionary.
        """
        pc = self.config.problem_config
        hyperparameters = copy.deepcopy(pc.hyperparameters)

        hyperparams = {}
        n_hyp = 0
        modelinit = True
        self._hypernames = []
        for datatype, composite in self.composites.items():
            hypernames = composite.get_hypernames()

            for hp_name in hypernames:
                if hp_name in hyperparameters.keys():
                    hyperpar = hyperparameters.pop(hp_name)
                    ndata = composite.get_hypersize(hp_name)
                else:
                    raise InconsistentNumberHyperparametersError(
                        "Datasets and -types require additional "
                        " hyperparameter(s): %s!" % hp_name
                    )

                if not num.array_equal(hyperpar.lower, hyperpar.upper):
                    dimension = hyperpar.dimension * ndata

                    kwargs = dict(
                        name=hyperpar.name,
                        shape=dimension,
                        lower=num.repeat(hyperpar.lower, ndata),
                        upper=num.repeat(hyperpar.upper, ndata),
                        testval=num.repeat(hyperpar.testvalue, ndata),
                        dtype=tconfig.floatX,
                        transform=None,
                    )

                    try:
                        hyperparams[hp_name] = Uniform(**kwargs)

                    except TypeError:
                        kwargs.pop("name")
                        hyperparams[hp_name] = Uniform.dist(**kwargs)
                        modelinit = False

                    n_hyp += dimension
                    self._hypernames.append(hyperpar.name)
                    logger.info(
                        "Initialised hyperparameter %s with "
                        "size %i " % (hp_name, ndata)
                    )
                else:
                    logger.info(
                        "not solving for %s, got fixed at %s"
                        % (hyperpar.name, list2string(hyperpar.lower.flatten()))
                    )
                    hyperparams[hyperpar.name] = hyperpar.lower

        if len(hyperparameters) > 0:
            print(hyperparameters)
            raise InconsistentNumberHyperparametersError(
                "There are hyperparameters in config file, which are not"
                " covered by datasets/datatypes."
            )

        if modelinit:
            logger.info("Optimization for %i hyperparameters in total!", n_hyp)

        self.hyperparams = hyperparams

    def update_llks(self, point):
        """
        Update posterior likelihoods of each composite of the problem with
        respect to one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        for composite in self.composites.values():
            composite.update_llks(point)

    def apply(self, weights_dict):
        """
        Update composites in problem object with given composites.
        """
        for comp_name, weights in weights_dict.items():
            if comp_name in self.composites:
                self.composites[comp_name].apply(weights)

    def get_variance_reductions(self, point):
        """
        Get composite variance reductions (VRs) with values from given point.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters, for which the VRs are calculated
        """
        vrs = {}
        pconfig = self.config.problem_config
        for composite in self.composites.values():
            logger.info("Calculating variance reductions for %s" % composite.name)
            kwargs = {}
            if composite.name == "seismic":
                if pconfig.mode == bconfig.ffi_mode_str:
                    chop_bounds = ["b", "c"]
                elif pconfig.mode == bconfig.geometry_mode_str:
                    chop_bounds = ["a", "d"]
                else:
                    raise ValueError("Invalid problem_config mode! %s" % pconfig.mode)
                kwargs["chop_bounds"] = chop_bounds

            vr = composite.get_variance_reductions(point, **kwargs)
            vrs.update(vr)
        return vrs

    def point2sources(self, point):
        """
        Update composite sources(in place) with values from given point.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters, for which the sources are
            updated
        """
        for composite in self.composites.values():
            self.composites[composite.name].point2sources(point)

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
        for composite in self.composites.values():
            if hasattr(composite, "update_weights"):
                composite.update_weights(point, n_jobs=n_jobs)

    def get_weights(self):
        """
        Assemble weights of problem composites in dict for saving.
        """
        outd = {}
        for datatype in self.config.problem_config.datatypes:
            if datatype in self.composites.keys():
                outd[datatype] = self.composites[datatype].weights

        return outd

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

        for composite in self.composites.values():
            d[composite.name] = composite.get_synthetics(point, outmode="data")

        return d

    def init_hierarchicals(self):
        """
        Initialise hierarchical random variables of all composites.
        """
        self._hierarchicalnames = []
        for composite in self.composites.values():
            try:
                composite.init_hierarchicals(self.config.problem_config)
                self._hierarchicalnames.extend(composite._hierarchicalnames)
            except AttributeError:
                pass

    @property
    def hierarchicals(self):
        """
        Return dictionary of all hierarchical variables of the problem.
        """
        d = {}
        for composite in self.composites.values():
            if composite.hierarchicals is not None:
                d.update(composite.hierarchicals)

        return d


class SourceOptimizer(Problem):
    """
    Defines the base-class setup involving non-linear fault geometry.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):

        super(SourceOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        if self.nevents != pc.n_sources and self.nevents != 1:
            raise ValueError(
                "Number of events and sources have to be equal or only one "
                "event has to be used! Number if events %i and number of "
                "sources: %i!" % (self.nevents, pc.n_sources)
            )

        # Init sources
        self.sources = []
        for i in range(pc.n_sources):
            if self.nevents > 1:
                event = self.events[i]
            else:
                event = self.event

            source = bconfig.source_catalog[pc.source_type].from_pyrocko_event(event)
            source.stf = bconfig.stf_catalog[pc.stf_type](duration=event.duration)

            # hardcoded inversion for hypocentral time
            if source.stf is not None:
                source.stf.anchor = -1.0

            self.sources.append(source)


class GeometryOptimizer(SourceOptimizer):
    """
    Defines the model setup to solve for the non-linear fault geometry.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):
        logger.info("... Initialising Geometry Optimizer ... \n")

        super(GeometryOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        dsources = transform_sources(self.sources, pc.datatypes, pc.decimation_factors)

        for datatype in pc.datatypes:
            self.composites[datatype] = geometry_composite_catalog[datatype](
                config[datatype + "_config"],
                config.project_dir,
                dsources[datatype],
                self.events,
                hypers,
            )

        self.config = config

        # updating source objects with test-value in bounds
        tpoint = pc.get_test_point()
        self.point2sources(tpoint)


class InterseismicOptimizer(SourceOptimizer):
    """
    Uses the backslip-model in combination with the blockmodel to formulate an
    interseismic model.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):
        logger.info("... Initialising Interseismic Optimizer ... \n")

        super(InterseismicOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        if pc.source_type == "RectangularSource":
            dsources = transform_sources(self.sources, pc.datatypes)
        else:
            raise TypeError(
                "Interseismic Optimizer has to be used with" " RectangularSources!"
            )

        for datatype in pc.datatypes:
            self.composites[datatype] = interseismic_composite_catalog[datatype](
                config[datatype + "_config"],
                config.project_dir,
                dsources[datatype],
                self.events,
                hypers,
            )

        self.config = config

        # updating source objects with fixed values
        point = self.get_random_point()
        self.point2sources(point)


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
        logger.info("... Initialising Distribution Optimizer ... \n")

        super(DistributionOptimizer, self).__init__(config, hypers)

        for datatype in config.problem_config.datatypes:
            data_config = config[datatype + "_config"]

            composite = distributer_composite_catalog[datatype](
                data_config, config.project_dir, self.events, hypers
            )

            composite.set_slip_varnames(self.varnames)
            self.composites[datatype] = composite

        regularization = config.problem_config.mode_config.regularization
        try:
            composite = distributer_composite_catalog[regularization](
                config.problem_config.mode_config.regularization_config,
                config.project_dir,
                self.events,
                hypers,
            )

            composite.set_slip_varnames(self.varnames)
            self.composites[regularization] = composite
        except KeyError:
            logger.info('Using "%s" regularization ...' % regularization)

        self.config = config

    def lsq_solution(self, point, plot=False):
        """
        Returns non-negtive least-squares solution for given input point.

        Parameters
        ----------
        point : dict
            in solution space

        Returns
        -------
        point with least-squares solution
        """
        from scipy.optimize import nnls

        if self.config.problem_config.mode_config.regularization != "laplacian":
            raise ValueError(
                "Least-squares- solution for distributed slip is only "
                "available with laplacian regularization!"
            )

        lc = self.composites["laplacian"]
        slip_varnames_candidates = ["uparr", "utens"]

        slip_varnames = []
        for var in slip_varnames_candidates:
            if var in self.varnames:
                slip_varnames.append(var)

        if len(slip_varnames) == 0.0:
            raise ValueError(
                "LSQ distributed slip solution is only available for %s,"
                " which were fixed in the setup!"
                % list2string(slip_varnames_candidates)
            )

        npatches = point[slip_varnames[0]].size
        dzero = num.zeros(npatches, dtype=tconfig.floatX)
        # set perp slip variables to zero
        if "uperp" in self.varnames:
            point["uperp"] = dzero

        # set slip variables that are inverted for to one
        for inv_var in slip_varnames:
            point[inv_var] = num.ones(npatches, dtype=tconfig.floatX)

        Gs = []
        ds = []
        for datatype, composite in self.composites.items():
            if datatype == "geodetic":
                crust_ind = composite.config.gf_config.reference_model_idx
                keys = [
                    composite.get_gflibrary_key(
                        crust_ind=crust_ind, wavename="static", component=var
                    )
                    for var in slip_varnames
                ]
                Gs.extend([composite.gfs[key]._gfmatrix.T for key in keys])

                # removing hierarchicals from data
                displacements = []
                for dataset in composite.datasets:
                    displacements.append(copy.deepcopy(dataset.displacement))

                displacements = composite.apply_corrections(
                    displacements, point=point, operation="-"
                )
                ds.extend(displacements)

            elif datatype == "seismic":

                targets_gfs = [[] for i in range(composite.n_t)]
                for pidx in range(npatches):
                    Gseis, dseis = composite.get_synthetics(
                        point, outmode="array", patchidxs=num.array([pidx], dtype="int")
                    )

                    for i, gseis in enumerate(Gseis):
                        targets_gfs[i].append(num.atleast_2d(gseis).T)

                else:
                    # concatenate all the patch gfs target wise
                    gseis_p = list(map(num.hstack, targets_gfs))
                    Gs.extend(gseis_p)

                ds.extend(dseis)

        if len(Gs) == 0:
            raise ValueError(
                "No Greens Function matrix available!" " (needs geodetic datatype!)"
            )

        G = num.vstack(Gs)
        D = (
            num.vstack([lc.smoothing_op for sv in slip_varnames])
            * point[bconfig.hyper_name_laplacian] ** 2.0
        )

        A = num.vstack([G, D])
        d = num.hstack(ds + [dzero for var in slip_varnames])

        # m, rmse, rankA, singularsA =  num.linalg.lstsq(A.T, d, rcond=None)
        m, res = nnls(A, d)
        npatches = self.config.problem_config.mode_config.npatches
        for i, var in enumerate(slip_varnames):
            point[var] = m[i * npatches : (i + 1) * npatches]

        if plot:
            from beat.plotting import source_geometry

            datatype = self.config.problem_config.datatypes[0]
            gc = self.composites[datatype]
            fault = gc.load_fault_geometry()
            source_geometry(
                fault,
                list(fault.iter_subfaults()),
                event=gc.event,
                values=point[slip_varnames[0]],
                title="slip [m]",
            )  # datasets=gc.datasets

        return point


problem_modes = list(bconfig.modes_catalog.keys())
problem_catalog = {
    problem_modes[0]: GeometryOptimizer,
    problem_modes[1]: DistributionOptimizer,
    problem_modes[2]: InterseismicOptimizer,
}


def load_model(project_dir, mode, hypers=False, build=True):
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
    build : boolean
        flag to build models

    Returns
    -------
    problem : :class:`Problem`
    """

    config = bconfig.load_config(project_dir, mode)

    pc = config.problem_config

    if hypers and len(pc.hyperparameters) == 0:
        raise ValueError(
            "No hyperparameters specified!" " option --hypers not applicable"
        )

    if pc.mode in problem_catalog.keys():
        problem = problem_catalog[pc.mode](config, hypers)
    else:
        logger.error("Modeling problem %s not supported" % pc.mode)
        raise ValueError("Model not supported")

    if build:
        if hypers:
            problem.built_hyper_model()
        else:
            problem.built_model()

    return problem
