import os
import time
import copy

from pymc3 import Uniform, Model, Deterministic, Potential

from pyrocko import util

import numpy as num

import theano.tensor as tt

from theano import config as tconfig
from theano import shared

from beat import heart, utility, backend
from beat import sampler
from beat.models.distributions import hyper_normal
from beat.models import geodetic, seismic

from beat import config as bconfig

from logging import getLogger

# disable theano rounding warning
tconfig.warn.round = False

km = 1000.

logger = getLogger('models')


__all__ = [
    'GeometryOptimizer',
    'DistributionOptimizer',
    'ConfigInconsistentError',
    'Composite',
    'LaplacianDistributerComposite',
    'sample',
    'load_model',
    'Stage',
    'load_stage',
    'estimate_hypers']


class InconsistentNumberHyperparametersError(Exception):

    context = 'Configuration file has to be updated!' + \
              ' Hyperparameters have to be re-estimated. \n' + \
              ' Please run "beat sample <project_dir> --hypers"'

    def __init__(self, errmess=''):
        self.errmess = errmess

    def __str__(self):
        return '\n%s\n%s' % (self.errmess, self.context)


class ConfigInconsistentError(Exception):

    def __init__(self, errmess=''):
        self.default = \
            '\n Please run: ' \
            '"beat update <project_dir> --parameters="hierarchicals"'
        self.errmess = errmess

    def __str__(self):
        return self.errmess + self.default


class Composite(object):
    """
    Class that comprises the rules to formulate the problem. Has to be
    used by an overarching problem object.
    """

    def __init__(self):

        self.input_rvs = {}
        self.fixed_rvs = {}
        self.hierarchicals = {}
        self.hyperparams = {}
        self.name = None
        self._like_name = None
        self.config = None

    def get_hyper_formula(self, hyperparams, problem_config):
        """
        Get likelihood formula for the hyper model built. Has to be called
        within a with model context.

        problem_config : :class:`config.ProblemConfig`
        """

        hp_specific = problem_config.dataset_specific_residual_noise_estimation
        logpts = hyper_normal(
            self.datasets, hyperparams, self._llks,
            hp_specific=hp_specific)
        llk = Deterministic(self._like_name, logpts)
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
            A = weight.get_value(borrow=True)
            self.weights[i].set_value(A)


class LaplacianDistributerComposite():

    def __init__(self, project_dir, hypers):

        self._mode = 'ffi'
        self.slip_varnames = bconfig.static_dist_vars
        self.gfpath = os.path.join(
            project_dir, self._mode, bconfig.linear_gf_dir_name)

        self.fault = self.load_fault_geometry()
        self.spatches = shared(self.fault.npatches, borrow=True)
        self._like_name = 'laplacian_like'

        # only one subfault so far, smoothing across and fast-sweep
        # not implemented for more yet

        self.smoothing_op = \
            self.fault.get_subfault_smoothing_operator(0).astype(
                tconfig.floatX)

        self.sdet_shared_smoothing_op = shared(
            heart.log_determinant(
                self.smoothing_op.T * self.smoothing_op, inverse=False),
            borrow=True)

        self.shared_smoothing_op = shared(self.smoothing_op, borrow=True)

        if hypers:
            self._llks = []
            for varname in self.slip_varnames:
                self._llks.append(shared(
                    num.array([1.]),
                    name='laplacian_llk_%s' % varname,
                    borrow=True))

    def load_fault_geometry(self):
        """
        Load fault-geometry, i.e. discretized patches.

        Returns
        -------
        :class:`heart.FaultGeometry`
        """
        return utility.load_objects(
            os.path.join(self.gfpath, bconfig.fault_geometry_name))[0]

    def _eval_prior(self, hyperparam, exponent):
        """
        Evaluate model parameter independend part of the smoothness prior.
        """
        return (-0.5) * \
            (-self.sdet_shared_smoothing_op +
             (self.spatches * tt.log(2 * num.pi) *
              2 * hyperparam) +
             (1. / tt.exp(hyperparam * 2) * exponent))

    def get_formula(self, input_rvs, fixed_rvs, hyperparams):
        """
        Get smoothing likelihood formula for the model built. Has to be called
        within a with model context.
        Part of the pymc3 model.

        Parameters
        ----------
        input_rvs : dict
            of :class:`pymc3.distribution.Distribution`
        fixed_rvs : dict
            of :class:`numpy.array` here only dummy
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        logger.info('Initialising Laplacian smoothing operator ...')

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        hp_name = bconfig.hyper_name_laplacian
        self.input_rvs.update(fixed_rvs)

        logpts = tt.zeros((self.n_t), tconfig.floatX)
        for l, var in enumerate(self.slip_varnames):
            Ls = self.shared_smoothing_op.dot(input_rvs[var])
            exponent = Ls.T.dot(Ls)

            logpts = tt.set_subtensor(
                logpts[l:l + 1],
                self._eval_prior(hyperparams[hp_name], exponent=exponent))

        llk = Deterministic(self._like_name, logpts)
        return llk.sum()

    def update_llks(self, point):
        """
        Update posterior likelihoods (in place) of the composite w.r.t.
        one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        for l, varname in enumerate(self.slip_varnames):
            Ls = self.smoothing_op.dot(point[varname])
            _llk = num.asarray([Ls.T.dot(Ls)])
            self._llks[l].set_value(_llk)

    def get_hyper_formula(self, hyperparams):
        """
        Get likelihood formula for the hyper model built. Has to be called
        within a with model context.
        """

        logpts = tt.zeros((self.n_t), tconfig.floatX)
        for k in range(self.n_t):
            logpt = self._eval_prior(
                hyperparams[bconfig.hyper_name_laplacian], self._llks[k])
            logpts = tt.set_subtensor(logpts[k:k + 1], logpt)

        llk = Deterministic(self._like_name, logpts)
        return llk.sum()

    @property
    def n_t(self):
        return len(self.slip_varnames)


geometry_composite_catalog = {
    'seismic': seismic.SeismicGeometryComposite,
    'geodetic': geodetic.GeodeticGeometryComposite}


distributer_composite_catalog = {
    'seismic': seismic.SeismicDistributerComposite,
    'geodetic': geodetic.GeodeticDistributerComposite,
    'laplacian': LaplacianDistributerComposite}

interseismic_composite_catalog = {
    'geodetic': geodetic.GeodeticInterseismicComposite}


class Problem(object):
    """
    Overarching class for the optimization problems to be solved.

    Parameters
    ----------
    config : :class:`beat.BEATConfig`
        Configuration object that contains the problem definition.
    """

    def __init__(self, config, hypers=False):

        self.model = None

        self._like_name = 'like'

        self.fixed_params = {}
        self.composites = {}
        self.hyperparams = {}

        logger.info('Analysing problem ...')
        logger.info('---------------------\n')

        # Load event
        if config.event is None:
            logger.warn('Found no event information!')
            raise AttributeError('Problem config has no event information!')
        else:
            self.event = config.event

        self.config = config

        mode = self.config.problem_config.mode

        outfolder = os.path.join(self.config.project_dir, mode)

        if hypers:
            outfolder = os.path.join(outfolder, 'hypers')

        self.outfolder = outfolder
        util.ensuredir(self.outfolder)

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
                        sc.parameters.proposal_dist,
                        sc.parameters.tune_interval,
                        sc.parameters.n_jobs))

                t1 = time.time()
                if hypers:
                    step = sampler.Metropolis(
                        n_chains=sc.parameters.n_chains,
                        likelihood_name=self._like_name,
                        tune_interval=sc.parameters.tune_interval,
                        proposal_name=sc.parameters.proposal_dist)
                else:
                    step = sampler.SMC(
                        n_chains=sc.parameters.n_jobs,
                        tune_interval=sc.parameters.tune_interval,
                        likelihood_name=self._like_name,
                        proposal_name=sc.parameters.proposal_dist)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

            elif sc.name == 'SMC':
                logger.info(
                    '... Initiate Sequential Monte Carlo ... \n'
                    ' n_chains=%i, tune_interval=%i, n_jobs=%i \n' % (
                        sc.parameters.n_chains, sc.parameters.tune_interval,
                        sc.parameters.n_jobs))

                t1 = time.time()
                step = sampler.SMC(
                    n_chains=sc.parameters.n_chains,
                    tune_interval=sc.parameters.tune_interval,
                    coef_variation=sc.parameters.coef_variation,
                    proposal_dist=sc.parameters.proposal_dist,
                    likelihood_name=self._like_name)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

            elif sc.name == 'PT':
                logger.info(
                    '... Initiate Metropolis for Parallel Tempering... \n'
                    ' proposal_distribution %s, tune_interval=%i,'
                    ' n_chains=%i \n' % (
                        sc.parameters.proposal_dist,
                        sc.parameters.tune_interval,
                        sc.parameters.n_chains))
                step = sampler.Metropolis(
                    n_chains=sc.parameters.n_chains,
                    likelihood_name=self._like_name,
                    tune_interval=sc.parameters.tune_interval,
                    proposal_name=sc.parameters.proposal_dist)

        return step

    def built_model(self):
        """
        Initialise :class:`pymc3.Model` depending on problem composites,
        geodetic and/or seismic data are included. Composites also determine
        the problem to be solved.
        """

        logger.info('... Building model ...\n')

        pc = self.config.problem_config

        with Model() as self.model:

            self.rvs, self.fixed_params = self.get_random_variables()

            self.init_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for datatype, composite in self.composites.iteritems():
                if datatype in bconfig.modes_catalog[pc.mode].keys():
                    input_rvs = utility.weed_input_rvs(
                        self.rvs, pc.mode, datatype=datatype)
                    fixed_rvs = utility.weed_input_rvs(
                        self.fixed_params, pc.mode, datatype=datatype)

                    if pc.mode == 'ffi':
                        # do the optimization only on the
                        # reference velocity model
                        logger.info("Loading %s Green's Functions" % datatype)
                        data_config = self.config[datatype + '_config']
                        composite.load_gfs(
                            crust_inds=[
                                data_config.gf_config.reference_model_idx],
                            make_shared=True)

                    total_llk += composite.get_formula(
                        input_rvs, fixed_rvs, self.hyperparams, pc)

            # deterministic RV to write out llks to file
            like = Deterministic('tmp', total_llk)

            # will overwrite deterministic name ...
            llk = Potential(self._like_name, like)
            logger.info('Model building was successful!')

    def built_hyper_model(self):
        """
        Initialise :class:`pymc3.Model` depending on configuration file,
        geodetic and/or seismic data are included. Estimates initial parameter
        bounds for hyperparameters.
        """

        logger.info('... Building Hyper model ...\n')

        pc = self.config.problem_config

        point = self.get_random_point(include=['hierarchicals', 'priors'])
        for param in pc.priors.values():
            point[param.name] = param.testvalue

        self.update_llks(point)

        with Model() as self.model:

            self.init_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for composite in self.composites.itervalues():
                total_llk += composite.get_hyper_formula(self.hyperparams, pc)

            like = Deterministic('tmp', total_llk)
            llk = Potential(self._like_name, like)
            logger.info('Hyper model building was successful!')

    def get_random_point(self, include=['priors', 'hierarchicals', 'hypers']):
        """
        Get random point in solution space.
        """
        pc = self.config.problem_config

        point = {}
        if 'hierarchicals' in include:
            if len(self.hierarchicals) == 0:
                self.init_hierarchicals()

            for name, param in self.hierarchicals.items():
                point[name] = param.random()

        if 'priors' in include:
            dummy = {
                param.name: param.random() for param in pc.priors.values()}

            point.update(dummy)

        if 'hypers' in include:
            if len(self.hyperparams) == 0:
                self.init_hyperparams()

            hps = {hp_name: param.random()
                   for hp_name, param in self.hyperparams.iteritems()}

            point.update(hps)

        return point

    def get_random_variables(self):
        """
        Evaluate problem setup and return random variables dictionary.
        Has to be executed in a "with model context"!

        Returns
        -------
        rvs : dict
            variable random variables
        fixed_params : dict
            fixed random parameters
        """
        pc = self.config.problem_config

        logger.debug('Optimization for %i sources', pc.n_sources)

        rvs = dict()
        fixed_params = dict()
        for param in pc.priors.itervalues():
            if not num.array_equal(param.lower, param.upper):
                rvs[param.name] = Uniform(
                    param.name,
                    shape=param.dimension,
                    lower=param.lower,
                    upper=param.upper,
                    testval=param.testvalue,
                    transform=None,
                    dtype=tconfig.floatX)
            else:
                logger.info(
                    'not solving for %s, got fixed at %s' % (
                        param.name,
                        utility.list_to_str(param.lower.flatten())))
                fixed_params[param.name] = param.lower

        return rvs, fixed_params

    def init_hyperparams(self):
        """
        Evaluate problem setup and return hyperparameter dictionary.
        """
        pc = self.config.problem_config
        hyperparameters = copy.deepcopy(pc.hyperparameters)

        hyperparams = {}
        n_hyp = 0
        modelinit = True
        for datatype, composite in self.composites.items():
            hypernames = composite.config.get_hypernames()

            for hp_name in hypernames:
                if hp_name in hyperparameters.keys():
                    hyperpar = hyperparameters.pop(hp_name)

                    if pc.dataset_specific_residual_noise_estimation:
                        ndata = len(composite.get_unique_stations())
                    else:
                        ndata = 1

                else:
                    raise InconsistentNumberHyperparametersError(
                        'Datasets and -types require additional '
                        ' hyperparameter(s): %s!' % hp_name)

                if not num.array_equal(hyperpar.lower, hyperpar.upper):
                    dimension = hyperpar.dimension * ndata

                    kwargs = dict(
                        name=hyperpar.name,
                        shape=dimension,
                        lower=num.repeat(hyperpar.lower, ndata),
                        upper=num.repeat(hyperpar.upper, ndata),
                        testval=num.repeat(hyperpar.testvalue, ndata),
                        dtype=tconfig.floatX,
                        transform=None)

                    try:
                        hyperparams[hp_name] = Uniform(**kwargs)

                    except TypeError:
                        kwargs.pop('name')
                        hyperparams[hp_name] = Uniform.dist(**kwargs)
                        modelinit = False

                    n_hyp += dimension

                else:
                    logger.info(
                        'not solving for %s, got fixed at %s' % (
                            hyperpar.name,
                            utility.list_to_str(hyperpar.lower.flatten())))
                    hyperparams[hyperpar.name] = hyperpar.lower

        if len(hyperparameters) > 0:
            raise InconsistentNumberHyperparametersError(
                'There are hyperparameters in config file, which are not'
                ' covered by datasets/datatypes.')

        if modelinit:
            logger.info('Optimization for %i hyperparemeters in total!', n_hyp)

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
        for composite in self.composites.itervalues():
            composite.update_llks(point)

    def apply(self, problem):
        """
        Update composites in problem object with given composites.
        """
        for composite in problem.composites.values():
            self.composites[composite.name].apply(composite)

    def point2sources(self, point):
        """
        Update composite sources(in place) with values from given point.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters, for which the covariance matrixes
            with respect to velocity model uncertainties are calculated
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

    def init_hierarchicals(self):
        """
        Initialise hierarchical random variables of all composites.
        """
        for composite in self.composites.values():
            composite.init_hierarchicals(self.config.problem_config)

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

        # Init sources
        self.sources = []
        for i in range(pc.n_sources):
            if self.event:
                source = \
                    bconfig.source_catalog[pc.source_type].from_pyrocko_event(
                        self.event)

                source.stf = bconfig.stf_catalog[pc.stf_type](
                    duration=self.event.duration)

                # hardcoded inversion for hypocentral time
                if source.stf is not None:
                    source.stf.anchor = -1.
            else:
                source = bconfig.source_catalog[pc.source_type]()

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
        logger.info('... Initialising Geometry Optimizer ... \n')

        super(GeometryOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        dsources = utility.transform_sources(
            self.sources,
            pc.datatypes,
            pc.decimation_factors)

        for datatype in pc.datatypes:
            self.composites[datatype] = geometry_composite_catalog[datatype](
                config[datatype + '_config'],
                config.project_dir,
                dsources[datatype],
                self.event,
                hypers)

        self.config = config

        # updating source objects with values in bounds
        point = self.get_random_point()
        self.point2sources(point)


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
        logger.info('... Initialising Interseismic Optimizer ... \n')

        super(InterseismicOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        if pc.source_type == 'RectangularSource':
            dsources = utility.transform_sources(
                self.sources,
                pc.datatypes)
        else:
            raise TypeError('Interseismic Optimizer has to be used with'
                            ' RectangularSources!')

        for datatype in pc.datatypes:
            self.composites[datatype] = \
                interseismic_composite_catalog[datatype](
                    config[datatype + '_config'],
                    config.project_dir,
                    dsources[datatype],
                    self.event,
                    hypers)

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
        logger.info('... Initialising Distribution Optimizer ... \n')

        super(DistributionOptimizer, self).__init__(config, hypers)

        for datatype in config.problem_config.datatypes:
            data_config = config[datatype + '_config']

            self.composites[datatype] = distributer_composite_catalog[
                datatype](
                    data_config,
                    config.project_dir,
                    self.event,
                    hypers)

        regularization = config.problem_config.mode_config.regularization
        try:
            self.composites[regularization] = distributer_composite_catalog[
                regularization](config.project_dir, hypers)
        except KeyError:
            logger.info('Using "%s" regularization ...' % regularization)

        self.config = config


problem_modes = bconfig.modes_catalog.keys()
problem_catalog = {
    problem_modes[0]: GeometryOptimizer,
    problem_modes[1]: DistributionOptimizer,
    problem_modes[2]: InterseismicOptimizer}


def sample(step, problem):
    """
    Sample solution space with the previously initalised algorithm.

    Parameters
    ----------

    step : :class:`SMC` or :class:`pymc3.metropolis.Metropolis`
        from problem.init_sampler()
    problem : :class:`Problem` with characteristics of problem to solve
    """

    sc = problem.config.sampler_config
    pa = sc.parameters

    if hasattr(pa, 'update_covariances'):
        if pa.update_covariances:
            update = problem
        else:
            update = None

    if sc.name == 'Metropolis':
        logger.info('... Starting Metropolis ...\n')

        util.ensuredir(problem.outfolder)

        sampler.metropolis_sample(
            n_steps=pa.n_steps,
            step=step,
            progressbar=sc.progressbar,
            buffer_size=sc.buffer_size,
            homepath=problem.outfolder,
            burn=pa.burn,
            thin=pa.thin,
            model=problem.model,
            n_jobs=pa.n_jobs,
            rm_flag=pa.rm_flag)

    elif sc.name == 'SMC':
        logger.info('... Starting SMC ...\n')

        sampler.smc_sample(
            pa.n_steps,
            step=step,
            progressbar=sc.progressbar,
            model=problem.model,
            n_jobs=pa.n_jobs,
            stage=pa.stage,
            update=update,
            homepath=problem.outfolder,
            buffer_size=sc.buffer_size,
            rm_flag=pa.rm_flag)

    elif sc.name == 'PT':
        logger.info('... Starting Parallel Tempering ...\n')

        sampler.pt_sample(
            step=step,
            n_chains=pa.n_chains,
            n_samples=pa.n_samples,
            swap_interval=pa.swap_interval,
            beta_tune_interval=pa.beta_tune_interval,
            n_workers_posterior=pa.n_chains_posterior,
            homepath=problem.outfolder,
            progressbar=sc.progressbar,
            model=problem.model,
            rm_flag=pa.rm_flag)

    else:
        logger.error('Sampler "%s" not implemented.' % sc.name)


def estimate_hypers(step, problem):
    """
    Get initial estimates of the hyperparameters
    """
    from beat.sampler.base import iter_parallel_chains, init_stage, \
        init_chain_hypers

    logger.info('... Estimating hyperparameters ...')

    pc = problem.config.problem_config
    sc = problem.config.hyper_sampler_config
    pa = sc.parameters

    name = problem.outfolder
    util.ensuredir(name)

    stage_handler = backend.TextStage(problem.outfolder)
    chains, step, update = init_stage(
        stage_handler=stage_handler,
        step=step,
        stage=0,
        progressbar=sc.progressbar,
        model=problem.model,
        rm_flag=pa.rm_flag)

    # setting stage to 1 otherwise only one sample
    step.stage = 1
    step.n_steps = pa.n_steps

    problem.composites['seismic']

    with problem.model:
        mtrace = iter_parallel_chains(
            draws=pa.n_steps,
            chains=chains,
            step=step,
            stage_path=stage_handler.stage_path(1),
            progressbar=sc.progressbar,
            model=problem.model,
            n_jobs=pa.n_jobs,
            initializer=init_chain_hypers,
            initargs=(problem,),
            buffer_size=sc.buffer_size,
            chunksize=int(pa.n_chains / pa.n_jobs))

    for v, i in pc.hyperparameters.iteritems():
        d = mtrace.get_values(
            v, combine=True, burn=int(pa.n_steps * pa.burn),
            thin=pa.thin, squeeze=True)

        lower = num.floor(d.min()) - 2.
        upper = num.ceil(d.max()) + 2.
        logger.info('Updating hyperparameter %s from %f, %f to %f, %f' % (
            v, i.lower, i.upper, lower, upper))
        pc.hyperparameters[v].lower = num.atleast_1d(lower)
        pc.hyperparameters[v].upper = num.atleast_1d(upper)
        pc.hyperparameters[v].testvalue = num.atleast_1d((upper + lower) / 2.)

    config_file_name = 'config_' + pc.mode + '.yaml'
    conf_out = os.path.join(problem.config.project_dir, config_file_name)

    problem.config.problem_config = pc
    bconfig.dump(problem.config, filename=conf_out)


def load_model(project_dir, mode, hypers=False, nobuild=False):
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
    nobuild : boolean
        flag to do not build models

    Returns
    -------
    problem : :class:`Problem`
    """

    config = bconfig.load_config(project_dir, mode)

    pc = config.problem_config

    if hypers and len(pc.hyperparameters) == 0:
        raise ValueError(
            'No hyperparameters specified!'
            ' option --hypers not applicable')

    if pc.mode in problem_catalog.keys():
        problem = problem_catalog[pc.mode](config, hypers)
    else:
        logger.error('Modeling problem %s not supported' % pc.mode)
        raise ValueError('Model not supported')

    if not nobuild:
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
    number = None
    path = None
    step = None
    updates = None
    mtrace = None

    def __init__(self, handler=None, homepath=None, stage_number=-1):

        if handler is not None:
            self.handler = handler
        elif handler is None and homepath is not None:
            self.handler = backend.TextStage(homepath)
        else:
            raise TypeError('Either handler or homepath have to be not None')

        self.number = stage_number

    def load_results(self, model=None, stage_number=None, load='trace'):
        """
        Load stage results from sampling.

        Parameters
        ----------
        model : :class:`pymc3.model.Model`
        stage_number : int
            Number of stage to load
        load : str
            what to load and return 'full', 'trace', 'params'
        """
        if stage_number is None:
            stage_number = self.number

        self.path = self.handler.stage_path(stage_number)

        if not os.path.exists(self.path):
            stage_number = self.handler.highest_sampled_stage()

            logger.info(
                'Stage results %s do not exist! Loading last completed'
                ' stage %s' % (self.path, stage_number))
            self.path = self.handler.stage_path(stage_number)

        self.number = stage_number

        if load == 'full':
            to_load = ['params', 'trace']
        else:
            to_load = [load]

        with model:
            if 'trace' in to_load:
                self.mtrace = self.handler.load_multitrace(
                    stage_number, model=model)

            if 'params' in to_load:
                self.step, self.updates = self.handler.load_sampler_params(
                    stage_number)


def load_stage(problem, stage_number, load='trace'):

    stage = Stage(
        homepath=problem.outfolder, stage_number=stage_number)
    stage.load_results(
        model=problem.model, stage_number=stage_number, load=load)
    return stage
