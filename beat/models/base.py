import os
from collections import OrderedDict
from logging import getLogger

import numpy as num
from pymc3 import Deterministic
from pyrocko.util import ensuredir

from beat import config as bconfig
from beat.backend import SampleStage, thin_buffer
from beat.models.distributions import hyper_normal, get_hyper_name

logger = getLogger("models.base")


__all__ = [
    "ConfigInconsistentError",
    "Composite",
    "sample",
    "Stage",
    "load_stage",
    "estimate_hypers",
    "get_hypervalue_from_point",
]


def get_hypervalue_from_point(point, observe, counter, hp_specific=False):
    hp_name = get_hyper_name(observe)

    if hp_name in point:
        if hp_specific:
            hp = point[hp_name][counter(hp_name)]
        else:
            hp = point[hp_name]
    else:
        hp = num.log(2.0)
    return hp


class ConfigInconsistentError(Exception):
    def __init__(self, errmess="", params="hierarchicals"):
        self.default = (
            "\n Please run: " '"beat update <project_dir> --parameters=%s' % params
        )
        self.errmess = errmess

    def __str__(self):
        return self.errmess + self.default


class FaultGeometryNotFoundError(Exception):
    def __str__(self):
        return "Fault geometry does not exist please run" ' "beat build_gfs ..." first!'


class Composite(object):
    """
    Class that comprises the rules to formulate the problem. Has to be
    used by an overarching problem object.
    """

    def __init__(self, events):

        self.input_rvs = OrderedDict()
        self.fixed_rvs = OrderedDict()
        self.hierarchicals = OrderedDict()
        self.hyperparams = OrderedDict()
        self.name = None
        self._like_name = None
        self.config = None
        self.slip_varnames = []
        self.events = events

    @property
    def event(self):
        """
        Reference event information
        """
        return self.events[0]

    @property
    def nevents(self):
        """
        Number of events with larger separation in time, i.e. hours.
        """
        return len(self.events)

    def set_slip_varnames(self, varnames):
        """
        Set slip components for GFs.
        """
        self.slip_varnames = [
            var for var in varnames if var in bconfig.static_dist_vars
        ]

    def get_hyper_formula(self, hyperparams):
        """
        Get likelihood formula for the hyper model built. Has to be called
        within a with model context.

        problem_config : :class:`config.ProblemConfig`
        """

        hp_specific = self.config.dataset_specific_residual_noise_estimation
        logpts = hyper_normal(
            self.datasets, hyperparams, self._llks, hp_specific=hp_specific
        )
        llk = Deterministic(self._like_name, logpts)
        return llk.sum()

    def apply(self, weights):
        """
        Update composite weight matrixes (in place) with weights in given
        composite.

        Parameters
        ----------
        list : of Theano shared variables
            containing weight matrixes to use for updates
        """

        for i, weight in enumerate(weights):
            A = weight.get_value(borrow=True)
            self.weights[i].set_value(A)

    def get_hypernames(self):
        if self.config is not None:
            return self.config.get_hypernames()
        else:
            return list(self.hyperparams.keys())

    def export(
        self,
        point,
        results_path,
        stage_number,
        fix_output=False,
        force=False,
        update=False,
    ):
        logger.warning(
            "Export method needs to be implemented for " "%s composite!" % self.name
        )
        pass

    def get_standardized_residuals(self, point):
        """
        Parameters
        ----------
        point : dict
            with parameters to point in solution space to calculate
            standardized residuals for

        Returns
        -------
        None
        """
        logger.warning(
            "Standardized residuals " "not implemented for %s composite!" % self.name
        )
        return None

    def get_variance_reductions(self, point):
        """
        Parameters
        ----------
        point : dict
            with parameters to point in solution space to calculate variance
            reductions

        Returns
        -------
        None
        """
        logger.warning(
            "Variance reductions " "not implemented for %s composite!" % self.name
        )
        return {}


def sample(step, problem):
    """
    Sample solution space with the previously initialised algorithm.

    Parameters
    ----------

    step : :class:`SMC` or :class:`pymc3.metropolis.Metropolis`
        from problem.init_sampler()
    problem : :class:`Problem` with characteristics of problem to solve
    """
    pc = problem.config.problem_config
    sc = problem.config.sampler_config
    pa = sc.parameters

    if hasattr(pa, "update_covariances"):
        if pa.update_covariances:
            update = problem
        else:
            update = None

    if pc.mode == bconfig.ffi_mode_str:
        logger.info("Chain initialization with:")
        if pc.mode_config.initialization == "random":
            logger.info("Random starting point.\n")
            start = None
        elif pc.mode_config.initialization == "lsq":
            logger.info("Least-squares-solution \n")
            from tqdm import tqdm

            start = []
            for i in tqdm(range(step.n_chains)):
                point = problem.get_random_point()
                start.append(problem.lsq_solution(point))
    else:
        start = None

    if sc.name == "Metropolis":
        from beat.sampler import metropolis_sample

        logger.info("... Starting Metropolis ...\n")

        ensuredir(problem.outfolder)

        metropolis_sample(
            n_steps=pa.n_steps,
            step=step,
            progressbar=sc.progressbar,
            buffer_size=sc.buffer_size,
            buffer_thinning=sc.buffer_thinning,
            homepath=problem.outfolder,
            start=start,
            burn=pa.burn,
            thin=pa.thin,
            model=problem.model,
            n_jobs=pa.n_jobs,
            rm_flag=pa.rm_flag,
        )

    elif sc.name == "SMC":
        from beat.sampler import smc_sample

        logger.info("... Starting SMC ...\n")

        smc_sample(
            pa.n_steps,
            step=step,
            progressbar=sc.progressbar,
            model=problem.model,
            start=start,
            n_jobs=pa.n_jobs,
            stage=pa.stage,
            update=update,
            buffer_thinning=sc.buffer_thinning,
            homepath=problem.outfolder,
            buffer_size=sc.buffer_size,
            rm_flag=pa.rm_flag,
        )

    elif sc.name == "PT":
        from beat.sampler import pt_sample

        logger.info("... Starting Parallel Tempering ...\n")

        pt_sample(
            step=step,
            n_chains=pa.n_chains + 1,  # add master
            n_samples=pa.n_samples,
            start=start,
            swap_interval=pa.swap_interval,
            beta_tune_interval=pa.beta_tune_interval,
            n_workers_posterior=pa.n_chains_posterior,
            homepath=problem.outfolder,
            progressbar=sc.progressbar,
            buffer_size=sc.buffer_size,
            buffer_thinning=sc.buffer_thinning,
            model=problem.model,
            resample=pa.resample,
            rm_flag=pa.rm_flag,
            record_worker_chains=pa.record_worker_chains,
        )

    else:
        logger.error('Sampler "%s" not implemented.' % sc.name)


def estimate_hypers(step, problem):
    """
    Get initial estimates of the hyperparameters
    """
    from beat.sampler.base import init_chain_hypers, init_stage, iter_parallel_chains

    logger.info("... Estimating hyperparameters ...")

    pc = problem.config.problem_config
    sc = problem.config.hyper_sampler_config
    pa = sc.parameters

    if not (pa.n_chains / pa.n_jobs).is_integer():
        raise ValueError("n_chains / n_jobs has to be a whole number!")

    name = problem.outfolder
    ensuredir(name)

    stage_handler = SampleStage(problem.outfolder, backend=sc.backend)
    chains, step, update = init_stage(
        stage_handler=stage_handler,
        step=step,
        stage=0,
        progressbar=sc.progressbar,
        model=problem.model,
        rm_flag=pa.rm_flag,
    )

    # setting stage to 1 otherwise only one sample
    step.stage = 1
    step.n_steps = pa.n_steps

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
            buffer_thinning=sc.buffer_thinning,
            chunksize=int(pa.n_chains / pa.n_jobs),
        )

    thinned_chain_length = len(
        thin_buffer(list(range(pa.n_steps)), sc.buffer_thinning, ensure_last=True)
    )
    for v in problem.hypernames:
        i = pc.hyperparameters[v]
        d = mtrace.get_values(
            v,
            combine=True,
            burn=int(thinned_chain_length * pa.burn),
            thin=pa.thin,
            squeeze=True,
        )

        lower = num.floor(d.min()) - 2.0
        upper = num.ceil(d.max()) + 2.0
        logger.info(
            "Updating hyperparameter %s from %f, %f to %f, %f"
            % (v, i.lower, i.upper, lower, upper)
        )
        pc.hyperparameters[v].lower = num.atleast_1d(lower)
        pc.hyperparameters[v].upper = num.atleast_1d(upper)
        pc.hyperparameters[v].testvalue = num.atleast_1d((upper + lower) / 2.0)

    config_file_name = "config_" + pc.mode + ".yaml"
    conf_out = os.path.join(problem.config.project_dir, config_file_name)

    problem.config.problem_config = pc
    bconfig.dump(problem.config, filename=conf_out)


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

    def __init__(self, handler=None, homepath=None, stage_number=-1, backend="csv"):

        if handler is not None:
            self.handler = handler
        elif handler is None and homepath is not None:
            self.handler = SampleStage(homepath, backend=backend)
        else:
            raise TypeError("Either handler or homepath have to be not None")

        self.backend = backend
        self.number = stage_number

    def load_results(
        self, varnames=None, model=None, stage_number=None, chains=None, load="trace"
    ):
        """
        Load stage results from sampling.

        Parameters
        ----------
        model : :class:`pymc3.model.Model`
        stage_number : int
            Number of stage to load
        chains : list, optional
            of result chains to load
        load : str
            what to load and return 'full', 'trace', 'params'
        """
        if varnames is None and model is not None:
            varnames = [var.name for var in model.unobserved_RVs]
        elif varnames is None and model is None:
            raise ValueError('Either "varnames" or "model" need to be not None!')

        if stage_number is None:
            stage_number = self.number

        self.path = self.handler.stage_path(stage_number)

        if not os.path.exists(self.path):
            stage_number = self.handler.highest_sampled_stage()

            logger.info(
                "Stage results %s do not exist! Loading last completed"
                " stage %s" % (self.path, stage_number)
            )
            self.path = self.handler.stage_path(stage_number)

        self.number = stage_number

        if load == "full":
            to_load = ["params", "trace"]
        else:
            to_load = [load]

        if "trace" in to_load:
            self.mtrace = self.handler.load_multitrace(
                stage_number, varnames=varnames, chains=chains
            )

        if "params" in to_load:
            if model is not None:
                with model:
                    self.step, self.updates = self.handler.load_sampler_params(
                        stage_number
                    )
            else:
                raise ValueError("To load sampler params model is required!")


def load_stage(problem, stage_number, load="trace", chains=[-1]):

    stage = Stage(
        homepath=problem.outfolder,
        stage_number=stage_number,
        backend=problem.config.sampler_config.backend,
    )
    stage.load_results(
        varnames=problem.varnames,
        chains=chains,
        model=problem.model,
        stage_number=stage_number,
        load=load,
    )
    return stage
