from logging import getLogger
import os

from beat import config as bconfig
from beat.models import hyper_normal
from beat import sampler
from beat.backend import TextStage

from pymc3 import Deterministic

import numpy as num

from pyrocko.util import ensuredir


logger = getLogger('models.base')


__all__ = [
    'ConfigInconsistentError',
    'Composite',
    'sample',
    'Stage',
    'load_stage',
    'estimate_hypers']


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

    def get_hypernames(self):
        if self.config is not None:
            return self.config.get_hypernames()
        else:
            return list(self.hyperparams.keys())


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

        ensuredir(problem.outfolder)

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
            buffer_size=sc.buffer_size,
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
    ensuredir(name)

    stage_handler = TextStage(problem.outfolder)
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

    for v, i in pc.hyperparameters.items():
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
            self.handler = TextStage(homepath)
        else:
            raise TypeError('Either handler or homepath have to be not None')

        self.number = stage_number

    def load_results(
            self, varnames=None, model=None,
            stage_number=None, chains=None, load='trace'):
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
            raise ValueError(
                'Either "varnames" or "model" need to be not None!')

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

        if 'trace' in to_load:
            self.mtrace = self.handler.load_multitrace(
                stage_number, varnames=varnames, chains=chains)

        if 'params' in to_load:
            if model is not None:
                with model:
                    self.step, self.updates = self.handler.load_sampler_params(
                        stage_number)
            else:
                raise ValueError('To load sampler params model is required!')


def load_stage(problem, stage_number, load='trace'):

    stage = Stage(
        homepath=problem.outfolder, stage_number=stage_number)
    stage.load_results(
        varnames=problem.varnames,
        model=problem.model, stage_number=stage_number, load=load)
    return stage
