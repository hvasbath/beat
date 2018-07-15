import logging
from copy import deepcopy
import os
import shutil

from beat import parallel
from beat.backend import check_multitrace, load_multitrace, TextChain
from beat.utility import list2string

from numpy.random import seed, randint
from numpy.random import normal, standard_cauchy, standard_exponential, \
    poisson
import numpy as np

from theano import function

from pymc3.model import modelcontext, Point
from pymc3 import CompoundStep
from pymc3.sampling import stop_tuning
from pymc3.theanof import join_nonshared_inputs

from tqdm import tqdm


logger = logging.getLogger('sampler')


__all__ = [
    'choose_proposal',
    'iter_parallel_chains',
    'init_stage',
    'available_proposals']


def multivariate_t_rvs(mean, cov, df=np.inf, size=1):
    """
    generate random variables of multivariate t distribution

    Parameters
    ----------
    mean : array_like
        mean of random variable, length determines dimension of random variable
    cov : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    size : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
        (same output format as random.multivariate_normal)

    Notes
    -----
    Modified from source:
    http://pydoc.net/scikits.statsmodels/0.3.1/scikits.statsmodels.sandbox.distributions.multivariate/
    """

    m = np.asarray(mean)
    d = len(mean)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, size) / df

    z = np.random.multivariate_normal(np.zeros(d), cov, (size,))
    return m + z / np.sqrt(x)[:, None]


class Proposal(object):
    """
    Proposal distributions modified from pymc3 to initially create all the
    Proposal steps without repeated execution of the RNG- significant speedup!

    Parameters
    ----------
    scale : :class:`numpy.ndarray`
    """
    def __init__(self, scale):
        self.scale = np.atleast_1d(scale)


class DiscreteBoundedUniformProposal(Proposal):
    """
    Returns uniform random integerers witin provided bounds.

    Parameters
    ----------
    lower : int
        lower bound of interval (included), (default: 0)
    upper : int
        upper bound of interval (excluded), (default: 10)
    scale : float or int
        returned values are multiples of this value, (default: 1)
    """
    def __init__(self, lower=0, upper=10, scale=1):
        self.lower = lower
        self.upper = upper
        super(DiscreteBoundedUniformProposal, self).__init__(scale)

    def __call__(self, size=1):
        """
        Returns random numbers within specifications.

        Parameters
        ----------
        size :  int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        :class:`numpy.NdArray` is returned
        the returned data type is determined by the data type of "scale"
        (default: int)
        """
        return (randint(
            low=self.upper - self.lower, size=size) +
            self.lower) * self.scale


class NormalProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.scale.shape)
        if num_draws:
            size += (num_draws,)
        return normal(scale=self.scale[0], size=size).T


class CauchyProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.scale.shape)
        if num_draws:
            size += (num_draws,)
        return standard_cauchy(size=size).T * self.scale


class LaplaceProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.scale.shape)
        if num_draws:
            size += (num_draws,)
        return (standard_exponential(size=size) -
                standard_exponential(size=size)).T * self.scale


class PoissonProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.scale.shape)
        if num_draws:
            size += (num_draws,)
        return poisson(lam=self.scale, size=size).T - self.scale


class MultivariateNormalProposal(Proposal):
    def __call__(self, num_draws=None):
        return np.random.multivariate_normal(
            mean=np.zeros(self.scale.shape[0]), cov=self.scale, size=num_draws)


class MultivariateStudentTProposal(Proposal):
    def __call__(self, df, num_draws=None):
        return multivariate_t_rvs(
            mean=np.zeros(self.scale.shape[0]),
            cov=self.scale, df=df, size=num_draws)


class MultivariateCauchyProposal(Proposal):
    """
    Uses multivariate student T distribution with degrees
    of freedom equal to one.
    """
    def __call__(self, num_draws=None):
        return multivariate_t_rvs(
            mean=np.zeros(self.scale.shape[0]),
            cov=self.scale, df=1, size=num_draws)


proposal_distributions = {
    'Cauchy': CauchyProposal,
    'Poisson': PoissonProposal,
    'Normal': NormalProposal,
    'Laplace': LaplaceProposal,
    'MultivariateNormal': MultivariateNormalProposal,
    'MultivariateCauchy': MultivariateCauchyProposal,
    'DiscreteBoundedUniform': DiscreteBoundedUniformProposal}


multivariate_proposals = ['MultivariateCauchy', 'MultivariateNormal']


def available_proposals():
    return proposal_distributions.keys()


def choose_proposal(proposal_name, **kwargs):
    """
    Initialises and selects proposal distribution.

    Parameters
    ----------
    proposal_name : string
        Name of the proposal distribution to initialise
        See function available_proposals
    kwargs : dict
        of arguments to the proposal distribution

    Returns
    -------
    class:`pymc3.Proposal` Object
    """
    return proposal_distributions[proposal_name](**kwargs)


def setup_chain_counter(n_chains, n_jobs):
    counter = ChainCounter(n=n_chains, n_jobs=n_jobs)
    parallel.counter = counter


class ChainCounter(object):

    def __init__(self, n, n_jobs, perc_disp=0.2, subject='chains'):

        n_chains_worker = n / n_jobs
        frac_disp = int(np.ceil(n_chains_worker * perc_disp))
        self.chain_count = 0
        self.n_chains = n_chains_worker
        self.subject = subject
        self.logger_steps = range(
            frac_disp, n_chains_worker + 1, frac_disp)

    def __call__(self, i):
        """
        Counts the number of finished chains within the execution
        of a pool.

        Parameters
        ----------
        i : int
            Process number
        """
        self.chain_count += 1
        if self.chain_count in self.logger_steps:
            logger.info(
                'Worker %i: Finished %i / %i %s' %
                (i, self.chain_count, self.n_chains, self.subject))


def _sample(draws, step=None, start=None, trace=None, chain=0, tune=None,
            progressbar=True, model=None, random_seed=-1):

    shared_params = [
        sparam for sparam in step.logp_forw.get_shared()
        if sparam.name in parallel._tobememshared]

    if len(shared_params) > 0:
        logger.debug('Accessing shared memory')
        parallel.borrow_all_memories(
            shared_params, parallel._shared_memory.values())

    sampling = _iter_sample(draws, step, start, trace, chain,
                            tune, model, random_seed)

    n = parallel.get_process_id()

    if progressbar:
        sampling = tqdm(
            sampling,
            total=draws,
            desc='chain: %i worker %i' % (chain, n),
            position=n,
            leave=False,
            ncols=65)
    try:
        for strace in sampling:
            pass

    except KeyboardInterrupt:
        raise
    finally:
        if progressbar:
            sampling.close()
        else:
            if hasattr(parallel, 'counter'):
                parallel.counter(n)

        strace.record_buffer()

    return chain


def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                 model=None, random_seed=-1):
    """
    Modified from :func:`pymc3.sampling._iter_sample`

    tune: int
        adaptiv step-size scaling is stopped after this chain sample
    """

    model = modelcontext(model)

    draws = int(draws)

    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    if random_seed != -1:
        seed(random_seed)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    step.chain_index = chain

    trace.setup(draws, chain)
    for i in range(draws):
        if i == tune:
            step = stop_tuning(step)

        logger.debug('Step: Chain_%i step_%i' % (chain, i))
        point, out_list = step.step(point)

        trace.write(out_list, i)
        yield trace


def init_chain_hypers(problem):
    """
    Use random source parameters and fix the source parameter dependend
    parts of the forward model.

    Parameters
    ----------
    problem : :class:`beat.models.Problem`
    """

    sc = problem.config.sampler_config

    point = problem.get_random_point(include=['hierarchicals', 'priors'])

    if hasattr(sc.parameters, 'update_covariances'):
        if sc.parameters.update_covariances:
            logger.info('Updating Covariances ...')
            problem.update_weights(point)

    logger.debug('Updating source point ...')
    problem.update_llks(point)


def iter_parallel_chains(
        draws, step, stage_path, progressbar, model, n_jobs,
        chains=None, initializer=None, initargs=(),
        buffer_size=5000, chunksize=None):
    """
    Do Metropolis sampling over all the chains with each chain being
    sampled 'draws' times. Parallel execution according to n_jobs.
    If jobs hang for any reason they are being killed after an estimated
    timeout. The chains in question are being rerun and the estimated timeout
    is added again.

    Parameters
    ----------
    draws : int
        number of steps that are taken within each Markov Chain
    step : step object of the sampler class, e.g.:
        :class:`beat.sampler.Metropolis`, :class:`beat.sampler.SMC`
    stage_path : str
        with absolute path to the directory where to store the sampling results
    progressbar : boolean
        flag for displaying a progressbar
    model : :class:`pymc3.model.Model` instance
        holds definition of the forward problem
    n_jobs : int
        number of jobs to run in parallel, must not be higher than the
        number of CPUs
    chains : list
        of integers to the chain numbers, if None then all chains from the
        step object are sampled
    initializer : function
        to run before execution of each sampling process
    initargs : tuple
        of arguments for the initializer
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    chunksize : int
        number of chains to sample within each process

    Returns
    -------
    MultiTrace object
    """
    timeout = 0

    if chains is None:
        chains = list(range(step.n_chains))

    n_chains = len(chains)

    if n_chains == 0:
        mtrace = load_multitrace(dirname=stage_path, model=model)

    # while is necessary if any worker times out - rerun in case
    while n_chains > 0:
        trace_list = []

        if n_chains > 100:
            setup_chain_counter(n_chains, n_jobs)

        logger.info('Initialising %i chain traces ...' % n_chains)
        for chain in chains:
            trace_list.append(
                TextChain(
                    name=stage_path, model=model,
                    buffer_size=buffer_size, progressbar=progressbar))

        max_int = np.iinfo(np.int32).max
        random_seeds = [randint(max_int) for _ in range(n_chains)]

        work = [(draws, step, step.population[step.resampling_indexes[chain]],
                trace, chain, None, progressbar, model, rseed)
                for chain, rseed, trace in zip(
                    chains, random_seeds, trace_list)]

        tps = step.time_per_sample(np.minimum(n_jobs, 10))
        logger.info('Serial time per sample: %f' % tps)

        if chunksize is None:
            if draws < 10:
                chunksize = int(np.ceil(float(n_chains) / n_jobs))
            elif draws > 10 and tps < 1.:
                chunksize = int(np.ceil(float(n_chains) / n_jobs))
            else:
                chunksize = n_jobs

        timeout += int(np.ceil(tps * draws)) * n_jobs + 10

        if n_jobs > 1:
            shared_params = [
                sparam for sparam in step.logp_forw.get_shared()
                if sparam.name in parallel._tobememshared]

            logger.info(
                'Data to be memory shared: %s' %
                list2string(shared_params))

            if len(shared_params) > 0:
                if len(parallel._shared_memory.keys()) == 0:
                    logger.info('Putting data into shared memory ...')
                    parallel.memshare_sparams(shared_params)
                else:
                    logger.info('Data already in shared memory!')

            else:
                logger.info('No data to be memshared!')

        else:
            logger.info('Not using shared memory.')

        p = parallel.paripool(
            _sample, work,
            chunksize=chunksize,
            timeout=timeout,
            nprocs=n_jobs,
            initializer=initializer,
            initargs=initargs)

        logger.info('Sampling ...')

        for res in p:
            pass

        # return chain indexes that have been corrupted
        mtrace = load_multitrace(dirname=stage_path, model=model)
        corrupted_chains = check_multitrace(
            mtrace, draws=draws, n_chains=step.n_chains)

        n_chains = len(corrupted_chains)

        if n_chains > 0:
            logger.warning(
                '%i Chains not finished sampling,'
                ' restarting ...' % n_chains)

        chains = corrupted_chains

    return mtrace


def logp_forw(out_vars, vars, shared):
    """
    Compile Theano function of the model and the input and output variables.

    Parameters
    ----------
    out_vars : List
        containing :class:`pymc3.Distribution` for the output variables
    vars : List
        containing :class:`pymc3.Distribution` for the input variables
    shared : List
        containing :class:`theano.tensor.Tensor` for dependend shared data
    """
    out_list, inarray0 = join_nonshared_inputs(out_vars, vars, shared)
    f = function([inarray0], out_list)
    f.trust_input = True
    return f


def init_stage(
        stage_handler, step, stage, model,
        progressbar=False, update=None, rm_flag=False):
    """
    Examine starting point of sampling, reload stages and initialise steps.
    """
    with model:
        if stage == 0:
            # continue or start initial stage
            step.stage = stage
            draws = 1
        else:
            sampler_state, updates = stage_handler.load_sampler_params(stage)
            step.apply_sampler_state(sampler_state)
            draws = step.n_steps

            if update is not None:
                update.apply(updates)

        stage_handler.clean_directory(stage, None, rm_flag)

        chains = stage_handler.recover_existing_results(stage, draws, step)

    return chains, step, update


def update_last_samples(
        homepath, step,
        progressbar=False, model=None, n_jobs=1, rm_flag=False):
    """
    Resampling the last stage samples with the updated covariances and
    accept the new sample.

    Return
    ------
    mtrace : multitrace
    """

    tmp_stage = deepcopy(step.stage)
    logger.info('Updating last samples ...')
    draws = 1
    step.stage = 0
    trans_stage_path = os.path.join(
        homepath, 'trans_stage_%i' % tmp_stage)
    logger.info('in %s' % trans_stage_path)

    if os.path.exists(trans_stage_path) and rm_flag:
        shutil.rmtree(trans_stage_path)

    chains = None
    # reset resampling indexes
    step.resampling_indexes = np.arange(step.n_chains)

    sample_args = {
        'draws': draws,
        'step': step,
        'stage_path': trans_stage_path,
        'progressbar': progressbar,
        'model': model,
        'n_jobs': n_jobs,
        'chains': chains}

    mtrace = iter_parallel_chains(**sample_args)

    step.stage = tmp_stage

    return mtrace
