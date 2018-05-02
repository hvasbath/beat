import logging
from beat import parallel, backend
from beat.utility import list2string

from numpy.random import seed, randint
from numpy.random import normal, standard_cauchy, standard_exponential, \
    poisson
import numpy as np

from pymc3.model import modelcontext, Point
from pymc3 import CompoundStep
from pymc3.sampling import stop_tuning

import multiprocessing as mp
from tqdm import tqdm


logger = logging.getLogger('base')


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
    s : :class:`numpy.ndarray`
    """
    def __init__(self, s):
        self.s = np.atleast_1d(s)


class NormalProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.s.shape)
        if num_draws:
            size += (num_draws,)
        return normal(scale=self.s[0], size=size).T


class CauchyProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.s.shape)
        if num_draws:
            size += (num_draws,)
        return standard_cauchy(size=size).T * self.s


class LaplaceProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.s.shape)
        if num_draws:
            size += (num_draws,)
        return (standard_exponential(size=size) -
                standard_exponential(size=size)).T * self.s


class PoissonProposal(Proposal):
    def __call__(self, num_draws=None):
        size = (self.s.shape)
        if num_draws:
            size += (num_draws,)
        return poisson(lam=self.s, size=size).T - self.s


class MultivariateNormalProposal(Proposal):
    def __call__(self, num_draws=None):
        return np.random.multivariate_normal(
            mean=np.zeros(self.s.shape[0]), cov=self.s, size=num_draws)


class MultivariateStudentTProposal(Proposal):
    def __call__(self, df, num_draws=None):
        return multivariate_t_rvs(
            mean=np.zeros(self.s.shape[0]),
            cov=self.s, df=df, size=num_draws)


class MultivariateCauchyProposal(Proposal):
    """
    Uses multivariate student T distribution with degrees
    of freedom equal to one.
    """
    def __call__(self, num_draws=None):
        return multivariate_t_rvs(
            mean=np.zeros(self.s.shape[0]),
            cov=self.s, df=1, size=num_draws)


proposal_dists = {
    'Cauchy': CauchyProposal,
    'Poisson': PoissonProposal,
    'Normal': NormalProposal,
    'Laplace': LaplaceProposal,
    'MultivariateNormal': MultivariateNormalProposal,
    'MultivariateCauchy': MultivariateCauchyProposal}


def choose_proposal(proposal_name, scale=1.):
    """
    Initialises and selects proposal distribution.

    Parameters
    ----------
    proposal_name : string
        Name of the proposal distribution to initialise
    scale : float or :class:`numpy.ndarray`

    Returns
    -------
    class:`pymc3.Proposal` Object
    """
    return proposal_dists[proposal_name](scale)


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

    if progressbar:
        try:
            current = mp.current_process()
            n = current._identity[0]
        except IndexError:
            # in case of only one used core ...
            n = 1

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

    return chain


def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                 model=None, random_seed=-1):
    """
    Modified from :func:`pymc3.sampling._iter_sample` to be more efficient with
    the SMC algorithm.
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
        logger.debug('Start Record: Chain_%i step_%i' % (chain, i))
        trace.record(out_list, i)
        logger.debug('End Record: Chain_%i step_%i' % (chain, i))
        yield trace


def iter_parallel_chains(
        draws, step, stage_path, progressbar, model, n_jobs,
        chains=None):
    """
    Do Metropolis sampling over all the chains with each chain being
    sampled 'draws' times. Parallel execution according to n_jobs.
    If jobs hang for any reason they are being killed after an estimated
    timeout. The chains in question are being rerun and the estimated timeout
    is added again.
    """
    timeout = 0

    if chains is None:
        chains = list(range(step.n_chains))

    n_chains = len(chains)

    if n_chains == 0:
        mtrace = backend.load_multitrace(dirname=stage_path, model=model)

    # while is necessary if any worker times out - rerun in case
    while n_chains > 0:
        trace_list = []

        logger.info('Initialising %i chain traces ...' % n_chains)
        for chain in chains:
            trace_list.append(backend.TextChain(stage_path, model=model))

        max_int = np.iinfo(np.int32).max
        random_seeds = [randint(max_int) for _ in range(n_chains)]

        work = [(draws, step, step.population[step.resampling_indexes[chain]],
                trace, chain, None, progressbar, model, rseed)
                for chain, rseed, trace in zip(
                    chains, random_seeds, trace_list)]

        tps = step.time_per_sample(10)

        if draws < 10:
            chunksize = int(np.ceil(float(n_chains) / n_jobs))
            tps += 5.
        elif draws > 10 and tps < 1.:
            chunksize = int(np.ceil(float(n_chains) / n_jobs))
        else:
            chunksize = n_jobs

        timeout += int(np.ceil(tps * draws)) * n_jobs

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
            nprocs=n_jobs)

        logger.info('Sampling ...')

        for res in p:
            pass

        # return chain indexes that have been corrupted
        mtrace = backend.load_multitrace(dirname=stage_path, model=model)
        corrupted_chains = backend.check_multitrace(
            mtrace, draws=draws, n_chains=step.n_chains)

        n_chains = len(corrupted_chains)

        if n_chains > 0:
            logger.warning(
                '%i Chains not finished sampling,'
                ' restarting ...' % n_chains)

        chains = corrupted_chains

    return mtrace
