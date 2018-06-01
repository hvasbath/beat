#!/usr/bin/env python
"""
Parallel Tempering algorithm with mpi4py
"""
import os

# disable internal(fine) blas parallelisation as we parallelise over chains
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
import numpy as num

from beat.utility import load_objects, list2string, gather
from beat.sampler import distributed
from beat.backend import MemoryTrace, TextChain, TextStage

from beat.sampler.base import _iter_sample, Proposal, choose_proposal
from logging import getLogger
from tqdm import tqdm
from theano import config as tconfig

from pymc3.step_methods.metropolis import metrop_select, tune

from logging import getLogger
from collections import OrderedDict
from copy import deepcopy


logger = getLogger('pt')


__all__ = [
    'pt_sample',
    'sample_pt_chain',
    'PackageManager']


class SamplingHistory(object):

    def __init__(self):
        self.acceptance = []
        self.acceptance_matrixes = []
        self.t_scales = []

    def record(self, acceptance_matrix, t_scale, acceptance):
        self.acceptance_matrixes.append(acceptance_matrix)
        self.acceptance.append(acceptance)
        self.t_scales.append(t_scale)


class TemperingManager(object):
    """
    Manages worker related work attributes and holds mappings
    between workers, betas and counts acceptance of chain swaps.

    Provides methods for chain_swapping and beta adaptation.
    """
    _worker2index = None

    def __init__(
            self, step, n_workers, model, progressbar,
            swap_interval, beta_tune_interval):

        self.n_workers = n_workers
        self.n_workers_posterior = self.n_workers / 2
        self.n_workers_tempered = self.n_workers - self.n_worker_post
        self._worker_package_mapping = OrderedDict()
        self._posterior_workers = None
        self._betas = None

        self.step = step
        self.acceptance_matrix = num.zeros(
            (n_workers, n_workers), dtype='int32')
        self.beta_tune_interval = beta_tune_interval

        self.betas = []
        self.current_scale = 1.
        self.history = SamplingHistory()
        self._worker_update_check = num.zeros(self.n_workers, dtype='bool')

        self._default_package_kwargs = {
            'draws': choose_proposal(
                'DiscreteBoundedUniform',
                lower=swap_interval[0], upper=swap_interval[1]),
            'step': None,
            'start': None,
            'chain': None,
            'trace': None,
            'tune': None,
            'progressbar': progressbar,
            'model': model,
            'random_seed': -1,
        }

    def worker_beta_updated(self, source, check=False):
        """
        Check if source beta is updated.

        Parameters
        ----------
        source : int
            mpi worker index
        check : boolean
            if True worker beta status is set to "updated"

        Returns
        -------
        boolean, True if beta is updated
        """
        if not check:
            return self._worker_update_check[source]
        else:
            self._worker_update_check[source] = check

    def update_betas(self, t_scale=None):
        """
        Update annealing schedule for all the workers.

        Parameters
        ----------
        t_scale : float
            factor to adjust the step size in the temperatures
            the base step size is 1.e1
        update : bool
            if true the current scale factor is updated by given

        Returns
        -------
        list of inverse temperatures (betas) in decreasing beta order
        """
        if t_scale is None:
            t_scale = self.current_scale

        self.current_scale = t_scale

        betas_post = [1 for _ in range(self.n_workers_posterior)]
        temperature = num.power(
            10., num.arange(1, self.n_workers_tempered) * t_scale)
        betas_temp = (1. / temperature).tolist()
        betas = betas_post + betas_temp

        # updating worker packages
        self._betas = None
        for beta, package in zip(
                betas, self._worker_package_mapping.values()):
            package['step'].beta = beta

        # reset worker process check
        self._worker_update_check = num.zeros(self.n_workers, dtype='bool')

    @property
    def betas(self):
        """
        Inverse of Sampler Temperatures.
        The lower the more likely a step is accepted.
        """
        if self._betas is None:
            self._betas = [
                step.beta
                for step in self._worker_package_mapping.values()['step']]

        return self._betas

    def record_tuning_history(self, acceptance=None):
        if self.current_scale is None:
            raise ValueError('No temperature scale to record!')

        if self.acceptance_matrix.sum() == 0:
            raise ValueError('No acceptance record!')

        self.history.record(
            self.acceptance_matrix, self.current_scale, acceptance)
        self.current_scale = None
        self.acceptance_matrix = num.zeros(
            (self.n_workers, self.n_workers), dtype='int32')

    def get_workers_ge_beta(self, beta):
        """
        Get worker source indexes greater, equal given beta.
        Workers in decreasing beta order.
        """
        return [
            source for source, package in self._worker_package_mapping.items()
            if package['step'].beta >= beta]

    def get_acceptance_swap(self, beta, beta_tune_interval):
        """
        Returns acceptance rate for swapping states between chains.
        """
        logger.debug(
            'Counting accepted swaps of '
            'posterior chains with beta == %f', beta)

        worker_idxs = self.get_workers_ge_beta(beta)
        tempered_worker = worker_idxs.pop()

        rowidxs, colidxs = num.meshgrid(worker_idxs, tempered_worker)
        return (
            float(self.acceptance_matrix[rowidxs, colidxs].sum()) +
            float(self.acceptance_matrix[colidxs, rowidxs].sum())) / \
            beta_tune_interval

    def tune_betas(self):
        """
        Evaluate the acceptance rate of posterior workers and the
        lowest tempered worker
        """
        beta = self.betas[self.n_workers_posterior]
        acceptance = self.get_acceptance_swap(beta, self.beta_tune_interval)

        t_scale = tune(self.current_scale, acceptance)

        # record scaling history
        self.record_tuning_history(acceptance)
        self.update_betas(t_scale)

    @property
    def workers(self):
        return self._worker_package_mapping.keys()

    @property
    def posterior_workers(self):
        """
        Worker indexes that are sampling from the posterior (beta == 1.)
        """
        if self._posterior_workers is None:
            self._posterior_workers = self.get_workers_ge_beta(1.)

        return self._posterior_workers

    def get_package(self, source, beta):
        """
        Register worker to the manager and get assigned the
        annealing parameter and the work package.
        If worker was registered previously continues old task.
        To ensure book-keeping of workers and their sampler states.

        Parameters
        ----------
        source : int
            MPI source id from a worker message
        beta : float
            annealing parameter (typically, between 1-e6 and 0)

        Returns
        -------
        step : class:`beat.sampler.Metropolis`
            object that contains the step method how to sample the
            solution space
        """

        if source not in self._worker_package_mapping.keys():
            logger.info('Initializing new package for worker %i' % source)
            step = deepcopy(self.step)
            step.beta = beta
            step.stage = 1

            package = deepcopy(self._default_package_kwargs)
            package['chain'] = source
            package['start'] = step.population[source]
            package['trace'] = MemoryTrace(source)
            package['step'] = step

            self._worker_package_mapping[source] = package
        else:
            logger.info('Worker already registered! Continuing on old package')
            package = self._worker_package_mapping[source]

        return package

    def propose_chain_swap(self, m1, m2, source1, source2):
        step1 = self.worker2package(source1)['step']
        step2 = self.worker2package(source2)['step']

        llk1 = step1.lij.dmap(step1.bij.rmap(m1))
        llk2 = step2.lij.dmap(step2.bij.rmap(m2))

        _, accepted = metrop_select(
            (step2.beta - step1.beta) * (
                llk1[step1._llk_index] - llk2[step2._llk_index]),
            q=m1, q0=m2)

        self.register_swap(source1, source2, accepted)

        if accepted:
            return m1, m2
        else:
            return m2, m1

    def register_swap(self, source1, source2, accepted):
        w2i = self.worker_index_mapping()
        self.acceptance_matrix[w2i[source1], w2i[source2]] += accepted

    def worker_index_mapping(self):
        if self._worker2index is None:
            self._worker2index = dict(
                (worker, i) for (i, worker) in enumerate(
                    self.workers))
        return self._worker2index

    def worker2package(self, source):
        return self._worker_package_mapping[source]


def master_process(
        comm, tags, status, step, n_samples, swap_interval,
        beta_tune_interval, homepath, progressbar,
        buffer_size, model, rm_flag):
    """
    Master process, that does the managing.
    Sends tasks to workers.
    Collects results and writes them to the trace.
    Fires workers once job is done.

    Parameters
    ----------
    comm : mpi.communicator
    tags : message tags
    status : mpt.status object

    the rest see pt_sample doc-string
    """

    size = comm.size        # total number of processes
    n_workers = size - 1

    stage = -1
    active_workers = 0
    steps_until_tune = 0

    # start sampling of chains with given seed
    logger.info('Master starting with %d workers' % n_workers)
    logger.info('Packing stuff for workers')
    manager = TemperingManager(
        step=step,
        n_workers=n_workers,
        model=model,
        progressbar=progressbar,
        swap_interval=swap_interval,
        beta_tune_interval=beta_tune_interval)

    stage_handler = TextStage(homepath)
    stage_handler.clean_directory(stage, chains=None, rm_flag=rm_flag)

    logger.info('Initializing result trace...')
    trace = TextChain(
        stage_path=stage_handler.stage_path(stage),
        model=model,
        buffer_size=buffer_size,
        progressbar=progressbar)
    trace.setup(n_samples, 0, overwrite=False)
    # TODO load starting points from existing trace

    logger.info('Sending work packages to workers...')
    for beta in manager.betas:
        comm.recv(source=MPI.ANY_SOURCE, tag=tags.READY, status=status)
        source = status.Get_source()
        package = manager.get_package(source, beta)
        comm.send(package, dest=source, tag=tags.INIT)
        logger.debug('Sent work package to worker %i' % source)
        active_workers += 1

    while True:
        m1 = num.empty()
        comm.Recv([m1, MPI.DOUBLE],
                  source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source1 = status.Get_source()
        logger.debug('Got sample 1 from worker %i' % source1)

        m2 = num.empty()
        comm.Recv([m2, MPI.DOUBLE],
                  source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source2 = status.Get_source()
        logger.debug('Got sample 2 from worker %i' % source1)

        m1, m2 = manager.propose_chain_swap(m1, m2, source1, source2)

        # write results to trace if workers sample from posterior
        for source, m in zip([source1, source2], [m1, m2]):
            if source in manager.posterior_workers:
                trace.write(m)

        # beta updating
        steps_until_tune += 2
        if steps_until_tune >= beta_tune_interval:
            manager.tune_betas()
            steps_until_tune = 0

        if len(trace) < n_samples:
            logger.debug('Sending states back to workers ...')
            for source in [source1, source2]:
                if not manager.worker_beta_updated(source1):
                    comm.Send(
                        manager.betas[source1, MPI.DOUBLE],
                        dest=source1, tag=tags.BETA)
                    manager.worker_beta_updated(source1, check=True)

            comm.Send(m1, dest=source2, tag=tags.SAMPLE)
            comm.Send(m2, dest=source2, tag=tags.SAMPLE)
        else:
            logger.info('Requested number of samples reached!')
            trace.record_buffer()
            break

    logger.info('Master finished! Chain complete!')
    logger.debug('Firing ...')
    for i in range(1, size):
        logger.debug('Sending pay cheque to %i' % i)
        comm.send(None, dest=i, tag=tags.EXIT)
        logger.debug('Fired worker %i' % i)
        active_workers -= 1

    logger.info('Feierabend! Sampling finished!')


def worker_process(comm, tags, status):
    """
    Worker processes, that do the actual sampling.
    They receive all arguments by the master process.

    Parameters
    ----------
    comm : mpi.communicator
    tags : message tags
    status : mpi.status object
    """
    name = MPI.Get_processor_name()
    logger.debug(
        "Entering worker process with rank %d on %s." % (comm.rank, name))
    comm.send(None, dest=0, tag=tags.READY)

    logger.debug('Worker %i recieving work package ...' % comm.rank)
    kwargs = comm.recv(source=0, tag=tags.INIT, status=status)
    logger.debug('Worker %i received package!' % comm.rank)

    try:
        step = kwargs['step']
    except KeyError:
        raise ValueError('Step method not defined!')

    # do initial sampling
    result = sample_pt_chain(**kwargs)
    comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)

    # enter repeated sampling
    while True:
        # TODO: make transd-compatible
        data = num.empty(step.ordering.size, dtype=tconfig.floatX)
        comm.Recv([startarray, MPI.DOUBLE],
                  tag=MPI.ANY_TAG, source=0, status=status)

        tag = status.Get_tag()
        if tag == tags.START:
            start = step.bij.rmap(data)
            kwargs['start'] = start

            result = sample_pt_chain(**kwargs)

            logger.debug('Worker %i attempting to send ...' % comm.rank)
            comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)
            logger.debug('Worker %i sent message successfully ...' % comm.rank)

        elif tag == tags.BETA:
            logger.info('Updating beta to: %f on worker %i' % (data[0], comm.rank))
            kwargs['step'].beta = data[0]

        elif tag == tags.EXIT:
            logger.debug('Worker %i went through EXIT!' % comm.rank)
            break


def sample_pt_chain(
        draws, step=None, start=None, trace=None, chain=0, tune=None,
        progressbar=True, model=None, random_seed=-1):
    """
    Sample a single chain of the Parallel Tempering algorithm and return
    the last sample of the chain.
    Depending on the step object the MarkovChain can have various step
    behaviour, e.g. Metropolis, NUTS, ...

    Parameters
    ----------
    draws : int or :class:`beat.sampler.base.Proposal`
        The number of samples to draw for each Markov-chain per stage
        or a Proposal distribution
    step : :class:`sampler.metropolis.Metropolis`
        Metropolis initialisation object
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to random draws from variables (defaults to empty dict)
    chain : int
        Chain number used to store sample in backend.
    stage : int
        Stage where to start or continue the calculation. It is possible to
        continue after completed stages (stage should be the number of the
        completed stage + 1). If None the start will be at stage = 0.
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    progressbar : bool
        Flag for displaying a progress bar
    model : :class:`pymc3.Model`
        (optional if in `with` context) has to contain deterministic
        variable name defined under step.likelihood_name' that contains the
        model likelihood

    Returns
    -------
    :class:`numpy.NdArray` with end-point of the MarkovChain
    """
    if isinstance(draws, Proposal):
        n_steps = draws()
    else:
        n_steps = draws

    step.n_steps = n_steps
    sampling = _iter_sample(n_steps, step, start, trace, chain,
                            tune, model, random_seed)

    n = parallel.get_process_id()

    if progressbar:
        sampling = tqdm(
            sampling,
            total=n_steps,
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

    return step.bij.map(step.lij.drmap(strace.buffer[-1]))


def pt_sample(
        step, n_jobs, n_samples=100000, swap_interval=(100, 300),
        beta_tune_interval=10000, homepath='', progressbar=True,
        buffer_size=5000, model=None, rm_flag=False, keep_tmp=False):
    """
    Paralell Tempering algorithm

    (adaptive) Metropolis sampling over n_jobs of MC chains.
    Half (floor) of these are sampling at beta = 1 (the posterior).
    The other half of the MC chains are tempered linearly down to
    beta = 1e-6. Randomly, the states of chains are swapped based on
    the Metropolis-Hastings acceptance criterion to the power of the
    differences in beta of the involved chains.
    The samples are written to disk only by the master process. Once
    the specified number of samples is reached sampling is stopped.

    Parameters
    ----------
    step : :class:`beat.sampler.Metropolis`
        sampler object
    n_jobs : int
        number of processors to use, which is equal to the number of
        paralell MC chains
    n_samples : int
        number of samples in the result trace, if reached sampling stops
    swap_interval : tuple
        interval for uniform random integer that determines the length
        of each MarkovChain on each worker. The chain end values of workers
        are proposed for swapping state and are written in the final trace
    beta_tune_interval : int
        Evaluate acceptance rate of chain swaps and tune betas similar
        to proposal step tuning
    homepath : string
        Result_folder for storing stages, will be created if not existing
    progressbar : bool
        Flag for displaying a progress bar
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    model : :class:`pymc3.Model`
        (optional if in `with` context) has to contain deterministic
        variable name defined under step.likelihood_name' that contains the
        model likelihood
    rm_flag : bool
        If True existing stage result folders are being deleted prior to
        sampling.
    keep_tmp : bool
        If True the execution directory (under '/tmp/') is not being deleted
        after process finishes
    """
    if n_jobs < 3:
        raise ValueError(
            'Parallel Tempering requires at least 3 processors!')

    sampler_args = [step, n_samples, swap_interval, beta_tune_interval,
                    homepath, progressbar, buffer_size, model, rm_flag]
    distributed.run_mpi_sampler(
        sampler_name='pt',
        sampler_args=sampler_args,
        keep_tmp=keep_tmp,
        n_jobs=n_jobs)


def _sample():
    # Define MPI message tags
    tags = distributed.enum('READY', 'INIT', 'DONE', 'EXIT', 'SAMPLE', 'BETA')

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    if comm.rank == 0:
        logger.init('Loading passed arguments ...')
        args = load_objects(distributed.mpiargs_name)

        master_process(comm, tags, status, *args)
    else:
        worker_process(comm, tags, status)


if __name__ == '__main__':

    _sample()
