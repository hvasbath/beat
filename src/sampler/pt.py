#!/usr/bin/env python
"""
Parallel Tempering algorithm with mpi4py
"""
import os

# disable internal(fine) blas parallelisation as we parallelise over chains
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
from numpy import random
from beat.utility import load_objects, list2string
from beat.sampler import distributed

from beat.sampler.base import _iter_sample, Proposal
from logging import getLogger
from tqdm import tqdm
from theano import config as tconfig

from pymc3.step_methods.metropolis import metrop_select

from logging import getLogger
from collections import OrderedDict


logger = getLogger('pt')


__all__ = [
    'pt_sample',
    'sample_pt_chain',
    'PackageManager']


class PackageManager(object):
    """
    Manages worker related work attributes and holds mappings
    between workers, betas and counts acceptance of chain swaps.
    """
    _worker2index = None

    def __init__(self, n_workers, step):

        self.n_workers = n_workers
        self.step = step
        self.acceptance_matrix = num.zeros(
            (n_workers, n_workers), dtype='int32')
        self._worker_package_mapping = OrderedDict()

    def get_betas(self):
        """
        Get temperature schedule for all the workers.

        Returns
        -------
        list of inverse temperatures (betas)
        """
        n_worker_post = self.n_workers / 2
        n_worker_temp = self.n_workers - n_worker_post

        betas_post = [1 for _ in range(n_worker_post)]
        betas_temp = num.logspace(-5, 0, 4, endpoint=False).tolist()
        return betas_temp + betas_post

    @property
    def workers(self):
        return self._worker_package_mapping.keys()

    def get_package(source, beta):
        """
        Register worker to the manager and get assigned work package.
        The temperature is applied to the'work package.

        Returns
        -------
        step : class:`beat.sampler.Metropolis`
            object that contains the step method how to sample the
            solution space
        """
        step = deepcopy(self.step)
        step.beta = beta
        step.stage = 1
        self._worker_package_mapping[source] = step
        return step

    def propose_chain_swap(m1, m2, source1, source2):
        step1 = self.worker2package(source1)
        step2 = self.worker2package(source2)

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



def init_worker_packages(step, n_workers, swap_interval=(100, 200), progressbar=False):



    packages = []
    for i, beta in enumerate(betas):
        wstep = step deepcopy(self.step)


        package = {
            'draws': choose_proposal(
                'DiscreteBoundedUniform',
                lower=swap_interval[0], upper=swap_interval[1]),
            'step': wstep,
            'start': step.population[i],
            'chain': i,
            'trace': backend.MemoryTrace(),
            tune=None,
            'progressbar': progressbar,
            model=None,
            random_seed=-1}


    return betas


def master_process(
        comm, tags, status, step, n_samples, swap_interval,
        homepath, progressbar, buffer_size, model, rm_flag, swap_interval):
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
    # start sampling of chains with given seed
    logger.info('Master starting with %d workers' % n_workers)
    logger.info('Packing stuff for workers')
    packages = init_worker_packages(
        step=step, n_workers=n_workers, swap_interval=swap_interval)

    stage_handler = backend.TextStage(homepath)
    _ = stage_handler.clean_directory(stage, chains=None, rm_flag=rm_flag)

    logger.info('Initializing result trace...')
    trace = backend.TextChain(
        stage_path=stage_handler.stage_path(stage),
        model=model,
        buffer_size=buffer_size,
        progressbar=progressbar)
    trace.setup(n_samples, 0, overwrite=False)
    # TODO load starting points from existing trace

    logger.info('Sending packages to workers...')
    for i in range(num_workers):
        comm.recv(source=MPI.ANY_SOURCE, tag=tags.READY, status=status)
        source = status.Get_source()
        comm.send(packages[i], dest=source, tag=tags.INIT)
        logger.debug('Sent task to worker %i' % source)
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

        m1, m2 = metrop_select(m1, m2)

        # TODO identify if worker beta == 1 then write out
        trace.write(m1)
        trace.write(m2)
        if len(chain) < nsamples:
            logger.debug('Sending states back to workers ...')
            comm.send(m1, dest=source1, tag=tags.START)
            comm.send(m2, dest=source2, tag=tags.START)
        else:
            logger.info('Requested number of samples reached!')
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
    kwargs = comm.recv(source=0, tag=tag.INIT, status=status)
    logger.debug('Worker %i received package!' % comm.rank)

    # do initial sampling
    result = sample_pt_chain(**kwargs)
    comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)

    # enter repeated sampling
    while True:
        # TODO: make transd-compatible
        startarray = num.empty(step.ordering.size, dtype=tconfig.floatX)
        comm.Recv([startarray, MPI.DOUBLE],
                  tag=MPI.ANY_TAG, source=0, status=status)

        tag = status.Get_tag()
        if tag == tags.START:
            start = step.bij.map(startarray)
            kwargs['start'] = start
            kwargs['draws'] = num.random.randint()

            result = sample_pt_chain(**kwargs)

            logger.debug('Worker %i attempting to send ...' % comm.rank)
            comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)
            logger.debug('Worker %i sent message successfully ...' % comm.rank)
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
        homepath='', progressbar=True,
        buffer_size=5000, model=None, rm_flag=False, keep_tmp=False):
    """
    Paralell Tempering algorithm

    (adaptive) Metropolis sampling over n_jobs of MC chains.
    Half (floor) of these are sampling at beta = 1 (the posterior).
    The other half of the MC chains are tempered linearly down to
    beta = 1e-5. Randomly, the states of chains are swapped based on
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

    sampler_args = [step, n_samples, swap_interval, homepath, progressbar,
                    buffer_size, model, rm_flag]
    distributed.run_mpi_sampler(
        sampler_name='pt',
        sampler_args=sampler_args,
        keep_tmp=keep_tmp,
        n_jobs=n_jobs)


def _sample():
    # Define MPI message tags
    tags = distributed.enum('READY', 'INIT', 'DONE', 'EXIT', 'START')

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
