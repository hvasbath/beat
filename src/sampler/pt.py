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

from beat.sampler.base import _iter_sample
from logging import getLogger
from tqdm import tqdm
from theano import config as tconfig

from logging import getLogger


logger = getLogger('pt')


__all__ = [
    'pt_sample']


def metrop_select(m1, m2):
    u = random.rand()
    if u < 0.5:
        print('Rejected swap')
        return m1, m2
    else:
        print('Accepted swap')
        return m2, m1


def sample_pt_chain(
        draws, step=None, start=None, trace=None, chain=0, tune=None,
        progressbar=True, model=None, random_seed=-1):

    step.n_steps = draws
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

    return step.bij.map(step.lij.drmap(strace.buffer[-1]))


def init_worker_packages(step, n_workers):

    n_worker_post = n_workers / 2
    n_worker_temp = n_workers - n_worker_post

    betas_post = [1 for _ in range(n_worker_post)]
    betas_temp = num.logspace(-5, 0, 4, endpoint=False).tolist()
    betas = betas_temp + betas_post

    packages = []
    for i, beta in enumerate(betas):
        wstep = deepcopy(step)
        wstep.beta = beta
        wstep.stage = 1

!!        package = {
            draws
            'step': wstep,
            'start': step.population[i],
            'chain': i}
                    draws, step=None, start=None, trace=None, chain=0, tune=None,
                progressbar=True, model=None, random_seed=-1


    return betas


def master_process(
        comm, tags, status, step, n_samples,
        homepath, progressbar, buffer_size, model, rm_flag):
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
    packages = init_worker_packages(step=step, n_workers=n_workers)

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
        comm.Recv([m1, MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source1 = status.Get_source()
        logger.debug('Got sample 1 from worker %i' % source1)

        m2 = num.empty()
        comm.Recv([m2, MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source2 = status.Get_source()
        logger.debug('Got sample 2 from worker %i' % source1)

        m1, m2 = metrop_select(m1, m2)

        trace.write(m1)
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
        comm.Recv([startarray, MPI.DOUBLE], tag=MPI.ANY_TAG, source=0, status=status)    

        tag = status.Get_tag()
        if tag == tags.START:
            start = step.bij.map(startarray)    
            kwargs['start'] = start

            result = sample_pt_chain(**kwargs)

            logger.debug('Worker %i attempting to send ...' % comm.rank)
            comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)
            logger.debug('Worker %i sent message successfully ...' % comm.rank)
        elif tag == tags.EXIT:
            logger.debug('Worker %i went through EXIT!' % comm.rank)
            break


def pt_sample(
        step, n_jobs, n_samples=100000, homepath='', progressbar=True,
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

    sampler_args = [step, n_samples, homepath, progressbar,
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
