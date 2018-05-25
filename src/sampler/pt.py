#!/usr/bin/env python
"""
Parallel Tempering algorithm with mpi4py
"""
from mpi4py import MPI
from numpy import random
from beat.utility import load_objects, list2string
from beat import distributed
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


def master_process(
        comm, tags, status, step, n_samples,
        homepath, progressbar, model, rm_flag):
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
    num_workers = size - 1
    tasks = range(num_workers)
    chain = []
    active_workers = 0
    # start sampling of chains with given seed
    print("Master starting with %d workers" % num_workers)
    for i in range(num_workers):
        comm.recv(source=MPI.ANY_SOURCE, tag=tags.READY, status=status)
        source = status.Get_source()
        comm.send(tasks[i], dest=source, tag=tags.START)
        print("Sent task to worker %i" % source)
        active_workers += 1

    logger.info("... Parallel tempering ...")
    logger.info("--------------------------")

    while True:
        m1 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source1 = status.Get_source()
        print("Got sample 1 from worker %i" % source1)
        m2 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source2 = status.Get_source()
        print("Got sample 2 from worker %i" % source1)

        m1, m2 = metrop_select(m1, m2)
        print('samples 1, 2 %i %i' % (m1, m2))
        chain.extend([m1, m2])
        if len(chain) < nsamples:
            print("Sending states back to workers ...")
            comm.send(m1, dest=source1, tag=tags.START)
            comm.send(m2, dest=source2, tag=tags.START)
        else:
            print('Requested number of samples reached!')
            break

    logger.info("Master finished! Chain complete!")
    logger.debug("Fireing ...")
    for i in range(1, size):
        logger.debug('Sending pay sleve to %i' % i)
        comm.send(None, dest=i, tag=tags.EXIT)
        logger.debug("Fired worker %i" % i)
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
    status : mpt.status object
    """
    name = MPI.Get_processor_name()
    logger.debug(
        "Entering worker process with rank %d on %s." % (comm.rank, name))
    comm.send(None, dest=0, tag=tags.READY)

    while True:
        logger.debug('Worker %i recieving message ...' % comm.rank)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        logger.debug('Worker %i received!' % comm.rank)

        tag = status.Get_tag()

        if tag == tags.START:
            # Do the work here
!            result = task + 1

            logger.debug('Worker %i attempting to send ...' % comm.rank)
            comm.send(result, dest=0, tag=tags.DONE)
            logger.debug('Worker %i sent message successfully ...' % comm.rank)
        elif tag == tags.EXIT:
            logger.debug('Worker %i went through EXIT!' % comm.rank)
            break


def pt_sample(
        step, n_jobs, n_samples=100000, homepath='', progressbar=True,
        model=None, rm_flag=False, keep_tmp=False):
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
    sampler_args = [step, n_samples, homepath, progressbar, model, rm_flag]
    distributed.run_mpi_sampler(
        sampler_name='pt',
        sampler_args=sampler_args,
        keep_tmp=keep_tmp,
        n_jobs=n_jobs)


def _sample():
    # Define MPI message tags
    tags = distributed.enum('READY', 'DONE', 'EXIT', 'START')

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    if comm.rank == 0:
        logger.init('Loading passed arguments ...')
        args = load_objects(distributed.mpiargs_name)

        master_process(comm, tags, status, *args)
    else:
        worker_process(comm, tags, status)

    print 'Done!'


if __name__ == '__main__':
    print 'here main'
    _sample()
