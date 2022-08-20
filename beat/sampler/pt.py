#!/usr/bin/env python
"""
Parallel Tempering algorithm with mpi4py
"""
import os
import sys

# disable internal(fine) blas parallelisation as we parallelise over chains
os.environ["OMP_NUM_THREADS"] = "1"

from collections import OrderedDict
from copy import deepcopy
from logging import getLevelName, getLogger
from pickle import HIGHEST_PROTOCOL

import numpy as num
from mpi4py import MPI
from theano import config as tconfig

from beat.backend import MemoryChain, SampleStage, backend_catalog
from beat.config import sample_p_outname
from beat.sampler import distributed
from beat.sampler.base import (
    ChainCounter,
    Proposal,
    _iter_sample,
    choose_proposal,
    multivariate_proposals,
)
from beat.utility import dump_objects, list2string, load_objects, setup_logging

logger = getLogger("pt")


MPI.pickle.PROTOCOL = HIGHEST_PROTOCOL


__all__ = [
    "pt_sample",
    "sample_pt_chain",
    "TemperingManager",
    "SamplingHistory",
    "tune",
]


def tune(scale, acc_rate):
    """
    Tunes the temperature scaling parameter
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.8
    <0.05         x 0.9
    <0.2          x 0.95
    >0.5          x 1.05
    >0.75         x 1.1
    >0.95         x 1.2

    """

    # Switch statement
    if acc_rate < 0.001:
        # reduce by 15 percent
        scale *= 0.85
    elif acc_rate < 0.05:
        # reduce by 10 percent
        scale *= 0.9
    elif acc_rate < 0.2:
        # reduce by 5 percent
        scale *= 0.95
    elif acc_rate > 0.95:
        # increase by 15 percent
        scale *= 1.15
    elif acc_rate > 0.75:
        # increase by 10
        scale *= 1.10
    elif acc_rate > 0.5:
        # increase by 5 percent
        scale *= 1.05

    return scale


class SamplingHistory(object):
    def __init__(self):
        self.acceptance = []
        self.acceptance_matrixes = []
        self.sample_counts = []
        self.t_scales = []
        self.filename = sample_p_outname

    def record(self, sample_count, acceptance_matrix, t_scale, acceptance):

        self.sample_counts.append(sample_count)
        self.acceptance_matrixes.append(acceptance_matrix)
        self.acceptance.append(acceptance)
        self.t_scales.append(t_scale)

    def __len__(self):
        return len(self.acceptance)

    def get_acceptance_matrixes_array(self):
        return num.dstack(self.acceptance_matrixes)

    def get_sample_counts_array(self):
        return num.dstack(self.sample_counts)


class TemperingManager(object):
    """
    Manages worker related work attributes and holds mappings
    between workers, betas and counts acceptance of chain swaps.

    Provides methods for chain_swapping and beta adaptation.
    """

    _worker2index = None

    def __init__(
        self,
        step,
        n_workers,
        model,
        progressbar,
        buffer_size,
        swap_interval,
        beta_tune_interval,
        n_workers_posterior,
    ):

        self.n_workers = n_workers
        self.n_workers_posterior = n_workers_posterior
        self.n_workers_tempered = int(self.n_workers - self.n_workers_posterior)

        self._worker_package_mapping = OrderedDict()
        self._betas = None
        self._t_scale_min = 1.01
        self._t_scale_max = 2.0

        self.step = step
        self.model = model
        self.buffer_size = buffer_size

        self.acceptance_matrix = num.zeros((n_workers, n_workers), dtype="int32")
        self.sample_count = num.zeros_like(self.acceptance_matrix)
        self.beta_tune_interval = beta_tune_interval

        self.current_scale = 1.2

        # TODO: make sampling history reloadable
        self.history = SamplingHistory()
        self._worker_update_check = num.zeros(self.n_workers, dtype="bool")

        self._default_package_kwargs = {
            "draws": choose_proposal(
                "DiscreteBoundedUniform", lower=swap_interval[0], upper=swap_interval[1]
            ),
            "step": None,
            "start": None,
            "chain": None,
            "trace": None,
            "tune": None,
            "progressbar": progressbar,
            "model": model,
            "random_seed": -1,
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
        worker_idx = self.worker2index(source)
        if not check:
            return self._worker_update_check[worker_idx]
        else:
            self._worker_update_check[worker_idx] = check

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

        betas_post = [1.0 for _ in range(self.n_workers_posterior)]
        temperature = num.power(t_scale, num.arange(1, self.n_workers_tempered + 1))
        betas_temp = (1.0 / temperature).tolist()
        betas = betas_post + betas_temp
        logger.info("Updating betas ...")

        if len(self._worker_package_mapping) > 0:
            # updating worker packages
            self._betas = None
            for source, package in self._worker_package_mapping.items():
                new_beta = betas[self.worker2index(source)]
                old_beta = package["step"].beta
                logger.info(
                    "Updating beta of worker %i from %f to %f"
                    % (source, old_beta, new_beta)
                )
                package["step"].beta = new_beta

            # reset worker process check
            self._worker_update_check = num.zeros(self.n_workers, dtype="bool")
        else:
            self._betas = betas

    @property
    def betas(self):
        """
        Inverse of Sampler Temperatures.
        The lower the more likely a step is accepted.
        """
        if self._betas is None:
            self._betas = sorted(
                [
                    package["step"].beta
                    for package in self._worker_package_mapping.values()
                ],
                reverse=True,
            )

        return self._betas

    def record_tuning_history(self, acceptance=None):
        if self.current_scale is None:
            raise ValueError("No temperature scale to record!")

        if self.sample_count.sum() == 0:
            raise ValueError("No sampling record!")

        print("Acceptance matrix: \n", self.acceptance_matrix)
        print("Sample matrix: \n", self.sample_count)
        self.history.record(
            self.sample_count, self.acceptance_matrix, self.current_scale, acceptance
        )
        self.current_scale = None
        self.acceptance_matrix = num.zeros(
            (self.n_workers, self.n_workers), dtype="int32"
        )
        self.sample_count = num.zeros_like(self.acceptance_matrix)

    def dump_history(self, save_dir=None):
        if save_dir is None:
            save_dir = os.getcwd()

        logger.info("Dumping sampler history to %s" % save_dir)
        dump_objects(os.path.join(save_dir, self.history.filename), self.history)

    def get_workers_ge_beta(self, beta, idxs=False):
        """
        Get worker source indexes greater, equal given beta.
        If idxs is True return indexes to sample and acceptance arrays
        """

        workers = sorted(
            [
                source
                for source, package in self._worker_package_mapping.items()
                if package["step"].beta >= beta
            ],
            reverse=True,
        )

        if idxs:
            return [self.worker2index(w) for w in workers]

        else:
            return workers

    def get_acceptance_swap(self, beta, beta_tune_interval):
        """
        Returns acceptance rate for swapping states between chains.
        """
        logger.info(
            "Counting accepted swaps of " "posterior chains with beta == %f", beta
        )

        worker_idxs = self.get_workers_ge_beta(beta, idxs=True)

        logger.info(
            "Worker indexes of beta greater %f: %s" % (beta, list2string(worker_idxs))
        )

        tempered_worker_idxs = deepcopy(worker_idxs)
        for worker_idx in self.get_posterior_workers(idxs=True):
            tempered_worker_idxs.remove(worker_idx)

        logger.info(
            "Workers of tempered chains: %s" % list2string(tempered_worker_idxs)
        )

        rowidxs, colidxs = num.meshgrid(worker_idxs, tempered_worker_idxs)

        n_samples = int(
            self.sample_count[rowidxs, colidxs].sum()
            + self.sample_count[colidxs, rowidxs].sum()
        )
        accepted_samples = int(
            self.acceptance_matrix[rowidxs, colidxs].sum()
            + self.acceptance_matrix[colidxs, rowidxs].sum()
        )

        logger.info(
            "Number of samples %i and accepted samples %i of acceptance"
            " evaluation, respectively." % (n_samples, accepted_samples)
        )
        if n_samples:
            return float(accepted_samples) / float(n_samples)
        else:
            return n_samples

    def get_beta(self, source):
        return num.atleast_1d(self.betas[self.worker2index(source)])

    def tune_betas(self):
        """
        Evaluate the acceptance rate of posterior workers and the
        lowest tempered worker. This scaling here has the inverse
        behaviour of metropolis step scaling! If there is little acceptance
        more exploration is needed and lower beta values are desired.
        """

        beta = self.betas[self.n_workers_posterior]

        acceptance = self.get_acceptance_swap(beta, self.beta_tune_interval)
        logger.info("Acceptance rate: %f", acceptance)

        t_scale = tune(self.current_scale, acceptance)
        if t_scale < self._t_scale_min:
            t_scale = self._t_scale_min
        elif t_scale > self._t_scale_max:
            t_scale = self._t_scale_max

        logger.debug("new temperature scale %f", t_scale)

        # record scaling history
        self.record_tuning_history(acceptance)
        self.update_betas(t_scale)

    def get_posterior_workers(self, idxs=False):
        """
        Worker indexes that are sampling from the posterior (beta == 1.)
        If idxs is True return indexes to sample and acceptance arrays
        """
        return self.get_workers_ge_beta(1.0, idxs=idxs)

    def get_package(self, source, trace=None, resample=False, burnin=1000):
        """
        Register worker to the manager and get assigned the
        annealing parameter and the work package.
        If worker was registered previously continues old task.
        To ensure book-keeping of workers and their sampler states.

        Parameters
        ----------
        source : int
            MPI source id from a worker message
        trace : :class:beat.backend.BaseTrace
            Trace object keeping the samples of the Markov Chain
        resample : bool
            If True all the Markov Chains are starting sampling in the
            testvalue
        burnin : int
            Number of samples the worker is taking before updating the proposal
            covariance matrix based on the trace samples

        Returns
        -------
        step : class:`beat.sampler.Metropolis`
            object that contains the step method how to sample the
            solution space
        """

        if source not in self._worker_package_mapping.keys():

            step = deepcopy(self.step)
            step.beta = self.betas[self.worker2index(source)]  # subtract master
            step.stage = 1
            step.burnin = burnin
            package = deepcopy(self._default_package_kwargs)
            package["chain"] = source
            package["trace"] = trace

            if resample:
                logger.info("Resampling chain %i at the testvalue" % source)
                start = step.population[0]
                step.chain_previous_lpoint[source] = step.chain_previous_lpoint[0]
            else:
                start = step.population[source]

            max_int = num.iinfo(num.int32).max
            package["random_seed"] = num.random.randint(max_int)
            package["step"] = step
            package["start"] = start

            logger.info(
                "Initializing new package for worker %i "
                "with beta %f" % (source, step.beta)
            )
            self._worker_package_mapping[source] = package
        else:
            logger.info("Worker already registered! Continuing on old package")
            package = self._worker_package_mapping[source]

        return package

    def worker_a2l(self, m, source):
        """
        Map m from array to list space worker deoendend.
        """
        step = self.worker2package(source)["step"]
        return step.lij.a2l(m)

    def propose_chain_swap(self, m1, m2, source1, source2):
        """
        Propose a swap between chain samples.

        Parameters
        ----------
        """
        llk1 = self.worker_a2l(m1, source1)
        llk2 = self.worker_a2l(m2, source2)

        step1 = self.worker2package(source1)["step"]
        step2 = self.worker2package(source2)["step"]

        alpha = (step2.beta - step1.beta) * (
            llk1[step1._llk_index] - llk2[step2._llk_index]
        )

        if num.log(num.random.uniform()) < alpha:
            accepted = True
        else:
            accepted = False
        # print('alpha b1, b2, acc', alpha, step1.beta, step2.beta, accepted)

        self.register_swap(source1, source2, accepted)

        if accepted:
            return m2, m1
        else:
            return m1, m2

    def register_swap(self, source1, source2, accepted):
        w1i = self.worker2index(source1)
        w2i = self.worker2index(source2)
        self.acceptance_matrix[w1i, w2i] += accepted
        self.sample_count[w1i, w2i] += 1

    def worker2index(self, source):
        return source - 1

    def worker2package(self, source):
        return self._worker_package_mapping[source]


def master_process(
    comm,
    tags,
    status,
    model,
    step,
    n_samples,
    swap_interval,
    beta_tune_interval,
    n_workers_posterior,
    homepath,
    progressbar,
    buffer_size,
    buffer_thinning,
    resample,
    rm_flag,
    record_worker_chains,
):
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

    size = comm.size  # total number of processes
    n_workers = size - 1

    if n_workers_posterior >= n_workers:
        raise ValueError(
            'Specified more workers that sample in the posterior "%i",'
            ' than there are total number of workers "%i"'
            % (n_workers_posterior, n_workers)
        )

    stage = -1
    active_workers = 0
    steps_until_tune = 0
    burnin = int(n_samples * 0.1)

    # start sampling of chains with given seed
    logger.info("Master starting with %d workers" % n_workers)
    logger.info("Packing stuff for workers")
    manager = TemperingManager(
        step=step,
        n_workers=n_workers,
        n_workers_posterior=n_workers_posterior,
        model=model,
        progressbar=progressbar,
        buffer_size=buffer_size,
        swap_interval=swap_interval,
        beta_tune_interval=beta_tune_interval,
    )

    stage_handler = SampleStage(homepath, backend=step.backend)
    stage_handler.clean_directory(stage, chains=None, rm_flag=rm_flag)

    logger.info("Sampling for %i samples in master chain." % n_samples)
    logger.info("Burn-in for %i samples." % burnin)
    logger.info("Initializing result trace...")
    logger.info("Writing samples to file every %i samples." % buffer_size)
    master_trace = backend_catalog[step.backend](
        dir_path=stage_handler.stage_path(stage),
        model=model,
        buffer_size=buffer_size,
        buffer_thinning=buffer_thinning,
        progressbar=progressbar,
    )
    master_trace.setup(n_samples, 0, overwrite=rm_flag)
    # TODO load starting points from existing trace

    logger.info("Sending work packages to workers...")
    manager.update_betas()
    for beta in manager.betas:
        comm.recv(source=MPI.ANY_SOURCE, tag=tags.READY, status=status)
        source = status.Get_source()

        if record_worker_chains:
            worker_trace = backend_catalog[step.backend](
                dir_path=stage_handler.stage_path(stage),
                model=model,
                buffer_size=buffer_size,
                buffer_thinning=buffer_thinning,
                progressbar=progressbar,
            )
        else:
            worker_trace = MemoryChain(buffer_size=buffer_size)

        worker_trace.setup(n_samples, source, overwrite=rm_flag)

        package = manager.get_package(
            source, trace=worker_trace, resample=resample, burnin=burnin
        )
        comm.send(package, dest=source, tag=tags.INIT)
        logger.debug("Sent work package to worker %i" % source)
        active_workers += 1

    count_sample = 0
    counter = ChainCounter(n=n_samples, n_jobs=1, perc_disp=0.01, subject="samples")
    posterior_workers = manager.get_posterior_workers()
    logger.info("Posterior workers %s", list2string(posterior_workers))
    logger.info("Tuning worker betas every %i samples. \n" % beta_tune_interval)
    logger.info("Sampling ...")
    logger.info("------------")
    while True:

        m1 = num.empty(manager.step.lordering.size)
        comm.Recv(
            [m1, MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
        )
        source1 = status.Get_source()
        logger.debug("Got sample 1 from worker %i" % source1)

        m2 = num.empty(manager.step.lordering.size)
        comm.Recv(
            [m2, MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
        )
        source2 = status.Get_source()
        logger.debug("Got sample 2 from worker %i" % source2)

        # write results to trace if workers sample from posterior
        for source, m in zip([source1, source2], [m1, m2]):
            if source in posterior_workers:
                count_sample += 1
                counter(source)
                # print(manager.worker_a2l(m, source))
                master_trace.write(manager.worker_a2l(m, source), count_sample)
                steps_until_tune += 1

        m1, m2 = manager.propose_chain_swap(m1, m2, source1, source2)
        # beta updating
        if steps_until_tune >= beta_tune_interval:
            manager.tune_betas()
            steps_until_tune = 0

        if count_sample < n_samples:
            logger.debug("Sending states back to workers ...")
            for source in [source1, source2]:
                if not manager.worker_beta_updated(source1):
                    comm.Send(
                        [manager.get_beta(source), MPI.DOUBLE],
                        dest=source,
                        tag=tags.BETA,
                    )
                    manager.worker_beta_updated(source, check=True)

            comm.Send(m1, dest=source1, tag=tags.SAMPLE)
            comm.Send(m2, dest=source2, tag=tags.SAMPLE)
        else:
            logger.info("Requested number of samples reached!")
            master_trace.record_buffer()
            manager.dump_history(save_dir=stage_handler.stage_path(stage))
            break

    logger.info("Master finished! Chain complete!")
    logger.debug("Firing ...")
    for i in range(1, size):
        logger.debug("Sending pay cheque to %i" % i)
        comm.send(None, dest=i, tag=tags.EXIT)
        logger.debug("Fired worker %i" % i)
        active_workers -= 1

    logger.info("Feierabend! Sampling finished!")


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
    logger.debug("Entering worker process with rank %d on %s." % (comm.rank, name))
    comm.send(None, dest=0, tag=tags.READY)

    logger.debug("Worker %i receiving work package ..." % comm.rank)
    kwargs = comm.recv(source=0, tag=tags.INIT, status=status)
    logger.debug("Worker %i received package!" % comm.rank)

    try:
        step = kwargs["step"]
    except KeyError:
        raise ValueError("Step method not defined!")

    # do initial sampling
    result = sample_pt_chain(**kwargs)
    comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)

    # enter repeated sampling
    while True:
        # TODO: make transd-compatible
        data = num.empty(step.lordering.size, dtype=tconfig.floatX)
        comm.Recv([data, MPI.DOUBLE], tag=MPI.ANY_TAG, source=0, status=status)

        tag = status.Get_tag()
        if tag == tags.SAMPLE:
            lpoint = step.lij.a2l(data)
            start = step.lij.l2d(lpoint)
            kwargs["start"] = start
            # overwrite previous point in case got swapped
            kwargs["step"].chain_previous_lpoint[comm.rank] = lpoint
            result = sample_pt_chain(**kwargs)

            logger.debug("Worker %i attempting to send ..." % comm.rank)
            comm.Send([result, MPI.DOUBLE], dest=0, tag=tags.DONE)
            logger.debug("Worker %i sent message successfully ..." % comm.rank)

        elif tag == tags.BETA:
            logger.debug("Worker %i received beta: %f" % (comm.rank, data[0]))
            kwargs["step"].beta = data[0]

        elif tag == tags.EXIT:
            logger.debug("Worker %i went through EXIT!" % comm.rank)
            break


def sample_pt_chain(
    draws,
    step=None,
    start=None,
    trace=None,
    chain=0,
    tune=None,
    progressbar=True,
    model=None,
    random_seed=-1,
):
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
        n_steps = int(draws())
    else:
        n_steps = draws

    if step.cumulative_samples > step.burnin:
        update_proposal = True
    else:
        update_proposal = False

    step.n_steps = n_steps

    sampling = _iter_sample(
        n_steps,
        step,
        start,
        trace,
        chain,
        tune,
        model,
        random_seed,
        overwrite=False,
        update_proposal=update_proposal,
        keep_last=True,
    )

    try:
        for strace in sampling:
            pass

    except KeyboardInterrupt:
        raise

    if strace.buffer:
        rsample = step.lij.l2a(strace.buffer[-1][0])
    else:
        BufferError("No samples in buffer to return to master!")

    return rsample


def pt_sample(
    step,
    n_chains,
    n_samples=100000,
    start=None,
    swap_interval=(100, 300),
    beta_tune_interval=10000,
    n_workers_posterior=1,
    homepath="",
    progressbar=True,
    buffer_size=5000,
    buffer_thinning=1,
    model=None,
    rm_flag=False,
    resample=False,
    keep_tmp=False,
    record_worker_chains=False,
):
    """
    Parallel Tempering algorithm

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
    n_chains : int
        number of Markov Chains to use
    n_samples : int
        number of samples in the result trace, if reached sampling stops
    swap_interval : tuple
        interval for uniform random integer that determines the length
        of each MarkovChain on each worker. The chain end values of workers
        are proposed for swapping state and are written in the final trace
    beta_tune_interval : int
        Evaluate acceptance rate of chain swaps and tune betas similar
        to proposal step tuning
    n_workers_posterior : int
        number of workers that sample from the posterior distribution at beta=1
    homepath : string
        Result_folder for storing stages, will be created if not existing
    progressbar : bool
        Flag for displaying a progress bar
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    buffer_thinning : int
        every nth sample of the buffer is written to disk,
        default: 1 (no thinning)
    model : :class:`pymc3.Model`
        (optional if in `with` context) has to contain deterministic
        variable name defined under step.likelihood_name' that contains the
        model likelihood
    rm_flag : bool
        If True existing stage result folders are being deleted prior to
        sampling.
    resample : bool
        If True all the Markov Chains are starting sampling at the testvalue
    keep_tmp : bool
        If True the execution directory (under '/tmp/') is not being deleted
        after process finishes
    record_worker_chains : bool
        If True worker chain samples are written to disc using the specified
        backend trace objects (during sampler initialization).
        Very useful for debugging purposes. MUST be False for runs on
        distributed computing systems!
    """
    if n_chains < 2:
        raise ValueError("Parallel Tempering requires at least 2 Markov Chains!")

    if start is not None:
        if len(start) != step.n_chains:
            raise TypeError(
                "Argument `start` should have dicts equal the "
                "number of chains (step.N-chains)"
            )
        else:
            step.population = start

    sampler_args = [
        step,
        n_samples,
        swap_interval,
        beta_tune_interval,
        n_workers_posterior,
        homepath,
        progressbar,
        buffer_size,
        buffer_thinning,
        resample,
        rm_flag,
        record_worker_chains,
    ]

    project_dir = os.path.dirname(homepath)
    loglevel = getLevelName(logger.getEffectiveLevel()).lower()

    distributed.run_mpi_sampler(
        sampler_name="pt",
        model=model,
        sampler_args=sampler_args,
        keep_tmp=keep_tmp,
        n_jobs=n_chains,
        loglevel=loglevel,
        project_dir=project_dir,
    )


def _sample():
    # Define MPI message tags
    tags = distributed.enum("READY", "INIT", "DONE", "EXIT", "SAMPLE", "BETA")

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    logger.debug("communicator size %i" % comm.size)
    model = load_objects(distributed.pymc_model_name)
    with model:
        for i in range(comm.size):
            if i == comm.rank:
                logger.info("Working %i" % i)

        comm.Barrier()

        if comm.rank == 0:
            logger.info("Loading passed arguments ...")

            arguments = load_objects(distributed.mpiargs_name)
            args = [model] + arguments

            master_process(comm, tags, status, *args)
        else:
            worker_process(comm, tags, status)


if __name__ == "__main__":

    try:
        _, levelname, project_dir = sys.argv
    except ValueError:
        _, levelname = sys.argv
        project_dir = ""

    setup_logging(
        project_dir=project_dir, levelname=levelname, logfilename="BEAT_log.txt"
    )
    _sample()
