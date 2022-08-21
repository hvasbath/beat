"""
Metropolis algorithm module, wrapping the pymc3 implementation.

Provides the possibility to update the involved covariance matrixes within
the course of sampling the chain.
"""

import logging
import os
import shutil
from copy import deepcopy
from time import time

import numpy as num
from pymc3.backends import text
from pymc3.model import Point, modelcontext
from pymc3.step_methods.metropolis import metrop_select
from pymc3.step_methods.metropolis import tune as step_tune
from pymc3.theanof import inputvars, make_shared_replacements
from pymc3.vartypes import discrete_types
from pyrocko import util
from theano import config as tconfig

from beat import backend, utility
from beat.covariance import init_proposal_covariance

from .base import (
    choose_proposal,
    init_stage,
    iter_parallel_chains,
    logp_forw,
    multivariate_proposals,
    update_last_samples,
)

__all__ = ["metropolis_sample", "get_trace_stats", "get_final_stage", "Metropolis"]


logger = logging.getLogger("metropolis")


class Metropolis(backend.ArrayStepSharedLLK):
    """
    Metropolis-Hastings sampler

    Parameters
    ----------
    vars : list
        List of variables for sampler
    out_vars : list
        List of output variables for trace recording. If empty unobserved_RVs
        are taken.
    n_chains : int
        Number of chains per stage has to be a large number
        of number of n_jobs (processors to be used) on the machine.
    scaling : float
        Factor applied to the proposal distribution i.e. the step size of the
        Markov Chain
    covariance : :class:`numpy.ndarray`
        (n_chains x n_chains) for MutlivariateNormal, otherwise (n_chains)
        Initial Covariance matrix for proposal distribution,
        if None - identity matrix taken
    likelihood_name : string
        name of the :class:`pymc3.determinsitic` variable that contains the
        model likelihood - defaults to 'like'
    backend :  str
        type of backend to use for sample results storage, for alternatives
        see :class:`backend.backend:catalog`
    proposal_dist :
        :class:`pymc3.metropolis.Proposal`
        Type of proposal distribution, see
        :mod:`pymc3.step_methods.metropolis` for options
    tune : boolean
        Flag for adaptive scaling based on the acceptance rate
    model : :class:`pymc3.Model`
        Optional model for sampling step.
        Defaults to None (taken from context).
    """

    default_blocked = True

    def __init__(
        self,
        vars=None,
        out_vars=None,
        covariance=None,
        scale=1.0,
        n_chains=100,
        tune=True,
        tune_interval=100,
        model=None,
        check_bound=True,
        likelihood_name="like",
        backend="csv",
        proposal_name="MultivariateNormal",
        **kwargs
    ):

        model = modelcontext(model)

        if vars is None:
            vars = model.vars

        vars = inputvars(vars)

        if out_vars is None:
            out_vars = model.unobserved_RVs

        out_varnames = [out_var.name for out_var in out_vars]

        self.scaling = utility.scalar2floatX(num.atleast_1d(scale))

        self.tune = tune
        self.check_bound = check_bound
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval

        self.stage_sample = 0
        self.cumulative_samples = 0
        self.accepted = 0

        self.beta = 1.0
        self.stage = 0
        self.chain_index = 0

        # needed to use the same parallel implementation function as for SMC
        self.resampling_indexes = num.arange(n_chains)
        self.n_chains = n_chains

        self.likelihood_name = likelihood_name
        self._llk_index = out_varnames.index(likelihood_name)
        self.backend = backend
        self.discrete = num.concatenate(
            [[v.dtype in discrete_types] * (v.dsize or 1) for v in vars]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # create initial population
        self.population = []
        self.array_population = num.zeros(n_chains)
        logger.info(
            "Creating initial population for {}" " chains ...".format(self.n_chains)
        )
        for i in range(self.n_chains):
            self.population.append(
                Point({v.name: v.random() for v in vars}, model=model)
            )

        self.population[0] = model.test_point

        shared = make_shared_replacements(vars, model)
        self.logp_forw = logp_forw(out_vars, vars, shared)
        self.check_bnd = logp_forw([model.varlogpt], vars, shared)

        super(Metropolis, self).__init__(vars, out_vars, shared)

        # init proposal
        if covariance is None and proposal_name in multivariate_proposals:
            t0 = time()
            self.covariance = init_proposal_covariance(
                bij=self.bij, vars=vars, model=model, pop_size=1000
            )
            t1 = time()
            logger.info("Time for proposal covariance init: %f" % (t1 - t0))
            scale = self.covariance
        elif covariance is None:
            scale = num.ones(sum(v.dsize for v in vars))
        else:
            scale = covariance

        self.proposal_name = proposal_name
        self.proposal_dist = choose_proposal(self.proposal_name, scale=scale)
        self.proposal_samples_array = self.proposal_dist(n_chains)

        self.chain_previous_lpoint = [[]] * self.n_chains
        self._tps = None

    def _sampler_state_blacklist(self):
        """
        Returns sampler attributes that are not saved.
        """
        bl = [
            "check_bnd",
            "logp_forw",
            "proposal_samples_array",
            "vars",
            "bij",
            "lij",
            "ordering",
            "lordering",
            "_BlockedStep__newargs",
        ]
        return bl

    def get_sampler_state(self):
        """
        Return dictionary of sampler state.

        Returns
        -------
        dict of sampler state
        """

        blacklist = self._sampler_state_blacklist()
        return {k: v for k, v in self.__dict__.items() if k not in blacklist}

    def apply_sampler_state(self, state):
        """
        Update sampler state to given state
        (obtained by 'get_sampler_state')

        Parameters
        ----------
        state : dict
            with sampler parameters
        """
        for k, v in state.items():
            setattr(self, k, v)

    def time_per_sample(self, n_points=10):
        if not self._tps:
            tps = num.zeros((n_points))
            for i in range(n_points):
                q = self.bij.map(self.population[i])
                t0 = time()
                self.logp_forw(q)
                t1 = time()
                tps[i] = t1 - t0
            self._tps = tps.mean()
        return self._tps

    def astep(self, q0):
        if self.stage == 0:
            l_new = self.logp_forw(q0)
            if not num.isfinite(l_new[self._llk_index]):
                raise ValueError(
                    "Got NaN in likelihood evaluation! "
                    "Invalid model definition? "
                    "Or starting point outside prior bounds!"
                )

            q_new = q0

        else:
            if self.stage_sample == 0:
                self.proposal_samples_array = self.proposal_dist(self.n_steps).astype(
                    tconfig.floatX
                )

            if not self.steps_until_tune and self.tune:
                # Tune scaling parameter
                logger.debug(
                    "Tuning: Chain_%i step_%i" % (self.chain_index, self.stage_sample)
                )

                self.scaling = utility.scalar2floatX(
                    step_tune(self.scaling, self.accepted / float(self.tune_interval))
                )

                # Reset counter
                self.steps_until_tune = self.tune_interval
                self.accepted = 0

            logger.debug(
                "Get delta: Chain_%i step_%i" % (self.chain_index, self.stage_sample)
            )
            delta = self.proposal_samples_array[self.stage_sample, :] * self.scaling

            if self.any_discrete:
                if self.all_discrete:
                    delta = num.round(delta, 0)
                    q0 = q0.astype(int)
                    q = (q0 + delta).astype(int)
                else:
                    delta[self.discrete] = num.round(delta[self.discrete], 0).astype(
                        int
                    )
                    q = q0 + delta
                    q = q[self.discrete].astype(int)
            else:

                q = q0 + delta

            try:
                l0 = self.chain_previous_lpoint[self.chain_index]
                llk0 = l0[self._llk_index]
            except IndexError:
                l0 = self.logp_forw(q0)
                self.chain_previous_lpoint[self.chain_index] = l0
                llk0 = l0[self._llk_index]

            if self.check_bound:
                logger.debug(
                    "Checking bound: Chain_%i step_%i"
                    % (self.chain_index, self.stage_sample)
                )
                varlogp = self.check_bnd(q)

                if num.isfinite(varlogp):
                    logger.debug(
                        "Calc llk: Chain_%i step_%i"
                        % (self.chain_index, self.stage_sample)
                    )

                    lp = self.logp_forw(q)

                    logger.debug(
                        "Select llk: Chain_%i step_%i"
                        % (self.chain_index, self.stage_sample)
                    )

                    tempered_llk_ratio = self.beta * (
                        lp[self._llk_index] - l0[self._llk_index]
                    )
                    q_new, accepted = metrop_select(tempered_llk_ratio, q, q0)

                    if accepted:
                        logger.debug(
                            "Accepted: Chain_%i step_%i"
                            % (self.chain_index, self.stage_sample)
                        )
                        logger.debug(
                            "proposed: %f previous: %f" % (lp[self._llk_index], llk0)
                        )
                        self.accepted += 1
                        l_new = lp
                        self.chain_previous_lpoint[self.chain_index] = l_new
                    else:
                        logger.debug(
                            "Rejected: Chain_%i step_%i"
                            % (self.chain_index, self.stage_sample)
                        )
                        logger.debug(
                            "proposed: %f previous: %f"
                            % (lp[self._llk_index], l0[self._llk_index])
                        )
                        l_new = l0
                else:
                    q_new = q0
                    l_new = l0

            else:
                logger.debug(
                    "Calc llk: Chain_%i step_%i" % (self.chain_index, self.stage_sample)
                )

                lp = self.logp_forw(q)

                logger.debug(
                    "Select: Chain_%i step_%i" % (self.chain_index, self.stage_sample)
                )
                q_new, accepted = metrop_select(
                    self.beta * (lp[self._llk_index] - llk0), q, q0
                )

                if accepted:
                    self.accepted += 1
                    l_new = lp
                    self.chain_previous_lpoint[self.chain_index] = l_new
                else:
                    l_new = l0

            logger.debug(
                "Counters: Chain_%i step_%i" % (self.chain_index, self.stage_sample)
            )
            self.steps_until_tune -= 1
            self.stage_sample += 1
            self.cumulative_samples += 1

            # reset sample counter
            if self.stage_sample == self.n_steps:
                self.stage_sample = 0

            logger.debug(
                "End step: Chain_%i step_%i" % (self.chain_index, self.stage_sample)
            )

        return q_new, l_new


def get_final_stage(homepath, n_stages, model):
    """
    Combine Metropolis results into final stage to get one single chain for
    plotting results.
    """

    util.ensuredir(homepath)

    mtraces = []
    for stage in range(n_stages):
        logger.info("Loading Metropolis stage %i" % stage)
        stage_outpath = os.path.join(homepath, "stage_%i" % stage)

        mtraces.append(backend.load(name=stage_outpath, model=model))

    ctrace = backend.concatenate_traces(mtraces)
    outname = os.path.join(homepath, "stage_final")

    if os.path.exists(outname):
        logger.info("Removing existing previous final stage!")
        shutil.rmtree(outname)

    util.ensuredir(outname)
    logger.info("Creating final Metropolis stage")

    text.dump(name=outname, trace=ctrace)


def metropolis_sample(
    n_steps=10000,
    homepath=None,
    start=None,
    progressbar=False,
    rm_flag=False,
    buffer_size=5000,
    buffer_thinning=1,
    step=None,
    model=None,
    n_jobs=1,
    update=None,
    burn=0.5,
    thin=2,
):
    """
    Execute Metropolis algorithm repeatedly depending on the number of chains.
    """

    # hardcoded stage here as there are no stages
    stage = -1
    model = modelcontext(model)
    step.n_steps = int(n_steps)

    if n_steps < 1:
        raise TypeError("Argument `n_steps` should be above 0.", exc_info=1)

    if step is None:
        raise TypeError("Argument `step` has to be a Metropolis step object.")

    if homepath is None:
        raise TypeError("Argument `homepath` should be path to result_directory.")

    if n_jobs > 1:
        if not (step.n_chains / float(n_jobs)).is_integer():
            raise Exception("n_chains / n_jobs has to be a whole number!")

    if start is not None:
        if len(start) != step.n_chains:
            raise Exception(
                "Argument `start` should have dicts equal the "
                "number of chains (step.N-chains)"
            )
        else:
            step.population = start

    if not any(step.likelihood_name in var.name for var in model.deterministics):
        raise Exception(
            "Model (deterministic) variables need to contain "
            "a variable %s as defined in `step`." % step.likelihood_name
        )

    stage_handler = backend.SampleStage(homepath, backend=step.backend)

    util.ensuredir(homepath)

    chains, step, update = init_stage(
        stage_handler=stage_handler,
        step=step,
        stage=stage,  # needs zero otherwise tries to load stage_0 results
        progressbar=progressbar,
        update=update,
        model=model,
        rm_flag=rm_flag,
    )

    with model:

        chains = stage_handler.clean_directory(step.stage, chains, rm_flag)

        logger.info("Sampling stage ...")

        draws = n_steps

        step.stage = stage

        sample_args = {
            "draws": draws,
            "step": step,
            "stage_path": stage_handler.stage_path(step.stage),
            "progressbar": progressbar,
            "model": model,
            "n_jobs": n_jobs,
            "buffer_size": buffer_size,
            "buffer_thinning": buffer_thinning,
            "chains": chains,
        }

        mtrace = iter_parallel_chains(**sample_args)

        if step.proposal_name == "MultivariateNormal":
            pdict, step.covariance = get_trace_stats(mtrace, step, burn, thin, n_jobs)

            step.proposal_dist = choose_proposal(
                step.proposal_name, scale=step.covariance
            )

        if update is not None:
            logger.info("Updating Covariances ...")
            update.update_weights(pdict["dist_mean"], n_jobs=n_jobs)

            mtrace = update_last_samples(
                homepath, step, progressbar, model, n_jobs, rm_flag
            )

        elif update is not None and stage == 0:
            update.engine.close_cashed_stores()

        outparam_list = [step.get_sampler_state(), update]
        stage_handler.dump_atmip_params(step.stage, outparam_list)


def get_trace_stats(mtrace, step, burn=0.5, thin=2, n_jobs=1):
    """
    Get mean value of trace variables and return point.

    Parameters
    ----------
    mtrace : :class:`pymc3.backends.base.MultiTrace`
        Multitrace sampling result
    step : initialised :class:`smc.SMC` sampler object
    burn : float
        Burn-in parameter to throw out samples from the beginning of the trace
    thin : int
        Thinning of samples in the trace

    Returns
    -------
    dict with points, covariance matrix
    """

    n_steps = len(mtrace)

    array_population = num.zeros(
        (n_jobs * int(num.ceil(n_steps * (1 - burn) / thin)), step.ordering.size)
    )

    # collect end points of each chain and put into array
    for var, slc, shp, _ in step.ordering.vmap:
        samples = mtrace.get_values(
            varname=var, burn=int(burn * n_steps), thin=thin, combine=True
        )

        if len(shp) == 0:
            array_population[:, slc] = num.atleast_2d(samples).T
        else:
            array_population[:, slc] = samples

    llks = mtrace.get_values(
        varname=step.likelihood_name, burn=int(burn * n_steps), thin=thin, combine=True
    )

    posterior_idxs = utility.get_fit_indexes(llks)
    d = {}
    for k, v in posterior_idxs.items():
        d[k] = step.bij.rmap(array_population[v, :])

    d["dist_mean"] = step.bij.rmap(array_population.mean(axis=0))
    avar = array_population.var(axis=0)
    if avar.sum() == 0.0:
        logger.warn("Trace std not valid not enough samples! Use 1.")
        avar = 1.0

    cov = num.eye(step.ordering.size) * avar
    return d, cov
