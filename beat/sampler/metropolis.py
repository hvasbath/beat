"""
Metropolis algorithm module, wrapping the pymc implementation.

Provides the possibility to update the involved covariance matrixes within
the course of sampling the chain.
"""

import logging
import warnings
from time import time

import numpy as num
from pymc.model import Point, modelcontext
from pymc.pytensorf import inputvars, make_shared_replacements
from pymc.sampling import sample_prior_predictive

# from pymc.smc.kernels import _logp_forw
from pymc.step_methods.metropolis import metrop_select
from pymc.step_methods.metropolis import tune as step_tune
from pymc.vartypes import discrete_types
from pyrocko import util
from pytensor import config as tconfig

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

__all__ = ["metropolis_sample", "get_trace_stats", "Metropolis"]


logger = logging.getLogger("metropolis")


class Metropolis(backend.ArrayStepSharedLLK):
    """
    Metropolis-Hastings sampler

    Parameters
    ----------
    value_vars : list
        List of variables for sampler
    n_chains : int
        Number of chains per stage has to be a large number
        of number of n_jobs (processors to be used) on the machine.
    scaling : float
        Factor applied to the proposal distribution i.e. the step size of the
        Markov Chain
    likelihood_name : string
        name of the :class:`pymc.determinsitic` variable that contains the
        model likelihood - defaults to 'like'
    backend :  str
        type of backend to use for sample results storage, for alternatives
        see :class:`backend.backend:catalog`
    proposal_dist :
        :class:`beat.sampler.base.Proposal` Type of proposal distribution
    tune : boolean
        Flag for adaptive scaling based on the acceptance rate
    model : :class:`pymc.model.Model`
        Optional model for sampling step.
        Defaults to None (taken from context).
    """

    default_blocked = True

    def __init__(
        self,
        value_vars=None,
        scale=1.0,
        n_chains=100,
        tune=True,
        tune_interval=100,
        model=None,
        check_bound=True,
        likelihood_name="like",
        backend="csv",
        proposal_name="MultivariateNormal",
        **kwargs,
    ):
        model = modelcontext(model)
        self.likelihood_name = likelihood_name
        self.proposal_name = proposal_name
        self.population = None

        if value_vars is None:
            value_vars = model.value_vars

        self.value_vars = inputvars(value_vars)

        self.scaling = utility.scalar2floatX(scale)

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
        self.backend = backend

        # initial point comes in reversed order for whatever reason
        # rearrange to order of value_vars
        init_point = model.initial_point()
        self.test_point = {
            val_var.name: init_point[val_var.name] for val_var in self.value_vars
        }

        self.initialize_population(model)
        self.compile_model_graph(model)
        self.initialize_proposal(model)

    def initialize_population(self, model):
        # create initial population from prior
        logger.info(
            "Creating initial population for {} chains ...".format(self.n_chains)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="The effect of Potentials"
            )
            var_names = [value_var.name for value_var in self.value_vars]
            prior_draws = sample_prior_predictive(
                samples=self.n_chains,
                var_names=var_names,
                model=model,
                return_inferencedata=False,
            )

        self.array_population = num.zeros(self.n_chains)
        self.population = []
        for i in range(self.n_chains):
            self.population.append(
                Point({v_name: prior_draws[v_name][i] for v_name in var_names})
            )

        self.population[0] = self.test_point

    def compile_model_graph(self, model):
        logger.info("Compiling model graph ...")
        shared = make_shared_replacements(self.test_point, self.value_vars, model)

        # collect all RVs and deterministics to write to file
        # out_vars = model.deterministics
        out_vars = model.unobserved_RVs
        out_varnames = [out_var.name for out_var in out_vars]
        self._llk_index = out_varnames.index(self.likelihood_name)

        # plot modelgraph
        # model_to_graphviz(model).view()

        in_rvs = [model.values_to_rvs[val_var] for val_var in self.value_vars]

        self.logp_forw_func = logp_forw(
            point=self.test_point,
            out_vars=out_vars,
            in_vars=in_rvs,  # values of dists
            shared=shared,
        )

        self.prior_logp_func = logp_forw(
            point=self.test_point,
            out_vars=[model.varlogp],
            in_vars=self.value_vars,  # logp of dists
            shared=shared,
        )

        # determine if there are discrete variables
        self.discrete = num.concatenate(
            [
                num.atleast_1d([v.dtype in discrete_types] * (v.size or 1))
                for v in self.test_point.values()
            ]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        super(Metropolis, self).__init__(self.value_vars, out_vars, shared)

    def initialize_proposal(self, model):
        # init proposal
        logger.info("Initializing proposal distribution ...%s", self.proposal_name)
        if self.proposal_name in multivariate_proposals:
            if self.population is None:
                raise ValueError("Sampler population needs to be initialised first!")

            t0 = time()
            self.covariance = init_proposal_covariance(
                bij=self.bij, population=self.population
            )
            t1 = time()
            logger.info("Time for proposal covariance init: %f" % (t1 - t0))
            scale = self.covariance
        else:
            scale = num.ones(sum(v.size for v in self.test_point.values()))

        self.proposal_dist = choose_proposal(self.proposal_name, scale=scale)
        self.proposal_samples_array = self.proposal_dist(self.n_chains)

        self.chain_previous_lpoint = [[]] * self.n_chains
        self._tps = None

    def _sampler_state_blacklist(self):
        """
        Returns sampler attributes that are not saved.
        """
        bl = [
            "check_bnd",
            "logp_forw_func",
            "proposal_samples_array",
            "value_vars",
            "bij",
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
                self.logp_forw_func(q.data)
                t1 = time()
                tps[i] = t1 - t0
            self._tps = tps.mean()
        return self._tps

    def astep(self, q0):
        if self.stage == 0:
            l_new = self.logp_forw_func(q0)
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
                l0 = self.logp_forw_func(q0)
                self.chain_previous_lpoint[self.chain_index] = l0
                llk0 = l0[self._llk_index]

            if self.check_bound:
                logger.debug(
                    "Checking bound: Chain_%i step_%i"
                    % (self.chain_index, self.stage_sample)
                )
                # print("before prior test", q)
                priorlogp = self.prior_logp_func(q)
                # print("prior", priorlogp)
                if num.isfinite(priorlogp):
                    logger.debug(
                        "Calc llk: Chain_%i step_%i"
                        % (self.chain_index, self.stage_sample)
                    )
                    # print("previous sample", q0)
                    lp = self.logp_forw_func(q)
                    logger.debug(
                        "Select llk: Chain_%i step_%i"
                        % (self.chain_index, self.stage_sample)
                    )
                    # print("current sample", q)
                    tempered_llk_ratio = self.beta * (
                        lp[self._llk_index] - l0[self._llk_index]
                    )
                    q_new, accepted = metrop_select(tempered_llk_ratio, q, q0)
                    # print("accepted:", q_new)
                    # print("-----------------------------------")
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

                lp = self.logp_forw_func(q)

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
        stage_handler.dump_smc_params(step.stage, outparam_list)


def get_trace_stats(mtrace, step, burn=0.5, thin=2, n_jobs=1):
    """
    Get mean value of trace variables and return point.

    Parameters
    ----------
    mtrace : :class:`pymc.backends.base.MultiTrace`
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
