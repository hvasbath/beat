"""
Sequential Monte Carlo Sampler module;

Runs on any pymc3 model.
"""

import logging

import numpy as np
from pymc3.model import modelcontext

from beat import backend, utility

from .base import choose_proposal, init_stage, iter_parallel_chains, update_last_samples
from .metropolis import Metropolis

__all__ = ["SMC", "smc_sample"]


logger = logging.getLogger("smc")

sample_factor_final_stage = 1


class SMC(Metropolis):
    """
    Sequential Monte-Carlo sampler class.

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
    proposal_dist :
        :class:`pymc3.metropolis.Proposal`
        Type of proposal distribution, see
        :mod:`pymc3.step_methods.metropolis` for options
    tune : boolean
        Flag for adaptive scaling based on the acceptance rate
    coef_variation : scalar, float
        Coefficient of variation, determines the change of beta
        from stage to stage, i.e.indirectly the number of stages,
        low coef_variation --> slow beta change,
        results in many stages and vice verca (default: 1.)
    check_bound : boolean
        Check if current sample lies outside of variable definition
        speeds up computation as the forward model won't be executed
        default: True
    model : :class:`pymc3.Model`
        Optional model for sampling step.
        Defaults to None (taken from context).
    backend :  str
        type of backend to use for sample results storage, for alternatives
        see :class:`backend.backend:catalog`

    References
    ----------
    .. [Ching2007] Ching, J. and Chen, Y. (2007).
        Transitional Markov Chain Monte Carlo Method for Bayesian Model
        Updating, Model Class Selection, and Model Averaging.
        J. Eng. Mech., 10.1061/(ASCE)0733-9399(2007)133:7(816), 816-832.
        `link <http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399
        %282007%29133:7%28816%29>`__
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
        proposal_name="MultivariateNormal",
        backend="csv",
        coef_variation=1.0,
        **kwargs
    ):

        super(SMC, self).__init__(
            vars=vars,
            out_vars=out_vars,
            covariance=covariance,
            scale=scale,
            n_chains=n_chains,
            tune=tune,
            tune_interval=tune_interval,
            model=model,
            check_bound=check_bound,
            likelihood_name=likelihood_name,
            backend=backend,
            proposal_name=proposal_name,
            **kwargs
        )

        self.beta = 0

        self.coef_variation = coef_variation
        self.likelihoods = np.zeros(n_chains)

    def _sampler_state_blacklist(self):
        """
        Returns sampler attributes that are not saved.
        """
        bl = [
            "likelihoods",
            "check_bnd",
            "logp_forw",
            "bij",
            "lij",
            "ordering",
            "lordering",
            "proposal_samples_array",
            "vars",
            "_BlockedStep__newargs",
        ]
        return bl

    def calc_beta(self):
        """
        Calculate next tempering beta and importance weights based on
        current beta and sample likelihoods.

        Returns
        -------
        beta(m+1) : scalar, float
            tempering parameter of the next stage
        beta(m) : scalar, float
            tempering parameter of the current stage
        weights : :class:`numpy.ndarray`
            Importance weights (floats)
        """

        low_beta = self.beta
        up_beta = 2.0
        old_beta = self.beta

        while up_beta - low_beta > 1e-6:
            current_beta = (low_beta + up_beta) / 2.0
            temp = np.exp(
                (current_beta - self.beta) * (self.likelihoods - self.likelihoods.max())
            )
            cov_temp = np.std(temp) / np.mean(temp)
            if cov_temp > self.coef_variation:
                up_beta = current_beta
            else:
                low_beta = current_beta

        beta = current_beta
        weights = temp / np.sum(temp)
        return beta, old_beta, weights

    def calc_covariance(self):
        """
        Calculate trace covariance matrix based on importance weights.

        Returns
        -------
        cov : :class:`numpy.ndarray`
            weighted covariances (NumPy > 1.10. required)
        """
        cov = np.cov(
            self.array_population, aweights=self.weights.ravel(), bias=False, rowvar=0
        )

        cov = utility.ensure_cov_psd(cov)
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError("Sample covariances contains Inf or NaN!")
        return cov

    def select_end_points(self, mtrace):
        """
        Read trace results (variables and model likelihood) and take end points
        for each chain and set as start population for the next stage.

        Parameters
        ----------
        mtrace : :class:`pymc3.backend.base.MultiTrace`

        Returns
        -------
        population : list
            of :func:`pymc3.Point` dictionaries
        array_population : :class:`numpy.ndarray`
            Array of trace end-points
        likelihoods : :class:`numpy.ndarray`
            Array of likelihoods of the trace end-points
        """

        array_population = np.zeros((self.n_chains, self.ordering.size))

        n_steps = len(mtrace)

        # collect end points of each chain and put into array
        for var, slc, shp, _ in self.ordering.vmap:
            slc_population = mtrace.get_values(
                varname=var, burn=n_steps - 1, combine=True
            )

            if len(shp) == 0:
                array_population[:, slc] = np.atleast_2d(slc_population).T
            else:
                array_population[:, slc] = slc_population

        # get likelihoods
        likelihoods = mtrace.get_values(
            varname=self.likelihood_name, burn=n_steps - 1, combine=True
        )
        population = []

        # map end array_endpoints to dict points
        for i in range(self.n_chains):
            population.append(self.bij.rmap(array_population[i, :]))

        return population, array_population, likelihoods

    def get_chain_previous_lpoint(self, mtrace):
        """
        Read trace results and take end points for each chain and set as
        previous chain result for comparison of metropolis select.

        Parameters
        ----------
        mtrace : :class:`pymc3.backend.base.MultiTrace`

        Returns
        -------
        chain_previous_lpoint : list
            all unobservedRV values, including dataset likelihoods
        """

        array_population = np.zeros((self.n_chains, self.lordering.size))

        n_steps = len(mtrace)

        for _, slc, shp, _, var in self.lordering.vmap:

            slc_population = mtrace.get_values(
                varname=var, burn=n_steps - 1, combine=True
            )

            if len(shp) == 0:
                array_population[:, slc] = np.atleast_2d(slc_population).T
            else:
                array_population[:, slc] = slc_population

        chain_previous_lpoint = []

        # map end array_endpoints to list lpoints and apply resampling
        for r_idx in self.resampling_indexes:
            chain_previous_lpoint.append(self.lij.a2l(array_population[r_idx, :]))

        return chain_previous_lpoint

    def get_map_end_points(self):
        """
        Calculate mean of the end-points and return point.

        Returns
        -------
        Dictionary of trace variables
        """
        idx = self.likelihoods.flatten().argmax()
        return self.bij.rmap(self.array_population[idx, :])

    def resample(self):
        """
        Resample pdf based on importance weights.
        based on Kitagawas deterministic resampling algorithm.

        Returns
        -------
        outindex : :class:`numpy.ndarray`
            Array of resampled trace indexes
        """

        parents = np.arange(self.n_chains)
        N_childs = np.zeros(self.n_chains, dtype=int)

        cum_dist = np.cumsum(self.weights)
        aux = np.random.rand(1)
        u = parents + aux
        u /= self.n_chains
        j = 0
        for i in parents:
            while u[i] > cum_dist[j]:
                j += 1

            N_childs[j] += 1

        indx = 0
        outindx = np.zeros(self.n_chains, dtype=int)
        for i in parents:
            if N_childs[i] > 0:
                for j in range(indx, (indx + N_childs[i])):
                    outindx[j] = parents[i]

            indx += N_childs[i]

        return outindx

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def smc_sample(
    n_steps,
    step=None,
    start=None,
    homepath=None,
    stage=0,
    n_jobs=1,
    progressbar=False,
    buffer_size=5000,
    buffer_thinning=1,
    model=None,
    update=None,
    random_seed=None,
    rm_flag=False,
):
    """
    Sequential Monte Carlo samlping

    Samples the solution space with n_chains of Metropolis chains, where each
    chain has n_steps iterations. Once finished, the sampled traces are
    evaluated:

    (1) Based on the likelihoods of the final samples, chains are weighted
    (2) the weighted covariance of the ensemble is calculated and set as new
        proposal distribution
    (3) the variation in the ensemble is calculated and the next tempering
        parameter (beta) calculated
    (4) New n_chains Metropolis chains are seeded on the traces with high
        weight for n_steps iterations
    (5) Repeat until beta > 1.

    Parameters
    ----------
    n_steps : int
        The number of samples to draw for each Markov-chain per stage
    step : :class:`SMC`
        SMC initialisation object
    start : List of dictionaries
        with length of (n_chains)
        Starting points in parameter space (or partial point)
        Defaults to random draws from variables (defaults to empty dict)
    stage : int
        Stage where to start or continue the calculation. It is possible to
        continue after completed stages (stage should be the number of the
        completed stage + 1). If None the start will be at stage = 0.
    n_jobs : int
        The number of cores to be used in parallel. Be aware that theano has
        internal parallelisation. Sometimes this is more efficient especially
        for simple models.
        step.n_chains / n_jobs has to be an integer number!
    homepath : string
        Result_folder for storing stages, will be created if not existing.
    progressbar : bool
        Flag for displaying a progress bar
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    buffer_thinning : int
        every nth sample of the buffer is written to disk
        default: 1 (no thinning)
    model : :class:`pymc3.Model`
        (optional if in `with` context) has to contain deterministic
        variable name defined under step.likelihood_name' that contains the
        model likelihood
    update : :py:class:`models.Problem`
        Problem object that contains all the observed data and (if applicable)
        covariances to be updated each transition step.
    rm_flag : bool
        If True existing stage result folders are being deleted prior to
        sampling.

    References
    ----------
    .. [Minson2013] Minson, S. E. and Simons, M. and Beck, J. L., (2013),
        Bayesian inversion for finite fault earthquake source models
        I- Theory and algorithm. Geophysical Journal International, 2013,
        194(3), pp.1701-1726,
        `link <https://gji.oxfordjournals.org/content/194/3/1701.full>`__
    """

    model = modelcontext(model)
    step.n_steps = int(n_steps)

    if n_steps < 1:
        raise TypeError("Argument `n_steps` should be above 0.", exc_info=1)

    if step is None:
        raise TypeError("Argument `step` has to be a SMC step object.")

    if homepath is None:
        raise TypeError("Argument `homepath` should be path to result_directory.")

    if n_jobs > 1:
        if not (step.n_chains / float(n_jobs)).is_integer():
            raise ValueError("n_chains / n_jobs has to be a whole number!")

    if start is not None:
        if len(start) != step.n_chains:
            raise TypeError(
                "Argument `start` should have dicts equal the "
                "number of chains (step.N-chains)"
            )
        else:
            step.population = start

    if not any(step.likelihood_name in var.name for var in model.deterministics):
        raise TypeError(
            "Model (deterministic) variables need to contain "
            "a variable %s "
            "as defined in `step`." % step.likelihood_name
        )

    stage_handler = backend.SampleStage(homepath, backend=step.backend)

    chains, step, update = init_stage(
        stage_handler=stage_handler,
        step=step,
        stage=stage,
        progressbar=progressbar,
        buffer_thinning=buffer_thinning,
        update=update,
        model=model,
        rm_flag=rm_flag,
    )

    with model:
        while step.beta < 1.0:
            if step.stage == 0:
                # Initial stage
                logger.info("Sample initial stage: ...")
                draws = 1
            else:
                draws = n_steps

            logger.info("Beta: %f Stage: %i" % (step.beta, step.stage))

            # Metropolis sampling intermediate stages
            chains = stage_handler.clean_directory(step.stage, chains, rm_flag)

            sample_args = {
                "draws": draws,
                "step": step,
                "stage_path": stage_handler.stage_path(step.stage),
                "progressbar": progressbar,
                "model": model,
                "n_jobs": n_jobs,
                "chains": chains,
                "buffer_size": buffer_size,
                "buffer_thinning": buffer_thinning,
            }

            mtrace = iter_parallel_chains(**sample_args)

            (
                step.population,
                step.array_population,
                step.likelihoods,
            ) = step.select_end_points(mtrace)

            if update is not None:
                logger.info("Updating Covariances ...")
                map_pt = step.get_map_end_points()
                update.update_weights(map_pt, n_jobs=n_jobs)
                mtrace = update_last_samples(
                    homepath, step, progressbar, model, n_jobs, rm_flag
                )
                (
                    step.population,
                    step.array_population,
                    step.likelihoods,
                ) = step.select_end_points(mtrace)

            step.beta, step.old_beta, step.weights = step.calc_beta()

            if step.beta > 1.0:
                logger.info("Beta > 1.: %f" % step.beta)
                step.beta = 1.0
                save_sampler_state(step, update, stage_handler)

                if stage == -1:
                    chains = []
                else:
                    chains = None
            else:
                step.covariance = step.calc_covariance()
                step.proposal_dist = choose_proposal(
                    step.proposal_name, scale=step.covariance
                )
                step.resampling_indexes = step.resample()
                step.chain_previous_lpoint = step.get_chain_previous_lpoint(mtrace)

                save_sampler_state(step, update, stage_handler)

                step.stage += 1
                del mtrace

        # Metropolis sampling final stage
        draws = n_steps * sample_factor_final_stage
        logger.info("Sample final stage with n_steps %i " % draws)
        step.stage = -1

        temp = np.exp((1 - step.old_beta) * (step.likelihoods - step.likelihoods.max()))
        step.weights = temp / np.sum(temp)
        step.covariance = step.calc_covariance()
        step.proposal_dist = choose_proposal(step.proposal_name, scale=step.covariance)

        step.resampling_indexes = step.resample()
        step.chain_previous_lpoint = step.get_chain_previous_lpoint(mtrace)

        sample_args["draws"] = draws
        sample_args["step"] = step
        sample_args["stage_path"] = stage_handler.stage_path(step.stage)
        sample_args["chains"] = chains
        iter_parallel_chains(**sample_args)

        save_sampler_state(step, update, stage_handler)
        logger.info("Finished sampling!")


def save_sampler_state(step, update, stage_handler):
    logger.info("Saving sampler state ...")
    if update is not None:
        weights = update.get_weights()
    else:
        weights = None

    outparam_list = [step.get_sampler_state(), weights]
    stage_handler.dump_atmip_params(step.stage, outparam_list)


def tune(acc_rate):
    """
    Tune adaptively based on the acceptance rate.

    Parameters
    ----------
    acc_rate: scalar, float
        Acceptance rate of the Metropolis sampling

    Returns
    -------
    scaling: scalar float
    """

    # a and b after Muto & Beck 2008 .
    a = 1.0 / 9
    b = 8.0 / 9
    return np.power((a + (b * acc_rate)), 2)
