"""
Metropolis algorithm module, wrapping the pymc3 implementation.
Provides the possibility to update the involved covariance matrixes within
the course of sampling the chain.
"""

import shutil
import os
import pymc3 as pm
import logging

import numpy as num

from beat import backend, utility
from beat.atmcmc import init_stage, _iter_parallel_chains, choose_proposal
from beat.config import sample_p_outname

from pyrocko import util

__all__ = ['Metropolis_sample', 'get_trace_stats', 'get_final_stage']

logger = logging.getLogger('ATMCMC')


def get_final_stage(homepath, n_stages, model):
    """
    Combine Metropolis results into final stage to get one single chain for
    plotting results.
    """

    util.ensuredir(homepath)

    mtraces = []
    for stage in range(n_stages):
        logger.info('Loading Metropolis stage %i' % stage)
        stage_outpath = os.path.join(homepath, 'stage_%i' % stage)

        mtraces.append(backend.load(
                name=stage_outpath, model=model))

    ctrace = backend.concatenate_traces(mtraces)
    outname = os.path.join(homepath, 'stage_final')

    if os.path.exists(outname):
        logger.info('Removing existing previous final stage!')
        shutil.rmtree(outname)

    util.ensuredir(outname)
    logger.info('Creating final Metropolis stage')

    pm.backends.text.dump(name=outname, trace=ctrace)


def Metropolis_sample(n_stages=10, n_steps=10000, trace=None, start=None,
            progressbar=False, stage=None, rm_flag=False,
            step=None, model=None, n_jobs=1, update=None, burn=0.5, thin=2):
    """
    Execute Metropolis algorithm repeatedly depending on the number of stages.
    The start point of each stage set to the end point of the previous stage.
    Update covariances if given.
    """

    model = pm.modelcontext(model)
    step.n_steps = int(n_steps)

    if n_steps < 1:
        raise Exception('Argument `n_steps` should be above 0.', exc_info=1)

    if step is None:
        raise Exception('Argument `step` has to be a TMCMC step object.')

    if trace is None:
        raise Exception('Argument `trace` should be path to result_directory.')

    if n_jobs > 1:
        if not (step.n_chains / float(n_jobs)).is_integer():
            raise Exception('n_chains / n_jobs has to be a whole number!')

    if start is not None:
        if len(start) != step.n_chains:
            raise Exception('Argument `start` should have dicts equal the '
                            'number of chains (step.N-chains)')
        else:
            step.population = start

    if not any(
            step.likelihood_name in var.name for var in model.deterministics):
            raise Exception('Model (deterministic) variables need to contain '
                            'a variable %s '
                            'as defined in `step`.' % step.likelihood_name)

    homepath = trace

    util.ensuredir(homepath)

    chains, step, update = init_stage(
        homepath=homepath,
        step=step,
        stage=stage,
        n_jobs=n_jobs,
        progressbar=progressbar,
        update=update,
        model=model,
        rm_flag=rm_flag)

    # set beta to 1 - standard Metropolis sampling
    step.beta = 1.
    step.n_jobs = n_jobs

    with model:

        for s in range(int(stage), n_stages):

            stage_path = os.path.join(homepath, 'stage_%i' % s)
            logger.info('Sampling stage %s' % stage_path)

            if s == 0:
                draws = 1
            else:
                draws = n_steps

            if not os.path.exists(stage_path):
                chains = None

            step.stage = s

            sample_args = {
                    'draws': draws,
                    'step': step,
                    'stage_path': stage_path,
                    'progressbar': progressbar,
                    'model': model,
                    'n_jobs': n_jobs,
                    'chains': chains}

            _iter_parallel_chains(**sample_args)

            mtrace = backend.load(stage_path, model)

            step.population, step.array_population, step.likelihoods = \
                                    step.select_end_points(mtrace)

            pdict, step.covariance = get_trace_stats(
                mtrace, step, burn, thin)

            if step.proposal_name == 'MultivariateNormal':
                step.proposal_dist = choose_proposal(
                    step.proposal_name, scale=step.covariance)

            if update is not None:
                logger.info('Updating Covariances ...')
                update.update_weights(pdict['dist_mean'], n_jobs=n_jobs)

                logger.info('Updating last samples ...')
                draws = 1
                step.stage = 0
                trans_stage_path = os.path.join(
                    homepath, 'trans_stage_%i' % s)
                logger.info('in %s' % trans_stage_path)

                chains = None

                sample_args = {
                    'draws': draws,
                    'step': step,
                    'stage_path': trans_stage_path,
                    'progressbar': progressbar,
                    'model': model,
                    'n_jobs': n_jobs,
                    'chains': chains}

                _iter_parallel_chains(**sample_args)

                mtrace = backend.load(trans_stage_path, model)

            elif update is not None and stage == 0:
                update.engine.close_cashed_stores()

            step.chain_previous_lpoint = step.get_chain_previous_lpoint(mtrace)

            outpath = os.path.join(stage_path, sample_p_outname)
            outparam_list = [step, update]
            utility.dump_objects(outpath, outparam_list)

        get_final_stage(homepath, n_stages, model=model)
        outpath = os.path.join(homepath, 'stage_final', sample_p_outname)
        utility.dump_objects(outpath, outparam_list)


def get_trace_stats(mtrace, step, burn=0.5, thin=2):
    """
    Get mean value of trace variables and return point.

    Parameters
    ----------
    mtrace : :class:`pymc3.backends.base.MultiTrace`
        Multitrace sampling result
    step : initialised :class:`atmcmc.ATMCMC` sampler object
    burn : float
        Burn-in parameter to throw out samples from the beginning of the trace
    thin : int
        Thinning of samples in the trace

    Returns
    -------
    dict with points, covariance matrix
    """

    n_steps = len(mtrace)

    array_population = num.zeros((step.n_jobs * int(
                                    num.ceil(n_steps * (1 - burn) / thin)),
                                    step.ordering.dimensions))

    # collect end points of each chain and put into array
    for var, slc, shp, _ in step.ordering.vmap:
        if len(shp) == 0:
            array_population[:, slc] = num.atleast_2d(
                                mtrace.get_values(varname=var,
                                            burn=int(burn * n_steps),
                                            thin=thin,
                                            combine=True)).T
        else:
            array_population[:, slc] = mtrace.get_values(
                                                varname=var,
                                                burn=int(burn * n_steps),
                                                thin=thin,
                                                combine=True)

    llks = mtrace.get_values(
        varname=step.likelihood_name,
        burn=int(burn * n_steps),
        thin=thin,
        combine=True)

    posterior_idxs = utility.get_fit_indexes(llks)
    d = {}
    for k, v in posterior_idxs.iteritems():
        d[k] = step.bij.rmap(array_population[v, :])

    d['dist_mean'] = step.bij.rmap(array_population.mean(axis=0))
    avar = array_population.var(axis=0)
    if avar.sum() == 0.:
        logger.warn('Trace std not valid not enough samples! Use 1.')
        avar = 1.
    cov = num.eye(step.ordering.dimensions) * avar
    return d, cov
