"""
Metropolis algorithm module, wrapping the pymc3 implementation.
Provides the possibility to update the involved covariance matrixes within
the course of sampling the chain.
"""

import os
import pymc3 as pm
import logging
from beat import backend

__all__ = ['sample']

logger = logging.getLogger('ATMCMC')


def sample(vars, n_stages=10, n_steps=10000, trace=None,
            step=None, model=None, n_jobs=1, update=None, burn=0.5, thin=2):
    """
    Execute Metropolis algorithm repeatedly depending on the number of stages.
    Execute covariances if given.
    """

    for stage in range(n_stages):
        stage_path = os.path.join(trace, 'stage_%i' % stage)

        pm.sample(
            draws=n_steps,
            step=step,
            trace=pm.backends.Text(
                name=stage_path,
                model=model,
                vars=vars),
            model=model,
            n_jobs=n_jobs)

        mtrace = backend.load(stage_path, model=model)

        if update is not None:
            logger.info('Updating Covariances ...')
            mean_pt = get_mean_point(mtrace, n_steps, burn, thin)
            update.update_weights(mean_pt, n_jobs=n_jobs)
            print update.sweights[0].get_value()


def get_mean_point(mtrace, n_steps=10000, burn=0.5, thin=2):
    """
    Get mean value of trace variables and return point.

    Parameters
    ----------
    mtrace : :class:`pymc3.backends.base.MultiTrace`
        Multitrace sampling result
    n_steps : int
        Number of steps in the Metropolis chain

    Returns
    -------
    dict
    """

    point = {}
    for var in mtrace.varnames:
        point[var] = mtrace.get_values(var, combine=True, squeeze=True,
            burn=int(n_steps * burn), thin=thin).mean()

    return point
