from beat.sampler.metropolis import ArrayStepSharedLLK, Metropolis
from beat import transd
from beat.config import k_name
from logging import getLogger
import numpy as num


logger = getLogger('rjmcmc')


class TransdArrayStepSharedLLK(ArrayStepSharedLLK):
    """
    Class to overwrite the step method of 
    :class:`beat.sampler.metropolis.ArrayStepSharedLLK` with the trans-d
    stepping that also has birth and death steps.

    Notes
    -----
    Due to the Method Resolution Order described here:
    https://www.artima.com/weblogs/viewpost.jsp?thread=236275
    the parent in multiple inheritances that defines the method or attribute
    first is the one that is used. 
    """
    def __init__(self, dimensions, vars, out_vars):

        self.lordering = transd.TransDListArrayOrdering(
            dimensions, out_vars, intype='tensor')

        self.ordering = transd.TransDArrayOrdering(dimensions, vars)
        self.bij = transd.TransDDictToArrayBijection(
            self.ordering, self.population[0])
        self.lij = transd.TransDListToArrayBijection(
            self.lordering, lpoint, blacklist=blacklist)

        self.steps = {
            0: self.bstep,
            1: self.dstep,
            2: self.mstep,
            3: self.astep}

        self.nchoice_steps = len(self.steps.keys())
        self.kmin = self.bij.ordering.ks().min()
        self.kmax = self.bij.ordering.ks().max()

    def step(self, kpoint):

        for var, share in self.shared.items():
            share.container.storage[0] = kpoint[var]

        k = kpoint[k_name]

        if k == self.kmin:
            weights = [1. / 4, 0., 1. / 4, 1. / 2]
        elif k == self.kmax:
            weights = [0., 1. / 4, 1. / 4, 1. / 2]
        else:
            weights = [1. / 6, 1. / 6, 1. / 6, 1. / 2]

        n = num.random.choice(self.nchoice_steps, p=num.array(weights))

        apoint, alist = self.steps[n](self.bij.map(kpoint))
        return self.bij.rmap(apoint), alist


class RJMCMC(TransdArrayStepSharedLLK):

    default_blocked = True

    def __init__(self, vars=None, out_vars=None, covariance=None, scale=1.,
                 n_chains=100, tune=True, tune_interval=100, model=None,
                 check_bound=True, likelihood_name='like',
                 proposal_name='MultivariateNormal',
                 coef_variation=1., **kwargs):


        super(Metropolis, self).__init__(vars, out_vars, shared)

        delattr(self, array_population)

        # k dependent proposals and sample covariance updates necessary
        # make calc_cov already method at metropolis? resolves PT easier?
        # Bijs? Lijs? how juggeling points around in PT?

    def astep(self, kq0):
        """
        Perturbation step
        """
        delta = self.proposal_dists[k]()[self.transform_idxs]
        q = q0 + delta
        varlogp = self.check_bnd(q)
        lp = self.logp_forw(q)
        

    def mstep(self, kq0):
        """
        Move step
        """
        pass

    def bstep(self, kq0):
        """
        Birth step
        """
        pass

    def dstep(self, kq0):
        """
        Death step
        """
        pass