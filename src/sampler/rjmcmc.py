from beat.sampler.metropolis import ArrayStepSharedLLK, Metropolis
from beat import transd
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
    def __init__(self):

        self.lordering = transd.TransDListArrayOrdering(
            dimensions, out_vars, intype='tensor')

        self.ordering = transd.TransDArrayOrdering(dimensions, vars)
        self.bij = transd.TransDDictToArrayBijection(
            self.ordering, self.population[0])
        self.lij = transd.TransDListToArrayBijection(
            self.lordering, lpoint, blacklist=blacklist)

    def step(self, point):

        # k has to be in point / kq0
        # k Bij maps? No! TrnasD

        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        n = num.random.rand()

        if n <= 1. / 6:
            self.bstep(kq0)

        elif n <= 1. / 3:
            self.dstep(kq0)

        elif n <= 1. / 2:
            self.mstep(kq0)

        else:
            apoint, alist = self.astep(self.bij.map(point))

            return self.bij.rmap(apoint), alist

        astep

        return k, lpoint


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