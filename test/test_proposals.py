import logging
import unittest
from beat.sampler import choose_proposal, available_proposals
from beat.sampler.base import multivariate_proposals
from beat.sampler import base

from pymc3 import kdeplot

from pyrocko import util
import numpy as num
from time import time
import matplotlib.pyplot as plt


logger = logging.getLogger('test_proposals')


class TestProposals(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.draws = 10
        self.plot = 1

        cov = num.array([[2., 0.5], [0.5, 1.]])
        self.normal = base.NormalProposal(1.)
        self.cauchy = base.CauchyProposal(1.)
        self.mvcauchy = base.MultivariateCauchyProposal(num.array([[1.]]))
        self.mvrotcauchy = base.MultivariateRotationCauchyProposal(cov)
        self.mvcauchycov = base.MultivariateCauchyProposal(cov)

    def test_proposals(self):

        for proposal in available_proposals():
            if proposal in multivariate_proposals:
                scale = num.eye(2) * 0.5
                print(proposal)
            else:
                scale = 1

            draw = choose_proposal(proposal, scale=scale)
            print((proposal, draw(self.draws)))

    def test_cauchy(self):

        nsamples = 100000
        discard = 1000

        ndist = self.normal(nsamples)

        cdist = self.cauchy(nsamples)
        cdist.sort(0)
        cdist = cdist[discard:-discard:1]

        mvcdist = self.mvcauchy(nsamples)
        mvcdist.sort(0)
        mvcdist = mvcdist[discard:-discard:1]

        if self.plot:
            ax = plt.axes()
            for d, color in zip(
                    [ndist, cdist, mvcdist], ['black', 'blue', 'red']):

                ax = kdeplot(d, ax=ax, color=color)

        ax.set_xlim([-10., 10.])
        plt.show()

    def test_cauchy(self):

        nsamples = 100000
        discard = 1000

        ndist = self.normal(nsamples)

        cdist = self.cauchy(nsamples)
        cdist.sort(0)
        cdist = cdist[discard:-discard:1]

        mvcdist = self.mvcauchy(nsamples)
        mvcdist.sort(0)
        mvcdist = mvcdist[discard:-discard:1]

        if self.plot:
            ax = plt.axes()
            for d, color in zip(
                    [ndist, cdist, mvcdist], ['black', 'blue', 'red']):

                ax = kdeplot(d, ax=ax, color=color)

        ax.set_xlim([-10., 10.])
        plt.show()

    def muhtest_rotation(self):

        t0 = time()
        num.random.seed(10)
        print(self.mvrotcauchy(2))
        t1 = time()

        num.random.seed(10)
        print(self.mvcauchycov(2))
        t2 = time()

        print(t2-t1, t1-t0)

if __name__ == '__main__':

    util.setup_logging('test_proposals', 'info')
    unittest.main()
