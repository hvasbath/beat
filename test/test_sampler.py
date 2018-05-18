import unittest
import logging

from pymc3.plots import kdeplot
import numpy as num

from beat.sampler import base
from pyrocko import util

import matplotlib.pyplot as plt


km = 1000.

logger = logging.getLogger('test_sampler')


class SamplerTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        self.plot = 1

        unittest.TestCase.__init__(self, *args, **kwargs)
        self.normal = base.NormalProposal(1.)
        self.cauchy = base.CauchyProposal(1.)
        self.mvcauchy = base.MultivariateCauchyProposal(num.array([[1.]]))

    def test_proposals(self):

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


if __name__ == '__main__':
    util.setup_logging('test_sampler', 'info')
    unittest.main()
