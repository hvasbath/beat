import logging
import unittest
from beat.sampler import choose_proposal, available_proposals
from beat.sampler.base import multivariate_proposals

from pyrocko import util
import numpy as num

logger = logging.getLogger('test_proposals')


class TestProposals(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.draws = 10

    def test_proposals(self):

        for proposal in available_proposals():
            if proposal in multivariate_proposals:
                scale = num.eye(2) * 0.5
            else:
                scale = 1

            draw = choose_proposal(proposal, scale=scale)
            print(proposal, draw(self.draws))


if __name__ == '__main__':

    util.setup_logging('test_proposals', 'info')
    unittest.main()
