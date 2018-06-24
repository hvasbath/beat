import logging
import unittest
from beat.sources import MTSourceQT

from pyrocko import util
import numpy as num

logger = logging.getLogger('test_sources')


class TestSources(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_MTSourceQT(self):

        mt = MTSourceQT()
        print mt.m9


if __name__ == '__main__':

    util.setup_logging('test_sources', 'info')
    unittest.main()
