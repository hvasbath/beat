import logging
import time
import unittest

import numpy as num
from pyrocko import util

from beat import paripool

logger = logging.getLogger("test_paripool")


def add(x, y):
    logger.info("waiting for %i seconds" % x)
    time.sleep(x)
    return x + y


class ParipoolTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.factors = num.array([0, 1, 2, 3, 2, 1, 0])

    def test_pool(self):

        featureClass = [[k, 1] for k in self.factors]  # list of arguments
        p = paripool.paripool(add, featureClass, chunksize=2, nprocs=4, timeout=3)

        ref_values = (self.factors + 1).tolist()
        ref_values[3] = None
        for e in p:
            for val, rval in zip(e, ref_values):
                assert val == rval


if __name__ == "__main__":
    util.setup_logging("test_paripool", "debug")
    unittest.main()
