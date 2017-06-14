import unittest
from beat import heart, models

import numpy as num
from numpy.testing import assert_allclose
from tempfile import mkdtemp
import os
import logging
import shutil

from pyrocko import util, trace
from pyrocko import plot, orthodrome


logger = logging.getLogger('test_heart')
km = 1000.


class RundirectoryError(Exception):
    pass


def get_run_directory():
    cwd = os.getcwd()
    if os.path.basename(cwd) != 'beat':
        raise RundirectoryError(
            'The test suite has to be run in the beat main-directory! '
            'Current work directory: %s' % cwd)
    else:
        return cwd


class TestHeart(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.beat_dir = get_run_directory()
        self.examples_dir = os.path.join(self.beat_dir, 'data/examples')

    def get_project_dir(self, name):
        return os.path.join(self.examples_dir, name)

    def _get_mt_source_params(self):
        source_point = {
            'magnitude': 4.8,
            'mnn': 0.84551376,
            'mee': -0.75868967,
            'mdd': -0.08682409,
            'mne': 0.51322155,
            'mnd': 0.14554675,
            'med': -0.25767963,
            'east_shift': 10.,
            'north_shift': 20.,
            'depth': 8.00,
            'time': -2.5,
            'duration': 5.,
            }
        return {k:num.atleast_1d(num.asarray(v)) for k, v in source_point.iteritems()}

    def test_full_mt(self):
        mode = 'geometry'

        project_dir = self.get_project_dir('FullMT')

        problem = models.load_model(project_dir, mode=mode)

        sc = problem.composites['seismic']

        point = self._get_mt_source_params()

        synths, obs = sc.get_synthetics(problem.model.test_point, outmode='data')

        for st, ot in zip(synths, obs):
            assert_allclose(st.ydata, ot.ydata, rtol=1e-05, atol=0)

if __name__ == "__main__":
    util.setup_logging('test_heart', 'info')
    unittest.main()
