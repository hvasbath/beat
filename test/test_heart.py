import unittest
from beat import heart

from tempfile import mkdtemp
import os
import shutil
import unittest

from pyrocko import util
from pyrocko import plot, orthodrome


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
        self.work_dir = get_run_directory()

