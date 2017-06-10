import unittest
from beat import heart

from tempfile import mkdtemp
import os
import logging
import shutil

from pyrocko import util
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

if __name__ == "__main__":
    util.setup_logging('test_heart', 'debug')
    unittest.main()
