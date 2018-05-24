import logging
import unittest
from beat.distributed import MPIRunner
from pyrocko import util


logger = logging.getLogger('test_distributed')


class TestSeisComposite(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.n_jobs = 4
        self.beatpath = '/home/vasyurhm/Software/beat'

    def test_mpi_runner(self):

        logger.info('testing')
        runner = MPIRunner()
        runner.run(self.beatpath + '/test/pt_toy_example.py', n_jobs=self.n_jobs)
        logger.info('successful!')

    def test_arg_passing(self):
        self.beatpath = '/home/vasyurhm/Software/beat'
        logger.info('testing')
        runner = MPIRunner()
        runner.run(beatpath + '/test/pt_toy_example.py', n_jobs=self.n_jobs)
        logger.info('successful!')

if __name__ == '__main__':

    util.setup_logging('test_distributed', 'info')
    unittest.main()
