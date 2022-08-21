import logging
import unittest

from pyrocko import util

from beat.info import project_root
from beat.sampler.distributed import MPIRunner, run_mpi_sampler

logger = logging.getLogger("test_distributed")


class TestDistributed(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.n_jobs = 4
        self.beatpath = project_root

    def test_mpi_runner(self):

        logger.info("testing")
        runner = MPIRunner()
        runner.run(self.beatpath + "/test/pt_toy_example.py", n_jobs=self.n_jobs)
        logger.info("successful!")

    def test_arg_passing(self):
        nsamples = 100
        sampler_args = [nsamples]
        run_mpi_sampler("pt", sampler_args, keep_tmp=True, n_jobs=self.n_jobs)


if __name__ == "__main__":

    util.setup_logging("test_distributed", "info")
    unittest.main()
