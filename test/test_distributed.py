import os

import pytest

from beat.sampler.distributed import MPIRunner, run_mpi_sampler


@pytest.mark.mpi
def test_mpi_runner(test_dir):
    cwd = os.getcwd()
    runner = MPIRunner()
    runner.run(str(test_dir / "pt_toy_example.py"), n_jobs=4)
    # the runner works in a temporary directory that is deleted afterwards
    assert os.getcwd() == cwd


def test_run_mpi_sampler_unknown_sampler():
    with pytest.raises(NotImplementedError):
        run_mpi_sampler(
            "unknown", model=None, sampler_args=[], keep_tmp=False, n_jobs=1
        )
