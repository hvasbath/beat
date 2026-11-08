import multiprocessing as mp

import numpy as num
import pytest
from toy_models import MU1, two_gaussians_model

from beat import backend, utility
from beat.sampler import smc

N_CHAINS = 100
N_STEPS = 100
TUNE_INTERVAL = 25


@pytest.mark.parametrize(
    "n_jobs",
    [1, utility.biggest_common_divisor(N_CHAINS, mp.cpu_count())],
    ids=["one_core", "multicore"],
)
def test_smc_two_gaussians(n_jobs, tmp_path):
    model = two_gaussians_model()

    with model:
        step = smc.SMC(
            n_chains=N_CHAINS,
            tune_interval=TUNE_INTERVAL,
            likelihood_name=model.deterministics[0].name,
        )

    smc.smc_sample(
        n_steps=N_STEPS,
        step=step,
        n_jobs=n_jobs,
        progressbar=False,
        stage=0,
        homepath=str(tmp_path),
        model=model,
        rm_flag=False,
    )

    stage_handler = backend.SampleStage(str(tmp_path))
    mtrace = stage_handler.load_multitrace(-1, varnames=model.value_vars)

    d = mtrace.get_values("X", combine=True, squeeze=True)
    last_samples = d[(N_STEPS - 1) :: N_STEPS]
    assert last_samples.shape == (N_CHAINS, MU1.size)

    # both modes are at +-0.5
    num.testing.assert_allclose(
        num.abs(last_samples).mean(axis=0), MU1, rtol=0.0, atol=0.03
    )
