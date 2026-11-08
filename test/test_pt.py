import os

import numpy as num
import pytest
from toy_models import MU1, two_gaussians_model

from beat.backend import SampleStage, multitrace_to_inference_data
from beat.config import sample_p_outname
from beat.sampler import metropolis, pt
from beat.utility import load_objects

N_CHAINS = 4
N_SAMPLES = 3000
BURN = 0.5


@pytest.mark.mpi
def test_pt_two_gaussians(tmp_path):
    model = two_gaussians_model(potential=True)

    with model:
        step = metropolis.Metropolis(
            n_chains=N_CHAINS,
            likelihood_name=model.deterministics[0].name,
            proposal_name="MultivariateCauchy",
            tune_interval=50,
        )

    pt.pt_sample(
        step,
        n_chains=N_CHAINS,
        n_samples=N_SAMPLES,
        swap_interval=(10, 15),
        beta_tune_interval=300,
        n_workers_posterior=1,
        homepath=str(tmp_path),
        progressbar=False,
        buffer_size=int(N_SAMPLES / 10),
        model=model,
        rm_flag=False,
        keep_tmp=False,
    )

    stage_handler = SampleStage(str(tmp_path))
    mtrace = stage_handler.load_multitrace(-1, varnames=model.value_vars)
    idata = multitrace_to_inference_data(mtrace)
    assert "X" in idata.posterior

    d = mtrace.get_values("X", combine=True, squeeze=True)
    assert d.shape[1] == MU1.size
    posterior = d[int(BURN * d.shape[0]) :]
    num.testing.assert_allclose(
        num.abs(posterior).mean(axis=0), MU1, rtol=0.0, atol=0.1
    )

    history = load_objects(os.path.join(stage_handler.stage_path(-1), sample_p_outname))
    assert len(history) > 0
    acceptance = num.asarray(history.acceptance)
    assert ((acceptance >= 0.0) & (acceptance <= 1.0)).all()
    assert len(history.t_scales) > 0
