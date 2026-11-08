import numpy as num
import pymc as pm
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from beat.backend import (
    NumpyChain,
    TextChain,
    check_multitrace,
    load_multitrace,
    multitrace_to_inference_data,
)

VARIABLE_SHAPES = {"A": 5, "B": 3, "C": 1}
N_DRAWS = 7


@pytest.fixture
def model():
    with pm.Model() as model:
        for name, size in VARIABLE_SHAPES.items():
            pm.Uniform(
                name,
                shape=size,
                lower=-10.0 * num.ones(size),
                upper=10.0 * num.ones(size),
                initval=num.zeros(size),
                transform=None,
            )
        pm.Deterministic("like", model["A"].sum() + model["B"].sum())

    return model


def make_lpoint(varnames, draw):
    """Distinct values for every variable and draw."""
    sizes = dict(VARIABLE_SHAPES, like=1)
    return [
        num.arange(sizes[name], dtype="float64") + 10.0 * i + 100.0 * draw
        for i, name in enumerate(varnames)
    ]


def fill_chain(chain_class, model, dir_path, chain_idx=0):
    chain = chain_class(dir_path=str(dir_path), model=model)
    chain.setup(N_DRAWS, chain_idx, overwrite=True)

    lpoints = []
    for draw in range(N_DRAWS):
        lpoint = make_lpoint(chain.varnames, draw)
        lpoints.append(lpoint)
        chain.write(lpoint, draw)

    chain.record_buffer()
    return chain, lpoints


@pytest.mark.parametrize("chain_class", [TextChain, NumpyChain])
def test_chain_roundtrip(chain_class, model, tmp_path):
    chain, lpoints = fill_chain(chain_class, model, tmp_path)

    assert set(chain.varnames) == set(VARIABLE_SHAPES) | {"like"}
    assert len(chain) == N_DRAWS

    for i, varname in enumerate(chain.varnames):
        expected = num.vstack([lpoint[i] for lpoint in lpoints])
        values = chain.get_values(varname)
        assert_allclose(num.atleast_2d(values.T).T.reshape(expected.shape), expected)

    for draw in [0, 3, N_DRAWS - 1]:
        point = chain.point(draw)
        for i, varname in enumerate(chain.varnames):
            assert_allclose(point[varname].ravel(), lpoints[draw][i])


@pytest.mark.parametrize("chain_class", [TextChain, NumpyChain])
def test_chain_reload_from_file(chain_class, model, tmp_path):
    _, lpoints = fill_chain(chain_class, model, tmp_path)

    # reopen existing chain file without overwriting
    chain = chain_class(dir_path=str(tmp_path), model=model)
    chain.setup(N_DRAWS, 0, overwrite=False)

    assert len(chain) == N_DRAWS
    point = chain.point(2)
    for i, varname in enumerate(chain.varnames):
        assert_allclose(point[varname].ravel(), lpoints[2][i])


@pytest.mark.parametrize(
    "chain_class, backend", [(TextChain, "csv"), (NumpyChain, "bin")]
)
def test_load_check_multitrace(chain_class, backend, model, tmp_path):
    n_chains = 3
    for chain_idx in range(n_chains):
        fill_chain(chain_class, model, tmp_path, chain_idx=chain_idx)

    mtrace = load_multitrace(str(tmp_path), varnames=model.value_vars, backend=backend)

    assert sorted(mtrace.chains) == list(range(n_chains))
    assert set(mtrace.varnames) == set(VARIABLE_SHAPES) | {"like"}
    assert check_multitrace(mtrace, N_DRAWS, n_chains) == []

    values = mtrace.get_values("A", combine=True, squeeze=True)
    assert values.shape == (n_chains * N_DRAWS, VARIABLE_SHAPES["A"])

    idata = multitrace_to_inference_data(mtrace)
    assert set(VARIABLE_SHAPES).issubset(idata.posterior.data_vars)


def test_check_multitrace_detects_incomplete(model, tmp_path):
    for chain_idx in range(2):
        fill_chain(TextChain, model, tmp_path, chain_idx=chain_idx)

    mtrace = load_multitrace(str(tmp_path), varnames=model.value_vars, backend="csv")
    corrupted = check_multitrace(mtrace, N_DRAWS + 5, 2)
    assert_array_equal(sorted(corrupted), [0, 1])
