"""
The example projects in data/examples (used in the tutorials) have to be
loadable with the current code, i.e. configs and pickled data must not be
broken by changes of the guts classes.
"""

import pytest
from conftest import EXAMPLES_DIR

from beat import config as bconfig
from beat import utility

EXAMPLES = sorted(
    (path.parent.name, path.stem.split("_", 1)[1])
    for path in EXAMPLES_DIR.glob("*/config_*.yaml")
)


def test_examples_found():
    assert len(EXAMPLES) >= 5


@pytest.mark.parametrize("name, mode", EXAMPLES, ids=["_".join(e) for e in EXAMPLES])
def test_example_config(name, mode):
    config = bconfig.load_config(str(EXAMPLES_DIR / name), mode)
    config.validate()

    pc = config.problem_config
    assert pc.mode == mode
    assert len(pc.priors) > 0
    assert set(pc.datatypes) <= set(bconfig._datatype_choices)
    for datatype in pc.datatypes:
        assert config[f"{datatype}_config"] is not None

    # test point within bounds
    for param in pc.priors.values():
        assert (param.lower <= param.testvalue).all()
        assert (param.testvalue <= param.upper).all()


@pytest.mark.parametrize(
    "path",
    sorted(EXAMPLES_DIR.glob("*/*_data.pkl")),
    ids=lambda p: f"{p.parent.name}_{p.stem}",
)
def test_example_data(path):
    data = utility.load_objects(str(path))
    assert len(data) > 0
    if path.name == bconfig.seismic_data_name:
        stations, traces = data
        assert len(stations) > 0
        assert len(traces) > 0
        assert {tr.station for tr in traces} <= {st.station for st in stations}
    else:
        for dataset in data:
            assert dataset.displacement.size == dataset.samples
