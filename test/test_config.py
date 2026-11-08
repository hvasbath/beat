import numpy as num
import pytest

from beat.config import SourcesParameterMapping
from beat.utility import split_point


@pytest.fixture
def mapping():
    # one source of type 1 and two sources of type 2
    mapping = SourcesParameterMapping(
        datatypes=["geodetic", "seismic"], n_sources=[1, 2]
    )

    sources_variables_one = {"east_shift": 1, "a_half_axis": 1}
    sources_variables_two = {"east_shift": 2, "depth_bottom": 2}

    mapping.add(
        datatype="geodetic",
        sources_variables=[sources_variables_one, sources_variables_two],
    )
    mapping.add(
        datatype="seismic",
        sources_variables=[
            dict(sources_variables_one, duration=1),
            sources_variables_two,
        ],
    )
    return mapping


def test_unique_variables_sizes(mapping):
    assert mapping.unique_variables_sizes() == {
        "east_shift": 3,
        "a_half_axis": 1,
        "depth_bottom": 2,
        "duration": 1,
    }


def test_point_to_sources_mapping(mapping):
    geodetic = mapping["geodetic"]
    assert geodetic.n_sources == 3
    assert geodetic.point_to_sources_mapping() == {
        "east_shift": [0, 1, 2],
        "a_half_axis": [0],
        "depth_bottom": [1, 2],
    }


@pytest.mark.parametrize("datatype", ["geodetic", "seismic"])
def test_split_point(mapping, datatype):
    vars_sizes = mapping.unique_variables_sizes()
    point = {varname: num.arange(size) for varname, size in vars_sizes.items()}

    spoint = split_point(point, mapping=mapping[datatype], weed_params=True)

    assert len(spoint) == 3
    assert "depth_bottom" not in spoint[0]
    assert spoint[1]["depth_bottom"] == 0.0
    assert spoint[2]["depth_bottom"] == 1.0
    assert [p["east_shift"] for p in spoint] == [0.0, 1.0, 2.0]

    if datatype == "seismic":
        assert "duration" in spoint[0]
    else:
        assert all("duration" not in p for p in spoint)


def test_split_point_without_mapping():
    point = {"east_shift": num.array([1.0, 2.0]), "depth": num.array([3.0, 4.0])}
    spoint = split_point(point, n_sources_total=2)
    assert spoint == [
        {"east_shift": 1.0, "depth": 3.0},
        {"east_shift": 2.0, "depth": 4.0},
    ]


def test_split_point_requires_exactly_one_of_mapping_or_n_sources(mapping):
    point = {"east_shift": num.arange(3)}
    with pytest.raises(ValueError):
        split_point(point)

    with pytest.raises(ValueError):
        split_point(point, mapping=mapping["geodetic"], n_sources_total=3)
