import logging
import unittest
from time import time

import numpy as num
from pyrocko import util

from beat.config import SourcesParameterMapping
from beat.utility import split_point

logger = logging.getLogger("test_config")


class TestConfig(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_parameter_source_mapping(self):
        mapping = SourcesParameterMapping(datatypes=["geodetic", "seismic"])

        sources_variables_one = {
            "east_shift": 1,
            "a_half_axis": 1,
        }

        sources_variables_two = {
            "east_shift": 2,
            "depth_bottom": 2,
        }

        mapping.add(
            datatype="geodetic",
            sources_variables=[sources_variables_one, sources_variables_two],
        )

        sources_variables_one["duration"] = 1

        mapping.add(
            datatype="seismic",
            sources_variables=[sources_variables_one, sources_variables_two],
        )

        vars_sizes = mapping.unique_variables_sizes()
        point = {varname: num.arange(size) for varname, size in vars_sizes.items()}

        t0 = time()
        spoint = split_point(point, mapping=mapping["geodetic"], n_sources_total=3)
        t1 = time()

        assert len(spoint) == 3
        assert "depth_bottom" not in spoint[0].keys()
        assert "depth_bottom" in spoint[1].keys()
        assert "depth_bottom" in spoint[2].keys()

        for point in spoint:
            assert "east_shift" in point.keys()
            assert "duration" not in point.keys()

        point = {varname: num.arange(size) for varname, size in vars_sizes.items()}
        spoint = split_point(point, mapping=mapping["seismic"], n_sources_total=3)

        assert "duration" in spoint[0].keys()
        print(spoint, t1 - t0)


if __name__ == "__main__":
    util.setup_logging("test_config", "info")
    unittest.main()
