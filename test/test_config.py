import logging
import unittest
from time import time

import numpy as num

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
            "major_axis": 1,
        }

        sources_variables_two = {
            "east_shift": 2,
            "delta_depth_bottom": 2,
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
        point_to_sources = mapping["geodetic"].point_to_sources_mapping()

        t0 = time()
        spoint = split_point(
            point, point_to_sources=point_to_sources, n_sources_total=3
        )
        t1 = time()
        print(spoint, t1 - t0)


if __name__ == "__main__":
    util.setup_logging("test_config", "info")
    unittest.main()
