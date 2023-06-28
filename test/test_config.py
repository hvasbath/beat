import logging
import unittest

import numpy as num

from beat.config import SourcesParameterMapping


logger = logging.getLogger("test_config")


class TestConfig(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_parameter_source_mapping(self):

        mapping = SourcesParameterMapping(datatypes=["geodetic", "seismic"])
        print(mapping)

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

        print(mapping)
        print(mapping.get_unique_variables_sizes())
        print(mapping.mappings["geodetic"].point_to_source_mapping())


if __name__ == "__main__":
    util.setup_logging("test_config", "info")
    unittest.main()
