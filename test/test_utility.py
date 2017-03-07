import numpy as num
from beat import utility
from tempfile import mkdtemp
import shutil
import unittest
from pyrocko import util


class TestUtility(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_rotation(self):

        self.Rx, self.Ry = utility.get_rotation_matrix(['x', 'y'])
        self.Rz = utility.get_rotation_matrix('z')

        A = num.array([1, 0, 0])
        B = num.array([0, 1, 0])
        C = num.array([0, 0, 1])

        num.testing.assert_allclose(self.Rz(90).dot(A), B, rtol=0., atol=1e-6)
        num.testing.assert_allclose(self.Rx(90).dot(B), C, rtol=0., atol=1e-6)
        num.testing.assert_allclose(self.Ry(90).dot(C), A, rtol=0., atol=1e-6)

if __name__ == '__main__':
    util.setup_logging('test_utility', 'info')
    unittest.main()
