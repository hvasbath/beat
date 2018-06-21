import numpy as num
from beat import utility
from tempfile import mkdtemp
import shutil
import unittest
from pyrocko import util
from theano import shared
import theano.tensor as tt


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

    def test_list_ordering(self):
        a = num.random.rand(100).reshape((5, 20))
        b = num.random.rand(100).reshape((5, 20))
        ta = tt.matrix('a')
        tb = tt.matrix('b')
        ta.tag.test_value = a
        tb.tag.test_value = b
        tvars = [ta, tb]
        with self.assertRaises(KeyError):
            ordering = utility.ListArrayOrdering(tvars)
            ordering['b']

        ordering = utility.ListArrayOrdering(tvars, 'tensor')
        ordering['b'].slc


if __name__ == '__main__':
    util.setup_logging('test_utility', 'warning')
    unittest.main()
