import numpy as num
from beat import utility
from tempfile import mkdtemp
import shutil
import unittest
from pyrocko import util
from theano import shared
import theano.tensor as tt
from pymc3 import DictToArrayBijection, ArrayOrdering

RAD = num.pi / 180.


class TestUtility(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_rotation(self):

        self.R = utility.get_rotation_matrix(['x', 'y'])
        self.Rz = utility.get_rotation_matrix('z')

        A = num.array([1, 0, 0])
        B = num.array([0, 1, 0])
        C = num.array([0, 0, 1])

        num.testing.assert_allclose(self.Rz(90. * RAD).dot(A), B, rtol=0., atol=1e-6)
        num.testing.assert_allclose(
            self.R['x'](90. * RAD).dot(B), C, rtol=0., atol=1e-6)
        num.testing.assert_allclose(
            self.R['y'](90. * RAD).dot(C), A, rtol=0., atol=1e-6)

    def test_list_ordering(self):
        a = num.random.rand(10).reshape((5, 2))
        b = num.random.rand(5).reshape((5, 1))
        c = num.random.rand(1).reshape((1, 1))
        ta = tt.matrix('a')
        tb = tt.matrix('b')
        tc = tt.matrix('c')
        ta.tag.test_value = a
        tb.tag.test_value = b
        tc.tag.test_value = c
        tvars = [ta, tb, tc]
        with self.assertRaises(KeyError):
            ordering = utility.ListArrayOrdering(tvars)
            ordering['b']

        lordering = utility.ListArrayOrdering(tvars, 'tensor')
        lordering['b'].slc

        for var in ordering:
            print var

        lpoint = [a, b, c]
        lij = utility.ListToArrayBijection(lordering, lpoint)

        ref_point = {'a': a, 'b':b, 'c':c}
        array = lij.l2a(lpoint)
        point = lij.l2d(lpoint)
        print 'arr', array
        #print 'point, ref_point', point, ref_point

        print lij.l2d(lij.a2l(array))

    def test_stencil(self):
        for order in [3, 5]:
            so = utility.StencilOperator(order=order, h=0.001)
            print so
            print len(so)
            print so.hsteps
            print so.coefficients
            print so.denominator


if __name__ == '__main__':
    util.setup_logging('test_utility', 'warning')
    unittest.main()
