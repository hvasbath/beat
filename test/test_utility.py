import logging
import unittest
from tempfile import mkdtemp
from time import time

import numpy as num
import theano.tensor as tt
from pyrocko import util

from beat import utility

RAD = num.pi / 180.0


logger = logging.getLogger("test_utility")


class TestUtility(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_rotation(self):

        self.R = utility.get_rotation_matrix(["x", "y"])
        self.Rz = utility.get_rotation_matrix("z")

        A = num.array([1, 0, 0])
        B = num.array([0, 1, 0])
        C = num.array([0, 0, 1])

        num.testing.assert_allclose(self.Rz(90.0 * RAD).dot(A), B, rtol=0.0, atol=1e-6)
        num.testing.assert_allclose(
            self.R["x"](90.0 * RAD).dot(B), C, rtol=0.0, atol=1e-6
        )
        num.testing.assert_allclose(
            self.R["y"](90.0 * RAD).dot(C), A, rtol=0.0, atol=1e-6
        )

    def test_list_ordering(self):
        a = num.random.rand(10).reshape((5, 2))
        b = num.random.rand(5).reshape((5, 1))
        c = num.random.rand(1).reshape((1, 1))
        ta = tt.matrix("a")
        tb = tt.matrix("b")
        tc = tt.matrix("c")
        ta.tag.test_value = a
        tb.tag.test_value = b
        tc.tag.test_value = c
        tvars = [ta, tb, tc]
        with self.assertRaises(KeyError):
            ordering = utility.ListArrayOrdering(tvars)
            ordering["b"]

        lordering = utility.ListArrayOrdering(tvars, "tensor")
        lordering["b"].slc

        for var in ordering:
            print(var)

        lpoint = [a, b, c]
        lij = utility.ListToArrayBijection(lordering, lpoint)

        ref_point = {"a": a, "b": b, "c": c}
        array = lij.l2a(lpoint)
        point = lij.l2d(lpoint)
        print("arr", array)
        print("point, ref_point", point, ref_point)
        print(lij.l2d(lij.a2l(array)))

    def test_window_rms(self):

        data = num.random.randn(5000)
        ws = int(data.size / 5)
        t0 = time()
        data_stds = utility.running_window_rms(data, window_size=ws)
        t1 = time()
        data_stds2 = num.array(
            [
                num.sqrt((data[i : i + ws] ** 2).sum() / ws)
                for i in range(data.size - ws + 1)
            ]
        )
        t2 = time()
        print("Convolution %f [s], loop %f [s]" % (t1 - t0, t2 - t1))
        num.testing.assert_allclose(data_stds, data_stds2, rtol=0.0, atol=1e-6)

        data_stds = utility.running_window_rms(data, window_size=ws, mode="same")
        print(data_stds.shape)

    def test_stencil(self):
        for order in [3, 5]:
            so = utility.StencilOperator(order=order, h=0.001)
            print(so)
            print(len(so))
            print("hsteps", so.hsteps)
            print("coeffs", so.coefficients)
            print("denom", so.denominator)


if __name__ == "__main__":
    util.setup_logging("test_utility", "warning")
    unittest.main()
