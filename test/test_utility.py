import numpy as num
import pytensor.tensor as tt
import pytest
from numpy.testing import assert_allclose

from beat import utility

RAD = num.pi / 180.0


def test_rotation():
    R = utility.get_rotation_matrix(["x", "y"])
    Rz = utility.get_rotation_matrix("z")

    A = num.array([1, 0, 0])
    B = num.array([0, 1, 0])
    C = num.array([0, 0, 1])

    assert_allclose(Rz(90.0 * RAD).dot(A), B, rtol=0.0, atol=1e-6)
    assert_allclose(R["x"](90.0 * RAD).dot(B), C, rtol=0.0, atol=1e-6)
    assert_allclose(R["y"](90.0 * RAD).dot(C), A, rtol=0.0, atol=1e-6)


@pytest.fixture
def tensor_variables():
    a = num.random.rand(10).reshape((5, 2))
    b = num.random.rand(5).reshape((5, 1))
    c = num.random.rand(1).reshape((1, 1))
    tvars = []
    for name, value in zip("abc", [a, b, c]):
        tvar = tt.matrix(name)
        tvar.tag.test_value = value
        tvars.append(tvar)

    return tvars, [a, b, c]


def test_list_ordering_numpy_has_no_names(tensor_variables):
    tvars, _ = tensor_variables
    ordering = utility.ListArrayOrdering(tvars)
    with pytest.raises(KeyError):
        ordering["b"]


def test_list_ordering_tensor(tensor_variables):
    tvars, (a, b, c) = tensor_variables
    lordering = utility.ListArrayOrdering(tvars, "tensor")

    assert lordering.size == a.size + b.size + c.size
    assert lordering["b"].slc == slice(a.size, a.size + b.size)
    assert list(lordering) == ["a", "b", "c"]


def test_list_to_array_bijection_roundtrip(tensor_variables):
    tvars, lpoint = tensor_variables
    lordering = utility.ListArrayOrdering(tvars, "tensor")
    lij = utility.ListToArrayBijection(lordering, lpoint)

    array = lij.l2a(lpoint)
    assert array.shape == (lordering.size,)

    point = lij.l2d(lpoint)
    assert set(point.keys()) == {"a", "b", "c"}
    for name, value in zip("abc", lpoint):
        # values are returned flattened
        assert_allclose(point[name], value.ravel())

    point_from_array = lij.l2d(lij.a2l(array))
    for k, val in point_from_array.items():
        assert_allclose(val, point[k])


def test_window_rms():
    data = num.random.randn(5000)
    ws = int(data.size / 5)
    data_stds = utility.running_window_rms(data, window_size=ws)
    data_stds2 = num.array(
        [
            num.sqrt((data[i : i + ws] ** 2).sum() / ws)
            for i in range(data.size - ws + 1)
        ]
    )
    assert_allclose(data_stds, data_stds2, rtol=0.0, atol=1e-6)

    data_stds_same = utility.running_window_rms(data, window_size=ws, mode="same")
    assert data_stds_same.shape == data.shape


def _apply_stencil(stencil, func, x):
    values = num.array([func(x + hstep) for hstep in stencil.hsteps])
    return (values.reshape(stencil.coefficients.shape) * stencil.coefficients).sum(
        axis=0
    ).ravel() / stencil.denominator


@pytest.mark.parametrize("order", [3, 5])
def test_stencil_operator_setup(order):
    so = utility.StencilOperator(order=order, h=0.001)
    assert len(so) == order - 1
    assert so.coefficients.shape == (order - 1, 1, 1)
    assert_allclose(so.hsteps.sum(), 0.0, atol=1e-15)
    assert_allclose(so.coefficients.sum(), 0.0, atol=1e-15)


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(
            3,
            marks=pytest.mark.xfail(
                strict=True, reason="order 3 coefficients give -f'(x) (sign error)"
            ),
        ),
        pytest.param(
            5,
            marks=pytest.mark.xfail(
                strict=True,
                reason="order 5 coefficients [1, 8, -8, -1] give -5/3 f'(x), "
                "five-point stencil needs [1, -8, 8, -1]",
            ),
        ),
    ],
)
def test_stencil_first_derivative(order):
    """StencilOperator is used for heart.seis_derivatives."""
    so = utility.StencilOperator(order=order, h=0.001)
    x = num.array([0.3])
    assert_allclose(_apply_stencil(so, num.sin, x), num.cos(x), rtol=0.0, atol=1e-5)


def test_stencil_invalid_order():
    with pytest.raises(ValueError):
        utility.StencilOperator(order=4)
