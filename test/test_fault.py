"""
Finite fault geometry (beat.ffi.fault): discretization, slip, rupture times
and moments, with analytical expectations.
"""

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pyrocko import gf
from pyrocko.moment_tensor import magnitude_to_moment

from beat.config import UniformDiscretizationConfig
from beat.ffi import discretize_sources
from beat.sources import RectangularSource

km = 1000.0
LENGTH = 6.0  # [km]
WIDTH = 4.0
PATCH_SIZE = 1.0


@pytest.fixture
def reference_source():
    return RectangularSource(
        lat=10.0,
        lon=20.0,
        depth=2 * km,
        length=LENGTH * km,
        width=WIDTH * km,
        strike=30.0,
        dip=50.0,
        rake=90.0,
        slip=1.0,
        anchor="top",
        stf=gf.HalfSinusoidSTF(duration=1.0, anchor=-1.0),
    )


@pytest.fixture
def fault(reference_source):
    config = UniformDiscretizationConfig(
        patch_widths=[PATCH_SIZE],
        patch_lengths=[PATCH_SIZE],
        extension_widths=[0.0],
        extension_lengths=[0.0],
    )
    return discretize_sources(
        config,
        sources=[reference_source],
        datatypes=["geodetic", "seismic"],
        varnames=["uparr", "uperp"],
    )


def test_uniform_discretization(fault, reference_source):
    npw, npl = int(WIDTH / PATCH_SIZE), int(LENGTH / PATCH_SIZE)
    assert fault.nsubfaults == 1
    assert fault.npatches == npw * npl
    assert fault.ordering.get_subfault_discretization(0) == (npw, npl)
    assert fault.is_discretized
    assert not fault.needs_optimization

    patches = fault.get_subfault_patches(0, datatype="geodetic", component="uparr")
    assert len(patches) == fault.npatches
    area = sum(p.length * p.width for p in patches)
    assert_allclose(area, reference_source.length * reference_source.width)
    for patch in patches:
        assert patch.strike == reference_source.strike
        assert patch.dip == reference_source.dip
        assert_allclose([patch.length, patch.width], PATCH_SIZE * km)

    # patches tile the fault plane: top depths are in steps of the patch width
    depths = sorted({round(p.depth, 3) for p in patches})
    dz = PATCH_SIZE * km * num.sin(num.deg2rad(reference_source.dip))
    assert_allclose(num.diff(depths), dz, rtol=1e-6)
    assert_allclose(depths[0], reference_source.depth, atol=1e-3)


def test_patchmap_and_locations(fault):
    npw, npl = fault.ordering.get_subfault_discretization(0)
    indexes = num.array(
        [[fault.patchmap(0, i, j) for j in range(npl)] for i in range(npw)]
    )
    assert sorted(indexes.ravel()) == list(range(fault.npatches))

    dipidx, strikeidx = fault.fault_locations2idxs(
        0,
        positions_dip=num.array([0.5, 2.7, 3.99]),
        positions_strike=num.array([0.1, 3.5, 5.9]),
    )
    assert list(dipidx) == [0, 2, 3]
    assert list(strikeidx) == [0, 3, 5]


def test_total_slip_and_rake(fault, reference_source):
    rstate = num.random.RandomState(0)
    point = {
        "uparr": rstate.uniform(0, 2, fault.npatches),
        "uperp": rstate.uniform(-1, 1, fault.npatches),
    }
    assert_allclose(
        fault.get_total_slip(0, point), num.hypot(point["uparr"], point["uperp"])
    )

    patches = fault.point2sources(point)
    assert len(patches) == fault.npatches
    rakes = num.deg2rad([p.rake for p in patches]) - num.deg2rad(reference_source.rake)
    slips = num.array([p.slip for p in patches])
    # uparr is slip parallel to the reference rake, uperp is perpendicular
    assert_allclose(slips * num.cos(rakes), point["uparr"], atol=1e-10)
    assert_allclose(slips * num.sin(rakes), -point["uperp"], atol=1e-10)


def test_rupture_starttimes_uniform_velocity(fault):
    npw, npl = fault.ordering.get_subfault_discretization(0)
    velocity = 3.0
    nuc_dip_idx, nuc_strike_idx = 1, 2
    starttimes = fault.get_subfault_starttimes(
        0, num.full(fault.npatches, velocity), nuc_dip_idx, nuc_strike_idx
    )
    assert starttimes.shape == (npw, npl)
    assert starttimes[nuc_dip_idx, nuc_strike_idx] == 0.0

    dip_idxs, strike_idxs = num.mgrid[:npw, :npl]
    distances = (
        num.hypot(dip_idxs - nuc_dip_idx, strike_idxs - nuc_strike_idx) * PATCH_SIZE
    )
    exact = distances / velocity
    # first order eikonal solution: exact along the grid axes, otherwise up to
    # ~21 % slower than the straight ray
    on_axes = (dip_idxs == nuc_dip_idx) | (strike_idxs == nuc_strike_idx)
    assert_allclose(starttimes[on_axes], exact[on_axes], atol=1e-10)
    assert (starttimes >= exact - 1e-10).all()
    assert (starttimes <= 1.25 * exact + 1e-10).all()

    point = {
        "nucleation_dip": num.array([1.5]),
        "nucleation_strike": num.array([2.5]),
        "time": num.array([0.7]),
        "velocities": num.full(fault.npatches, velocity),
    }
    assert_allclose(fault.point2starttimes(point), starttimes + 0.7)


def test_moment_uniform_slip(fault, reference_source, geodetic_project):
    from synthetic import geodetic_store_id

    engine = gf.LocalEngine(store_superdirs=[geodetic_project.store_superdir])
    store = engine.get_store(geodetic_store_id())
    target = gf.StaticTarget(store_id=geodetic_store_id())

    slip = 1.5
    point = {"uparr": num.full(fault.npatches, slip)}
    moment = fault.get_moment(point=point, store=store, target=target)

    # homogeneous medium: M0 = mu * A * slip
    reference_source.slip = slip
    reference_moment = reference_source.get_moment(store=store, target=target)
    assert_allclose(moment, reference_moment, rtol=1e-6)

    magnitude = fault.get_magnitude(point=point, store=store, target=target)
    assert_allclose(magnitude_to_moment(magnitude), moment, rtol=1e-6)

    # in place slips
    patch_moments = fault.get_subfault_patch_moments(
        0, slips=None, store=store, target=target, datatype="geodetic"
    )
    assert_allclose(sum(patch_moments), moment, rtol=1e-6)


def test_smoothing_operator(fault):
    from pyrocko import model

    event = model.Event(lat=10.0, lon=20.0)
    operator = fault.get_smoothing_operator(event)
    assert operator.shape == (fault.npatches, fault.npatches)
    assert_allclose(operator, operator.T)
    # Laplacian: smooth slip has zero roughness
    assert_allclose(operator.dot(num.ones(fault.npatches)), 0.0, atol=1e-10)
    rough = num.zeros(fault.npatches)
    rough[fault.patchmap(0, 1, 2)] = 1.0
    assert num.abs(operator.dot(rough)).sum() > 0
