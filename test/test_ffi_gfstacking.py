from os.path import join as pjoin

import numpy as num
import pytest
from conftest import require_external
from pyrocko import gf, trace
from pytensor import config as tconfig

from beat import ffi, heart, models

km = 1000.0
nuc_dip = 5.0
nuc_strike = 2.0
time_shift = -10.0  # from previous inversion

PROJECT = ("BEATS", "LaquilaJointPonlyUPDATE_wide_kin3_v2")
GF_STORES = ("GF", "Laquila")


def array_to_traces(synthetics, reference_times, deltat, targets, location_tag=None):
    synth_traces = []
    for i, target in enumerate(targets):
        tr = trace.Trace(ydata=synthetics[i, :], tmin=reference_times[i], deltat=deltat)

        tr.set_codes(*target.codes)
        if location_tag is not None:
            tr.set_location(location_tag)

        synth_traces.append(tr)

    return synth_traces


def get_max_relative_and_absolute_errors(a, b):
    abs_err = num.abs(a - b).max()
    rel_err = (num.abs((a - b) / b).max(),)
    return abs_err, rel_err


def assert_traces(ref_traces, test_traces):
    assert len(ref_traces) == len(test_traces)

    for ref_trace, test_trace in zip(ref_traces, test_traces):
        num.testing.assert_allclose(
            ref_trace.ydata, test_trace.ydata, rtol=5e-6, atol=5e-6
        )
        num.testing.assert_allclose(
            ref_trace.tmin, test_trace.tmin, rtol=1e-3, atol=1e-3
        )


@pytest.fixture
def no_test_values():
    with tconfig.change_flags(compute_test_value="off"):
        yield


@pytest.mark.slow
def test_gf_stacking(no_test_values):
    project_dir = str(require_external(*PROJECT))
    store_superdirs = [str(require_external(*GF_STORES))]

    problem = models.load_model(project_dir, mode="ffi", build=False)
    event = problem.config.event

    components = ["uparr"]  # , 'uperp']

    starttime_sampling = 0.5  # noqa: F841

    arrival_taper = heart.ArrivalTaper(a=-15.0, b=-10.0, c=50.0, d=55.0)

    sc = problem.composites["seismic"]
    fault = sc.load_fault_geometry()

    # get number of patches in dip and strike direction
    npdip, npstrike = fault.ordering.get_subfault_discretization(0)

    # do fast sweeping to get rupture onset times for patches with respect to hypocenter
    velocities = num.ones((npdip, npstrike)) * 3.5

    nuc_dip_idx, nuc_strike_idx = fault.fault_locations2idxs(
        0, nuc_dip, nuc_strike, backend="numpy"
    )

    starttimes = (
        fault.get_subfault_starttimes(
            0, velocities, nuc_dip_idx, nuc_strike_idx
        ).ravel()
        + time_shift
    )

    # defining distributed slip values for slip parallel and perpendicular directions
    uparr = num.ones((npdip, npstrike)) * 2.0
    uperp = num.zeros((npdip, npstrike))
    uperp[1:3, 3:7] = 1.0

    # define rupture durations on each patch
    durations = num.ones((npdip, npstrike)) * 0.5

    slips = {
        components[0]: uparr.ravel(),
        #    components[1]: uperp.ravel(),
        "durations": durations.ravel(),
        "velocities": velocities.ravel(),
    }

    # update patches with distributed slip and STF values
    for comp in components:
        patches = fault.get_subfault_patches(0, datatype="seismic", component=comp)

        for patch, starttime, duration, slip in zip(
            patches, starttimes, durations.ravel(), slips[comp]
        ):
            # stf = gf.HalfSinusoidSTF(anchor=-1., duration=float(duration))
            patch.stf.duration = float(duration)
            # stime = num.round(starttime / starttime_sampling) * starttime_sampling
            patch.update(slip=float(slip), time=event.time + float(starttime))
            # print(patch)

    # synthetics generation
    engine = gf.LocalEngine(store_superdirs=store_superdirs)
    targets = sc.wavemaps[0].targets
    filterer = sc.wavemaps[0].config.filterer
    ntargets = len(targets)

    gfs = ffi.load_gf_library(
        directory=pjoin(project_dir, "ffi/linear_gfs/"),
        filename="seismic_uparr_any_P_0",
    )
    ats = gfs.reference_times - arrival_taper.b

    # seismosizer engine --> reference
    ref_traces, _ = heart.seis_synthetics(
        engine,
        patches,
        targets,
        arrival_times=ats,
        wavename="any_P",
        arrival_taper=arrival_taper,
        filterer=filterer,
        outmode="stacked_traces",
    )

    targetidxs = num.atleast_2d(num.arange(ntargets)).T
    starttimes = num.tile(starttimes, ntargets).reshape((ntargets, uparr.size))

    durations_dim2 = num.atleast_2d(durations.ravel())
    patchidxs = num.arange(uparr.size, dtype="int")

    # numpy stacking
    gfs.set_stack_mode("numpy")
    synthetics_nn = gfs.stack_all(
        patchidxs=patchidxs,
        targetidxs=targetidxs,
        starttimes=starttimes[:, patchidxs],
        durations=durations_dim2,
        slips=slips[components[0]],
        interpolation="nearest_neighbor",
    )

    synthetics_ml = gfs.stack_all(
        patchidxs=patchidxs,
        targetidxs=targetidxs,
        starttimes=starttimes[:, patchidxs],
        durations=durations_dim2,
        slips=slips[components[0]],
        interpolation="multilinear",
    )

    # Pytensor stacking
    gfs.init_optimization()

    synthetics_nn_t = gfs.stack_all(
        targetidxs=targetidxs,
        starttimes=starttimes,
        durations=durations_dim2,
        slips=slips[components[0]],
        interpolation="nearest_neighbor",
    ).eval()

    synthetics_ml_t = gfs.stack_all(
        targetidxs=targetidxs,
        starttimes=starttimes,
        durations=durations_dim2,
        slips=slips[components[0]],
        interpolation="multilinear",
    ).eval()

    for test_synthetics, location_tag in zip(
        [synthetics_nn, synthetics_ml, synthetics_nn_t, synthetics_ml_t],
        ["nn", "ml", "nn_t", "ml_t"],
    ):
        test_traces = array_to_traces(
            test_synthetics,
            reference_times=gfs.reference_times,
            deltat=gfs.deltat,
            targets=targets,
            location_tag=location_tag,
        )

        assert_traces(ref_traces, test_traces)
