"""
Stacking of the linear seismic GF library of a finite fault (ffi) project,
calculated by "beat build_gfs --mode ffi", compared to synthetics calculated
directly with the pyrocko engine for the same kinematic rupture.
"""

from os.path import join as pjoin

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pyrocko import gf
from pytensor import config as tconfig

from beat import ffi, heart

# nucleation point [km] along dip and strike of the 3 km x 5 km fault
NUC_DIP = 1.5
NUC_STRIKE = 2.5
TIME_SHIFT = 0.5  # nucleation time [s]
DURATION = 1.0  # [s], on the duration grid of the library
VELOCITY = 3.0  # [km/s]


@pytest.fixture(scope="module")
def rupture(ffi_project):
    problem = ffi_project.load_problem(build=False)
    event = problem.config.event
    sc = problem.composites["seismic"]
    fault = sc.load_fault_geometry()

    npdip, npstrike = fault.ordering.get_subfault_discretization(0)
    velocities = num.full((npdip, npstrike), VELOCITY)
    nuc_dip_idx, nuc_strike_idx = fault.fault_locations2idxs(
        0, NUC_DIP, NUC_STRIKE, backend="numpy"
    )
    rupture_times = fault.get_subfault_starttimes(
        0, velocities, nuc_dip_idx, nuc_strike_idx
    ).ravel()

    wmap = sc.wavemaps[0]
    gfs = ffi.load_gf_library(
        directory=pjoin(ffi_project.project_dir, "ffi", "linear_gfs"),
        filename=ffi.get_gf_prefix("seismic", "uparr", wmap._mapid, 0),
    )

    # snap onto the starttime grid of the library for exact comparison
    starttimes = (
        num.round((rupture_times + TIME_SHIFT) / gfs.starttime_sampling)
        * gfs.starttime_sampling
    )
    assert starttimes.min() >= gfs.starttime_min
    assert starttimes.max() <= gfs.starttime_min + gfs.starttime_sampling * (
        gfs.nstarttimes - 1
    )

    npatches = npdip * npstrike
    slips = num.linspace(0.5, 1.5, npatches)
    durations = num.full(npatches, DURATION)

    patches = fault.get_subfault_patches(0, datatype="seismic", component="uparr")
    assert len(patches) == npatches
    for patch, starttime, duration, slip in zip(patches, starttimes, durations, slips):
        patch.stf.duration = float(duration)
        patch.update(slip=float(slip), time=event.time + float(starttime))

    engine = gf.LocalEngine(store_superdirs=[ffi_project.store_superdir])
    arrival_taper = wmap.config.arrival_taper
    reference, _ = heart.seis_synthetics(
        engine,
        patches,
        wmap.targets,
        arrival_times=gfs.reference_times - arrival_taper.b,
        wavename=wmap.name,
        arrival_taper=arrival_taper,
        filterer=wmap.config.filterer,
        outmode="array",
    )

    return dict(
        gfs=gfs,
        reference=reference,
        starttimes=starttimes,
        durations=durations,
        slips=slips,
    )


def test_library_dimensions(ffi_project, rupture):
    gfs = rupture["gfs"]
    problem = ffi_project.load_problem(build=False)
    wmap = problem.composites["seismic"].wavemaps[0]

    assert gfs.ntargets == wmap.n_t
    assert gfs.npatches == 15
    assert gfs.nsamples == wmap.config.arrival_taper.nsamples(gfs.deltat**-1)
    assert rupture["reference"].shape == (gfs.ntargets, gfs.nsamples)
    assert num.abs(rupture["reference"]).max() > 0.0


@pytest.mark.parametrize("interpolation", ["nearest_neighbor", "multilinear"])
def test_stacking_numpy(rupture, interpolation):
    gfs = rupture["gfs"]
    ntargets, npatches = gfs.ntargets, gfs.npatches
    gfs.set_stack_mode("numpy")

    synthetics = gfs.stack_all(
        patchidxs=num.arange(npatches),
        targetidxs=num.atleast_2d(num.arange(ntargets)).T,
        starttimes=num.tile(rupture["starttimes"], (ntargets, 1)),
        durations=num.atleast_2d(rupture["durations"]),
        slips=rupture["slips"],
        interpolation=interpolation,
    )

    reference = rupture["reference"]
    assert_allclose(synthetics, reference, rtol=0, atol=1e-6 * num.abs(reference).max())


@pytest.mark.parametrize("interpolation", ["nearest_neighbor", "multilinear"])
def test_stacking_pytensor(rupture, interpolation):
    gfs = rupture["gfs"]
    ntargets = gfs.ntargets

    with tconfig.change_flags(compute_test_value="off"):
        gfs.init_optimization()
        synthetics = gfs.stack_all(
            targetidxs=num.atleast_2d(num.arange(ntargets)).T,
            starttimes=num.tile(rupture["starttimes"], (ntargets, 1)),
            durations=num.atleast_2d(rupture["durations"]),
            slips=rupture["slips"],
            interpolation=interpolation,
        ).eval()

    reference = rupture["reference"]
    assert_allclose(synthetics, reference, rtol=0, atol=1e-6 * num.abs(reference).max())


def test_no_shared_arrays_after_build(ffi_project):
    """
    Serial GF calculation (nworkers=1) sets the shared arrays in the main
    process, they must not leak into GF libraries created afterwards.
    """
    from beat import parallel

    assert not hasattr(parallel, "gfmatrix")
    assert not hasattr(parallel, "tmins")
