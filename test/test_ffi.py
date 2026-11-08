import numpy as num
import pytensor.tensor as tt
import pytest
from numpy.testing import assert_allclose
from pyrocko import model
from pytensor import config as tconfig
from pytensor import function

from beat import ffi
from beat.config import SeismicGFLibraryConfig, WaveformFitConfig
from beat.ffi.fault import get_division_mapping
from beat.sources import RectangularSource

km = 1000.0

NSAMPLES = 10
NTARGETS = 6
NPATCHES = 8

STARTTIME_MIN, STARTTIME_MAX, STARTTIME_SAMPLING = 0.0, 15.0, 0.5
DURATION_MIN, DURATION_MAX, DURATION_SAMPLING = 5.0, 10.0, 0.5

NSTARTTIMES = int((STARTTIME_MAX - STARTTIME_MIN) / STARTTIME_SAMPLING) + 1
NDURATIONS = int((DURATION_MAX - DURATION_MIN) / DURATION_SAMPLING) + 1


def gf_entry(targetidx, patchidx, durationidx, starttimeidx):
    """Deterministic, index-dependent trace so wrong lookups are detected."""
    return (
        num.arange(NSAMPLES, dtype="float64")
        + 1000.0 * targetidx
        + 100.0 * patchidx
        + 10.0 * durationidx
        + starttimeidx
    )


@pytest.fixture
def gfs():
    event = model.Event(lat=10.0, lon=10.0, depth=2.0 * km)

    durations = num.linspace(DURATION_MIN, DURATION_MAX, NDURATIONS)
    starttimes = num.linspace(STARTTIME_MIN, STARTTIME_MAX, NSTARTTIMES)

    gfl_config = SeismicGFLibraryConfig(
        component="uparr",
        datatype="seismic",
        event=event,
        reference_sources=[RectangularSource.from_pyrocko_event(event)],
        duration_sampling=DURATION_SAMPLING,
        starttime_sampling=STARTTIME_SAMPLING,
        wave_config=WaveformFitConfig(),
        dimensions=(NTARGETS, NPATCHES, NDURATIONS, NSTARTTIMES, NSAMPLES),
        starttime_min=float(starttimes.min()),
        duration_min=float(durations.min()),
        mapnumber=1,
    )

    library = ffi.SeismicGFLibrary(config=gfl_config)
    library.setup(NTARGETS, NPATCHES, NDURATIONS, NSTARTTIMES, NSAMPLES, allocate=True)

    # fill the same way as beat.ffi.base.seis_construct_gf does:
    # one call per target, patch and starttime with all durations at once
    for targetidx in range(NTARGETS):
        library.set_patch_time(targetidx=targetidx, tmin=100.0 + targetidx)
        for patchidx in range(NPATCHES):
            for starttimeidx, starttime in enumerate(starttimes):
                entries = num.vstack(
                    [
                        gf_entry(targetidx, patchidx, durationidx, starttimeidx)
                        for durationidx in range(NDURATIONS)
                    ]
                )
                library.put(
                    entries=entries,
                    targetidx=targetidx,
                    patchidx=patchidx,
                    durations=durations,
                    starttimes=starttime,
                )

    return library


@pytest.fixture
def stacking_inputs():
    rng = num.random.default_rng(0)
    # values on the library grid so that nearest neighbor lookup is exact
    durationidxs = rng.integers(0, NDURATIONS, NPATCHES)
    starttimeidxs = rng.integers(0, NSTARTTIMES, (NTARGETS, NPATCHES))
    durations = DURATION_MIN + durationidxs * DURATION_SAMPLING
    starttimes = STARTTIME_MIN + starttimeidxs * STARTTIME_SAMPLING
    slips = rng.random(NPATCHES)
    return durations, starttimes, slips, durationidxs, starttimeidxs


def reference_stack(durationidxs, starttimeidxs, slips):
    out = num.zeros((NTARGETS, NSAMPLES))
    for t in range(NTARGETS):
        for p in range(NPATCHES):
            out[t] += slips[p] * gf_entry(t, p, durationidxs[p], starttimeidxs[t, p])
    return out


def test_gf_setup(gfs):
    assert gfs.dimensions == (NTARGETS, NPATCHES, NDURATIONS, NSTARTTIMES, NSAMPLES)
    assert gfs.ntargets == NTARGETS
    assert gfs.npatches == NPATCHES
    assert gfs.nsamples == NSAMPLES
    taper_b = gfs.config.wave_config.arrival_taper.b
    assert gfs.trace_tmin(3) == pytest.approx(103.0 + taper_b)
    assert_allclose(gfs._gfmatrix[2, 3, 4, 5], gf_entry(2, 3, 4, 5))


def test_put_wrong_number_of_samples(gfs):
    with pytest.raises(ffi.base.GFLibraryError):
        gfs.put(
            entries=num.zeros((NDURATIONS, NSAMPLES + 1)),
            targetidx=0,
            patchidx=0,
            durations=num.linspace(DURATION_MIN, DURATION_MAX, NDURATIONS),
            starttimes=STARTTIME_MIN,
        )


def test_index_conversion_roundtrip(gfs):
    idxs = num.arange(NDURATIONS)
    durationidxs, _ = gfs.durations2idxs(gfs.idxs2durations(idxs))
    assert_allclose(durationidxs, idxs)

    idxs = num.arange(NSTARTTIMES)
    starttimeidxs, _ = gfs.starttimes2idxs(gfs.idxs2starttimes(idxs))
    assert_allclose(starttimeidxs, idxs)


def test_stack_all_numpy(gfs, stacking_inputs):
    durations, starttimes, slips, durationidxs, starttimeidxs = stacking_inputs
    targetidxs = num.atleast_2d(num.arange(NTARGETS)).T

    gfs.set_stack_mode("numpy")
    out = gfs.stack_all(
        targetidxs=targetidxs,
        durations=durations,
        starttimes=starttimes,
        slips=slips,
    )
    assert_allclose(out, reference_stack(durationidxs, starttimeidxs, slips))


def test_stack_all_pytensor(gfs, stacking_inputs):
    durations, starttimes, slips, durationidxs, starttimeidxs = stacking_inputs
    targetidxs = num.atleast_2d(num.arange(NTARGETS)).T

    t_durations = tt.dvector("durations")
    t_starttimes = tt.dmatrix("starttimes")
    t_slips = tt.dvector("slips")

    gfs.init_optimization()
    with tconfig.change_flags(compute_test_value="off"):
        outstack = gfs.stack_all(
            targetidxs=targetidxs,
            durations=t_durations,
            starttimes=t_starttimes,
            slips=t_slips,
        )
        f = function([t_slips, t_durations, t_starttimes], outstack)
    out = f(slips, durations, starttimes)

    assert_allclose(out, reference_stack(durationidxs, starttimeidxs, slips))


def test_stack_single_target(gfs, stacking_inputs):
    durations, starttimes, slips, durationidxs, starttimeidxs = stacking_inputs
    reference = reference_stack(durationidxs, starttimeidxs, slips)

    gfs.set_stack_mode("numpy")
    for targetidx in range(NTARGETS):
        synths = gfs.stack(
            targetidx=targetidx,
            patchidxs=num.arange(NPATCHES),
            durations=durations,
            starttimes=starttimes[targetidx],
            slips=slips,
        )
        assert_allclose(synths, reference[targetidx])


def test_get_traces(gfs):
    traces = gfs.get_traces(
        targetidxs=[0, 1],
        patchidxs=[0],
        durationidxs=list(range(NDURATIONS)),
        starttimeidxs=[0],
    )
    assert len(traces) == 2 * NDURATIONS
    assert_allclose(traces[-1].ydata, gf_entry(1, 0, NDURATIONS - 1, 0))
    taper_b = gfs.config.wave_config.arrival_taper.b
    assert traces[-1].tmin == pytest.approx(101.0 + taper_b)


def test_division_mapping():
    old2new, div2new, subfault_npatches = get_division_mapping(
        range(5), [0, 2, 4], [2, 3]
    )
    npatches_new = len(old2new) + len(div2new)

    assert old2new[1] == 2
    assert old2new[3] == 5
    assert npatches_new == subfault_npatches.sum()
    assert subfault_npatches[0] == 3
    assert subfault_npatches[1] == 5
