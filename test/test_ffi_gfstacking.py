from os.path import join as pjoin
from pathlib import Path

import numpy as num
from pyrocko import gf, trace, util
from pytensor import config as tconfig
from pytest import mark

from beat import ffi, heart, models

tconfig.compute_test_value = "off"

util.setup_logging("test_ffi_stacking", "info")


# set random seed for reproducible station locations
num.random.seed(10)

km = 1000.0
nuc_dip = 5.0
nuc_strike = 2.0
time_shift = -10.0  # from previous inversion

project_dir = Path("/home/vasyurhm/BEATS/LaquilaJointPonlyUPDATE_wide_kin3_v2")


# @mark.skip("Needs version 2.0.0 compliant setup")
@mark.skipif(project_dir.is_dir() is False, reason="Needs project dir")
def test_gf_stacking():
    # general
    store_superdirs = ["/home/vasyurhm/GF/Laquila"]

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

    print(starttimes)

    # defining distributed slip values for slip parallel and perpendicular directions
    uparr = num.ones((npdip, npstrike)) * 2.0
    # uparr[1:3, 3:7] = 1.5
    uperp = num.zeros((npdip, npstrike))
    # uperp[0,0] = 1.
    # uperp[3,9] = 1.
    uperp[1:3, 3:7] = 1.0

    # define rupture durations on each patch
    durations = num.ones((npdip, npstrike)) * 0.5

    slips = {
        components[0]: uparr.ravel(),
        #    components[1]: uperp.ravel(),
        "durations": durations.ravel(),
        "velocities": velocities.ravel(),
    }

    print("fault parameters", slips)

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

    patchidx = fault.patchmap(index=0, dipidx=nuc_dip_idx, strikeidx=nuc_strike_idx)  # noqa: F841

    targets = sc.wavemaps[0].targets
    filterer = sc.wavemaps[0].config.filterer
    ntargets = len(targets)

    gfs = ffi.load_gf_library(
        directory=pjoin(project_dir, "ffi/linear_gfs/"),
        filename="seismic_uparr_any_P_0",
    )
    ats = gfs.reference_times - arrival_taper.b

    if False:
        traces, tmins = heart.seis_synthetics(
            engine,
            patches,
            targets,
            arrival_times=ats,
            wavename="any_P",
            arrival_taper=arrival_taper,
            filterer=filterer,
            outmode="stacked_traces",
        )

    targetidxs = num.lib.index_tricks.s_[:]

    if False:
        # for station corrections maybe in the future?
        station_corrections = num.zeros(len(traces))
        starttimes = (
            num.tile(starttimes, ntargets)
            + num.repeat(station_corrections, fault.npatches)
        ).reshape(ntargets, fault.npatches)
        targetidxs = num.atleast_2d(num.arange(ntargets)).T

    durations_dim2 = num.atleast_2d(durations.ravel())
    gfs.set_stack_mode("numpy")
    synthetics_nn = gfs.stack_all(
        targetidxs=targetidxs,
        starttimes=starttimes,
        durations=durations_dim2,
        slips=slips[components[0]],
        interpolation="nearest_neighbor",
    )

    synthetics_ml = gfs.stack_all(
        targetidxs=targetidxs,
        starttimes=starttimes,
        durations=durations_dim2,
        slips=slips[components[0]],
        interpolation="multilinear",
    )

    gfs.init_optimization()

    if True:
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

    synth_traces_nn = []
    for i, target in enumerate(targets):
        tr = trace.Trace(
            ydata=synthetics_nn[i, :], tmin=gfs.reference_times[i], deltat=gfs.deltat
        )
        # print('trace tmin synthst', tr.tmin)
        tr.set_codes(*target.codes)
        tr.set_location("nn")
        synth_traces_nn.append(tr)

    synth_traces_ml = []
    for i, target in enumerate(targets):
        tr = trace.Trace(
            ydata=synthetics_ml[i, :], tmin=gfs.reference_times[i], deltat=gfs.deltat
        )
        # print 'trace tmin synthst', tr.tmin
        tr.set_codes(*target.codes)
        tr.set_location("ml")
        synth_traces_ml.append(tr)

    if True:
        synth_traces_nn_t = []
        for i, target in enumerate(targets):
            tr = trace.Trace(
                ydata=synthetics_nn_t[i, :],
                tmin=gfs.reference_times[i],
                deltat=gfs.deltat,
            )
            # print('trace tmin synthst', tr.tmin)
            tr.set_codes(*target.codes)
            tr.set_location("nn_t")
            synth_traces_nn_t.append(tr)

        synth_traces_ml_t = []
        for i, target in enumerate(targets):
            tr = trace.Trace(
                ydata=synthetics_ml_t[i, :],
                tmin=gfs.reference_times[i],
                deltat=gfs.deltat,
            )
            # print 'trace tmin synthst', tr.tmin
            tr.set_codes(*target.codes)
            tr.set_location("ml_t")
            synth_traces_ml_t.append(tr)

    # display to check
    trace.snuffle(
        # traces
        synth_traces_nn + synth_traces_ml + synth_traces_nn_t + synth_traces_ml_t,
        stations=sc.wavemaps[0].stations,
        events=[event],
    )

    traces1, tmins = heart.seis_synthetics(
        engine,
        [patches[0]],
        targets,
        arrival_times=ats,
        wavename="any_P",
        arrival_taper=arrival_taper,
        filterer=filterer,
        outmode="stacked_traces",
    )

    gfs.set_stack_mode("numpy")

    synth_traces_ml1 = []
    for i in range(1):
        synthetics_ml1 = gfs.stack_all(
            targetidxs=targetidxs,
            patchidxs=[i],
            starttimes=starttimes[0],
            durations=num.atleast_2d(durations.ravel()[0]),
            slips=num.atleast_1d(slips[components[0]][0]),
            interpolation="multilinear",
        )

        for i, target in enumerate(targets):
            tr = trace.Trace(
                ydata=synthetics_ml1[i, :],
                tmin=gfs.reference_times[i],
                deltat=gfs.deltat,
            )
            print("trace tmin synthst", tr.tmin)
            # print(target.codes)
            tr.set_codes(*target.codes)
            tr.set_location("ml%i" % i)
            synth_traces_ml1.append(tr)

    trace.snuffle(
        traces1 + synth_traces_ml1, stations=sc.wavemaps[0].stations, events=[event]
    )
