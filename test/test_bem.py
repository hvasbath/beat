import numpy as num
import pytest
from matplotlib import pyplot as plt
from pyrocko.gf.targets import StaticTarget

pytest.importorskip("pygmsh")
pytest.importorskip("cutde")

from beat.bem import (  # noqa: E402
    BEMEngine,
    CurvedBEMSource,
    DiskBEMSource,
    RectangularBEMSource,
    RingfaultBEMSource,
    TriangleBEMSource,
    check_intersection,
)
from beat.config import BEMConfig  # noqa: E402
from beat.plotting.bem import slip_distribution_3d  # noqa: E402

km = 1.0e3
pi = num.pi

mesh_size = 1.0


def get_static_target(bounds, npoints=100):
    xs = num.linspace(*bounds, npoints)
    ys = num.linspace(*bounds, npoints)
    obsx, obsy = num.meshgrid(xs, ys)
    return StaticTarget(east_shifts=obsx.ravel(), north_shifts=obsy.ravel())


def plot_static_result(result, target, npoints=100):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), dpi=300)
    for i, comp in enumerate(["n", "e", "d"]):
        ax = axs[i]
        disp_grid = result[f"displacement.{comp}"].reshape((npoints, npoints))
        if comp == "d":
            disp_grid *= -1
        grd_e = target.east_shifts.reshape((npoints, npoints))
        grd_n = target.north_shifts.reshape((npoints, npoints))
        cntf = ax.contourf(grd_e, grd_n, disp_grid, levels=21)
        ax.contour(
            grd_e,
            grd_n,
            disp_grid,
            colors="k",
            linestyles="-",
            linewidths=0.5,
            levels=21,
        )
        # print("displ min max", disp_grid.min(), disp_grid.max())
        ax.set_title(f"$u_{comp}$")

    cb = plt.colorbar(cntf)
    cb.set_label("Displacement [m]")
    fig.tight_layout()
    return fig, axs


def get_triangle_tensile_only_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        TriangleBEMSource(
            normal_traction=2.15e6,
            p1=(km, km, -2 * km),
            p2=(-km, -0.5 * km, -2 * km),
            p3=(1 * km, -1.5 * km, -2 * km),
        )
    ]

    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = []
    return config, sources, targets


def get_disk_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        DiskBEMSource(
            traction=1.15e6,
            normal_traction=0,
            rake=-45,
            north_shift=0.5 * km,
            depth=3.5 * km,
            a_half_axis=3 * km,
            b_half_axis=1.8 * km,
            dip=45,
            strike=210,
        )
    ]
    return BEMConfig(mesh_size=mesh_size), sources, targets


def get_disk_tensile_only_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        DiskBEMSource(
            normal_traction=2.15e6,
            north_shift=0.5 * km,
            depth=3.5 * km,
            a_half_axis=1 * km,
            b_half_axis=1.0 * km,
            dip=0,
            strike=30,
        )
    ]
    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = []
    return config, sources, targets


def get_disk_ringfault_setup(intersect=False):
    targets = [get_static_target([-10 * km, 10 * km], 100)]

    if intersect:
        a_half_axis_bottom = 2.5 * km
        b_half_axis_bottom = 2.5 * km
        depth = 3.0 * km
    else:
        a_half_axis_bottom = 3.5 * km
        b_half_axis_bottom = 3.5 * km
        depth = 4.2 * km

    sources = [
        DiskBEMSource(
            normal_traction=2.15e6,
            north_shift=0.0 * km,
            east_shift=3.5 * km,
            depth=depth,
            a_half_axis=a_half_axis_bottom,
            b_half_axis=b_half_axis_bottom,
            dip=0,
            strike=0,
        ),
        RingfaultBEMSource(
            north_shift=0.0,
            east_shift=3.5 * km,
            delta_north_shift_bottom=0.0 * km,
            depth=0.5 * km,
            depth_bottom=3.9 * km,
            a_half_axis=2 * km,
            b_half_axis=2 * km,
            a_half_axis_bottom=a_half_axis_bottom,
            b_half_axis_bottom=b_half_axis_bottom,
            strike=5,
        ),
    ]

    config = BEMConfig(mesh_size=1.5)
    for bcond in config.boundary_conditions.iter_conditions():
        bcond.receiver_idxs = [0, 1]
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = [1]

    return config, sources, targets


def get_rectangular_setup_strikeslip():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        RectangularBEMSource(
            traction=1.15e6,
            rake=0,
            north_shift=0.5 * km,
            depth=3.5 * km,
            length=10 * km,
            width=5 * km,
            dip=75,
            strike=20,
        )
    ]
    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["normal"]:
            bcond.source_idxs = []

    return config, sources, targets


def get_rectangular_setup_dipslip():
    # mesh_size = 1. * km
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        RectangularBEMSource(
            traction=1.15e6,
            rake=90,
            north_shift=0.5 * km,
            depth=3.5 * km,
            length=10 * km,
            width=5 * km,
            dip=30,
            strike=0,
        )
    ]
    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["normal"]:
            bcond.source_idxs = []

    return config, sources, targets


def get_rectangular_setup_opening():
    # mesh_size = 1. * km
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        RectangularBEMSource(
            normal_traction=1.15e6,
            rake=90,
            north_shift=0.5 * km,
            depth=3.5 * km,
            length=10 * km,
            width=5 * km,
            dip=30,
            strike=0,
        )
    ]
    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["dip", "strike"]:
            bcond.source_idxs = []

    return config, sources, targets


def get_curved_setup_dipslip():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        CurvedBEMSource(
            traction=1.15e6,
            rake=90,
            north_shift=0.5 * km,
            depth=3.5 * km,
            length=15 * km,
            width=7 * km,
            dip=30,
            strike=310,
            bend_location=0.5,
            bend_amplitude=0.3,
            curv_location_bottom=0.0,
            curv_amplitude_bottom=0.0,
        )
    ]
    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["normal"]:
            bcond.source_idxs = []

    return config, sources, targets


def run_bem_engine(setup_function, **kwargs):
    config, sources, targets = setup_function(**kwargs)

    engine = BEMEngine(config)
    response = engine.process(sources=sources, targets=targets)
    return response, sources, targets


@pytest.mark.parametrize(
    "setup_function, kwargs",
    [
        (get_triangle_tensile_only_setup, {}),
        (get_disk_tensile_only_setup, {}),
        (get_disk_setup, {}),
        (get_rectangular_setup_strikeslip, {}),
        (get_rectangular_setup_dipslip, {}),
        (get_rectangular_setup_opening, {}),
        (get_curved_setup_dipslip, {}),
        (get_disk_ringfault_setup, {"intersect": False}),
    ],
    ids=lambda x: getattr(x, "__name__", str(x)),
)
def test_bem_engine(setup_function, kwargs):
    response, sources, targets = run_bem_engine(setup_function, **kwargs)

    assert response.is_valid

    slip_vectors = response.source_slips()
    assert len(slip_vectors) == len(sources)
    for slips in slip_vectors:
        assert slips.ndim == 2 and slips.shape[1] == 3
        assert num.isfinite(slips).all()
        assert num.abs(slips).max() > 0.0

    results = response.static_results()
    assert len(results) == len(targets)
    for result, target in zip(results, targets):
        for component in ["displacement.n", "displacement.e", "displacement.d"]:
            displacement = result.result[component]
            assert displacement.size == target.east_shifts.size
            assert num.isfinite(displacement).all()

        assert num.abs(result.result["displacement.d"]).max() > 0.0
        plot_static_result(result.result, target)

    slip_distribution_3d(response.discretized_sources, slip_vectors, debug=True)


def test_bem_engine_intersecting_sources_invalid():
    response, _, _ = run_bem_engine(get_disk_ringfault_setup, intersect=True)
    assert not response.is_valid


@pytest.mark.parametrize("intersect", [True, False])
def test_bem_source_intersection(intersect):
    config, sources, _ = get_disk_ringfault_setup(intersect=intersect)
    assert check_intersection(sources, mesh_size=config.mesh_size * km) is intersect
