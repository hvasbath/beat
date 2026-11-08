import numpy as num
import pytest
from matplotlib import pyplot as plt

from beat.plotting import (
    draw_line_on_array,
    format_axes,
    hist2d_plot_op,
    lune_plot,
    spherical_kde_op,
)


def test_draw_line_array():
    amplitude = 10
    x = num.arange(0, 100, 0.5)
    y = amplitude * num.sin(x)
    y2 = amplitude / 2.0 * num.cos(x)

    grid, extent = draw_line_on_array(x, y, grid_resolution=(600, 200), linewidth=2)
    assert grid.shape == (600, 200)
    assert extent[0] == pytest.approx(x.min())
    assert extent[1] == pytest.approx(x.max())
    nonzero = num.count_nonzero(grid)
    assert nonzero > 0

    grid, extent = draw_line_on_array(
        x, y2, grid=grid, extent=extent, grid_resolution=(600, 200), linewidth=2
    )
    assert num.count_nonzero(grid) >= nonzero

    ax = plt.axes()
    ax.matshow(grid)


def test_spherical_kde_op():
    nsamples = 10
    lats0 = num.rad2deg(num.random.normal(loc=0.0, scale=0.1, size=nsamples))
    lons0 = num.rad2deg(num.random.normal(loc=-3.14, scale=0.3, size=nsamples))

    kde, lats, lons = spherical_kde_op(lats0, lons0, grid_size=(100, 100))

    assert num.isfinite(kde).all()
    assert (kde >= 0).all()

    ax = plt.axes()
    im = ax.matshow(kde, extent=(-180, 180, -90, 90), origin="lower")
    plt.colorbar(im)


@pytest.mark.gmt
def test_lune_plot(tmp_path):
    nsamples = 1100
    # latitude
    w = num.random.normal(loc=0.5, scale=0.1, size=nsamples)
    w_bound = 3.0 * num.pi / 8.0
    w = num.clip(w, -w_bound, w_bound)
    # longitude
    v = num.random.normal(loc=-0.1, scale=0.05, size=nsamples)
    v_bound = 1.0 / 3.0
    v = num.clip(v, -v_bound, v_bound)

    gmt = lune_plot(v_tape=v, w_tape=w)
    outpath = tmp_path / "lune_test.pdf"
    gmt.save(str(outpath), resolution=300, size=10)
    assert outpath.stat().st_size > 0


def test_hist2d_plot_op():
    rng = num.random.default_rng(1001)
    ndraws = 300
    ones = num.ones((ndraws))

    fig, axs = plt.subplots(nrows=1, ncols=10)

    locs = rng.random(10) * 100
    for i, ax in enumerate(axs):
        variance_red = rng.normal(loc=locs[i], scale=2, size=ndraws)
        hist2d_plot_op(ax, ones, variance_red, bins=(1, 40))
        ax.set_ylim(locs.min() - 4, locs.max() + 4)
        if i > 0:
            format_axes(ax)
            ax.get_yaxis().set_ticklabels([])
        elif i == 0:
            format_axes(ax, remove=["top", "right"])
            ax.set_ylabel("VR [%]")

        xax = ax.get_xaxis()
        xax.set_ticks([1])
        xax.set_ticklabels([])
        ax.set_xlabel("%i0" % i, rotation=90)
