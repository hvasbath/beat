import numpy as num
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from pyrocko.moment_tensor import symmat6
from pyrocko.plot import beachball

from beat.heart import calculate_radiation_weights, radiation_matmul
from beat.plotting import draw_ray_piercing_points_bb, plot_fuzzy_beachball_mpl_pixmap
from beat.utility import get_random_uniform

WAVENAMES = ["any_P", "any_SH", "any_SV"]

# Taken from Dahm, T.: Relative moment tensor inversion based on ray
# theory: theory and synthetic tests, 1996, GJI
AZIMUTHS_DEG = [10.0, 35.0, 85.0, 55.0, 64.0, 40.0, 110.0, 125.0, 140.0, 160.0]
AZIMUTHS_DEG += [-155.0, -70.0]
TAKEOFF_ANGLES_DEG = [20.0, 45.0, 23.0, 39.0, 25.0, 32.0, 16.0, 41.0, 10.0, 41.0]
TAKEOFF_ANGLES_DEG += [70.0, 90.0]
M6 = num.array([-0.87, 0.85, 0.02, 0.48, 0.04, -0.17])


@pytest.mark.parametrize("wavename", WAVENAMES)
def test_radiation(wavename):
    azimuths_rad = num.deg2rad(AZIMUTHS_DEG)
    takeoff_angles_rad = num.deg2rad(TAKEOFF_ANGLES_DEG)

    amps = radiation_matmul(
        symmat6(*M6), takeoff_angles_rad, azimuths_rad, wavename=wavename
    )
    rad_weights = calculate_radiation_weights(
        takeoff_angles_rad, azimuths_rad, wavename=wavename
    )
    amps_weights = rad_weights.T.dot(M6)

    assert amps_weights.shape == (len(AZIMUTHS_DEG),)
    assert_allclose(amps, amps_weights, atol=1e-6, rtol=1e-6)


def test_radiation_p_isotropic():
    """P radiation of an explosion is 1 in every direction."""
    azimuths_rad = num.deg2rad(AZIMUTHS_DEG)
    takeoff_angles_rad = num.deg2rad(TAKEOFF_ANGLES_DEG)
    rad_weights = calculate_radiation_weights(
        takeoff_angles_rad, azimuths_rad, wavename="any_P"
    )
    m6_iso = num.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert_allclose(rad_weights.T.dot(m6_iso), 1.0, atol=1e-10)


def test_polarity_bb():
    nstations = 2000
    takeoff_angles_rad = num.deg2rad(
        get_random_uniform(lower=1, upper=90, dimension=nstations)
    )
    azimuths_rad = num.deg2rad(
        get_random_uniform(lower=-180.0, upper=180, dimension=nstations)
    )

    kwargs = {
        "beachball_type": "full",
        "size": 6,
        "size_units": "data",
        "position": (4, 4),
        "color_t": "black",
        "edgecolor": "black",
        "grid_resolution": 100,
    }

    fig, axs = plt.subplots(1, 3, figsize=(15.0, 5.0))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    for ax, wavename in zip(axs, WAVENAMES):
        rad_weights = calculate_radiation_weights(
            takeoff_angles_rad, azimuths_rad, wavename=wavename
        )
        amps_weights = rad_weights.T.dot(M6)

        transform, position, size = beachball.choose_transform(
            ax, kwargs["size_units"], kwargs["position"], kwargs["size"]
        )
        plot_fuzzy_beachball_mpl_pixmap(
            num.atleast_2d(M6),
            ax,
            wavename=wavename,
            best_mt=None,
            **kwargs,
        )
        draw_ray_piercing_points_bb(
            ax,
            takeoff_angles_rad,
            azimuths_rad,
            amps_weights,
            size=size,
            position=position,
            transform=transform,
            nomask=True,
            markersize=8,
        )
        ax.set_title(wavename, fontsize=12)
