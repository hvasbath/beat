from .common import *  # noqa
from .ffi import *  # noqa
from .geodetic import *  # noqa
from .marginals import *  # noqa
from .seismic import *  # noqa

plots_catalog = {
    "correlation_hist": draw_correlation_hist,  # noqa: F405
    "stage_posteriors": draw_posteriors,  # noqa: F405
    "waveform_fits": draw_seismic_fits,  # noqa: F405
    "scene_fits": draw_scene_fits,  # noqa: F405
    "gnss_fits": draw_gnss_fits,  # noqa: F405
    "geodetic_covariances": draw_geodetic_covariances,  # noqa: F405
    "velocity_models": draw_earthmodels,  # noqa: F405
    "slip_distribution": draw_slip_dist,  # noqa: F405
    "slip_distribution_3d": draw_3d_slip_distribution,  # noqa: F405
    "hudson": draw_hudson,  # noqa: F405
    "lune": draw_lune_plot,  # noqa: F405
    "fuzzy_beachball": draw_fuzzy_beachball,  # noqa: F405
    "fuzzy_mt_decomp": draw_fuzzy_mt_decomposition,  # noqa: F405
    "moment_rate": draw_moment_rate,  # noqa: F405
    "station_map": draw_station_map_gmt,  # noqa: F405
    "station_variance_reductions": draw_station_variance_reductions,  # noqa: F405
}


common_plots = ["stage_posteriors"]


seismic_plots = [
    "station_map",
    "waveform_fits",
    "fuzzy_mt_decomp",
    "hudson",
    "lune",
    "station_variance_reductions",
]


geodetic_plots = ["scene_fits", "gnss_fits", "geodetic_covariances"]
polarity_plots = ["fuzzy_mt_decomp", "lune", "hudson", "station_map"]

geometry_plots = ["correlation_hist", "velocity_models", "fuzzy_beachball"]
bem_plots = ["correlation_hist", "slip_distribution_3d", "fuzzy_beachball"]
ffi_plots = ["moment_rate", "slip_distribution", "slip_distribution_3d"]

plots_mode_catalog = {
    "geometry": common_plots + geometry_plots,
    "ffi": common_plots + ffi_plots,
    "bem": common_plots + bem_plots,
}

plots_datatype_catalog = {
    "seismic": seismic_plots,
    "geodetic": geodetic_plots,
    "polarity": polarity_plots,
}


def available_plots(mode=None, datatypes=["geodetic", "seismic"]):
    if mode is None:
        return list(plots_catalog.keys())
    else:
        plots = plots_mode_catalog[mode]
        for datatype in datatypes:
            plots.extend(plots_datatype_catalog[datatype])

        return list(set(plots))
