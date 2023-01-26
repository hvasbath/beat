from .common import *  # noqa
from .ffi import *  # noqa
from .geodetic import *  # noqa
from .marginals import *  # noqa
from .seismic import *  # noqa

plots_catalog = {
    "correlation_hist": draw_correlation_hist,
    "stage_posteriors": draw_posteriors,
    "waveform_fits": draw_seismic_fits,
    "scene_fits": draw_scene_fits,
    "gnss_fits": draw_gnss_fits,
    "geodetic_covariances": draw_geodetic_covariances,
    "velocity_models": draw_earthmodels,
    "slip_distribution": draw_slip_dist,
    "slip_distribution_3d": draw_3d_slip_distribution,
    "hudson": draw_hudson,
    "lune": draw_lune_plot,
    "fuzzy_beachball": draw_fuzzy_beachball,
    "fuzzy_mt_decomp": draw_fuzzy_mt_decomposition,
    "moment_rate": draw_moment_rate,
    "station_map": draw_station_map_gmt,
}


common_plots = ["stage_posteriors"]


seismic_plots = [
    "station_map",
    "waveform_fits",
    "fuzzy_mt_decomp",
    "hudson",
    "lune",
    "fuzzy_beachball",
]


geodetic_plots = ["scene_fits", "gnss_fits", "geodetic_covariances"]
polarity_plots = ["fuzzy_beachball", "fuzzy_mt_decomp", "lune", "hudson", "station_map"]

geometry_plots = ["correlation_hist", "velocity_models"]


ffi_plots = ["moment_rate", "slip_distribution"]


plots_mode_catalog = {
    "geometry": common_plots + geometry_plots,
    "ffi": common_plots + ffi_plots,
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
