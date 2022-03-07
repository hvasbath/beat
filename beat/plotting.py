from pymc3 import plots as pmp
from pymc3 import quantiles

import math
import os
import logging
import copy

from beat import utility
from beat.models import Stage, load_stage
from beat.models.corrections import StrainRateCorrection

from beat.sampler.metropolis import get_trace_stats
from beat.heart import (DynamicTarget, SpectrumTarget, init_seismic_targets, init_geodetic_targets,
                        physical_bounds, StrainRateTensor)
from beat.config import ffi_mode_str, geometry_mode_str, dist_vars

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import kde
import numpy as num
from theano import config as tconfig
from pyrocko.guts import (Object, String, Dict, List,
                          Bool, Int, load, StringChoice)
from pyrocko import util, trace
from pyrocko import cake_plot as cp
from pyrocko import orthodrome as otd

from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light
from pyrocko.plot import beachball, nice_value, AutoScaler
from pyrocko import gmtpy

import pyrocko.moment_tensor as mt
from pyrocko.plot import mpl_papersize, mpl_init, mpl_graph_color, mpl_margins

logger = logging.getLogger('plotting')

km = 1000.


__all__ = [
    'PlotOptions', 'correlation_plot', 'correlation_plot_hist',
    'get_result_point', 'seismic_fits', 'scene_fits', 'traceplot',
    'histplot_op']



