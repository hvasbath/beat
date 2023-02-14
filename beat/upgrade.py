"""
Config file upgrading module modified from grond
"""

import copy
import difflib
import os
import sys
from logging import getLogger

from pyrocko import guts
from pyrocko import guts_agnostic as aguts

logger = getLogger("upgrade")


def rename_attribute(old, new):
    def func(path, obj):
        if old in obj:
            obj.rename_attribute(old, new)

    return func


def rename_class(new):
    def func(path, obj):
        obj._tagname = new

    return func


def drop_attribute(old, newclass=None):
    def func(path, obj):
        if old in obj:
            if newclass is not None:
                oldclass = obj.get(old)
                if not isinstance(oldclass, newclass):
                    obj.drop_attribute(old)
                else:
                    logger.debug(
                        'Not updating "%s" attribute, it already meets' " new class.",
                        old,
                    )
            else:
                obj.drop_attribute(old)

    return func


def set_attribute(k, v, cond=None):
    def func(path, obj):
        if cond is not None:
            if obj[k] == cond:
                obj[k] = v
        else:
            obj[k] = v

    return func


def color_diff(diff):
    green = "\x1b[32m"
    red = "\x1b[31m"
    blue = "\x1b[34m"
    dim = "\x1b[2m"
    reset = "\x1b[0m"

    for line in diff:
        if line.startswith("+"):
            yield green + line + reset
        elif line.startswith("-"):
            yield red + line + reset
        elif line.startswith("^"):
            yield blue + line + reset
        elif line.startswith("@"):
            yield dim + line + reset
        else:
            yield line


def upgrade_config_file(fn, diff=True, update=[]):
    rules = [
        ("beat.SeismicConfig", drop_attribute("blacklist")),
        ("beat.SeismicConfig", drop_attribute("calc_data_cov")),
        ("beat.GeodeticConfig", drop_attribute("calc_data_cov")),
        (
            "beat.SeismicConfig",
            set_attribute(
                "noise_estimator",
                aguts.load(
                    string="""!beat.SeismicNoiseAnalyserConfig
                      structure: variance
                      pre_arrival_time: 5
                    """
                ),
                False,
            ),
        ),
        (
            "beat.ProblemConfig",
            drop_attribute("dataset_specific_residual_noise_estimation"),
        ),
        ("beat.ProblemConfig", set_attribute("mode", "ffi", "ffo")),
        ("beat.WaveformFitConfig", set_attribute("preprocess_data", True, True)),
        (
            "beat.WaveformFitConfig",
            set_attribute(
                "filterer",
                aguts.load(
                    string="""
                    - !beat.heart.Filter
                      lower_corner: 0.01
                      upper_corner: 0.1
                      order: 4
                      """
                ),
            ),
        ),
        ("beat.MetropolisConfig", drop_attribute("n_stages")),
        ("beat.MetropolisConfig", drop_attribute("stage")),
        ("beat.ParallelTemperingConfig", set_attribute("resample", False, False)),
        ("beat.FFOConfig", rename_class("beat.FFIConfig")),
        ("beat.GeodeticLinearGFConfig", drop_attribute("extension_widths")),
        ("beat.GeodeticLinearGFConfig", drop_attribute("extension_lengths")),
        ("beat.GeodeticLinearGFConfig", drop_attribute("patch_widths")),
        ("beat.GeodeticLinearGFConfig", drop_attribute("patch_lengths")),
        ("beat.SeismicLinearGFConfig", drop_attribute("extension_widths")),
        ("beat.SeismicLinearGFConfig", drop_attribute("extension_lengths")),
        ("beat.SeismicLinearGFConfig", drop_attribute("patch_widths")),
        ("beat.SeismicLinearGFConfig", drop_attribute("patch_lengths")),
        ("beat.GeodeticConfig", drop_attribute("fit_plane")),
        ("beat.GeodeticConfig", drop_attribute("datadir")),
        ("beat.GeodeticConfig", drop_attribute("names")),
        ("beat.GeodeticConfig", drop_attribute("blacklist")),
        ("beat.GeodeticConfig", drop_attribute("types", dict)),
        ("beat.EulerPoleConfig", rename_attribute("blacklist", "station_blacklist")),
    ]

    def apply_rules(path, obj):
        for tagname, func in rules:
            if obj._tagname == tagname:
                func(path, obj)

    updates_avail = ["hierarchicals", "hypers", "structure"]

    t1 = aguts.load(filename=fn)
    t2 = copy.deepcopy(t1)

    for upd in update:
        if upd not in updates_avail:
            raise TypeError('Update not available for "%s"' % upd)

    n_upd = len(update)
    if n_upd > 0:
        fn_tmp = fn + "tmp"
        if "structure" in update:
            aguts.apply_tree(t2, apply_rules)

        aguts.dump(t2, filename=fn_tmp, header=True)
        t2 = guts.load(filename=fn_tmp)
        if "hypers" in update:
            t2.update_hypers()

        if "hierarchicals" in update:
            t2.update_hierarchicals()

        t2.problem_config.validate_priors()
        guts.dump(t2, filename=fn_tmp)
        t2 = aguts.load(filename=fn_tmp)
    else:
        fn_tmp = fn

    s1 = aguts.dump(t1)
    s2 = aguts.dump(t2)

    if diff:
        result = list(
            difflib.unified_diff(
                s1.splitlines(1), s2.splitlines(1), "normalized old", "normalized new"
            )
        )

        if sys.stdout.isatty():
            sys.stdout.writelines(color_diff(result))
        else:
            sys.stdout.writelines(result)
    else:
        aguts.dump(t2, filename=fn_tmp, header=True)
        upd_config = guts.load(filename=fn_tmp)
        if "hypers" in update:
            logger.info(
                "Updated hyper parameters! Previous hyper"
                " parameter bounds are invalid now!"
            )

        if "hierarchicals" in update:
            logger.info("Updated hierarchicals.")

        guts.dump(upd_config, filename=fn)

    if n_upd > 0:
        os.remove(fn_tmp)


if __name__ == "__main__":
    fn = sys.argv[1]
    upgrade_config_file(fn)
