#!/usr/bin/env python
import os
from glob import glob
from os.path import basename
from os.path import join as pjoin

# disable internal blas parallelisation as we parallelise over chains
nthreads = "1"
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads

import copy
import logging
import shutil
import sys
from collections import OrderedDict
from optparse import OptionParser

from numpy import array, atleast_2d, floor, zeros, cumsum
from pyrocko import model, util
from pyrocko.gf import LocalEngine
from pyrocko.guts import Dict, dump, load
from pyrocko.trace import snuffle
from tqdm import tqdm

from beat import config as bconfig
from beat import heart, inputf, plotting, utility
from beat.backend import backend_catalog, extract_bounds_from_summary, thin_buffer
from beat.config import dist_vars, ffi_mode_str, geometry_mode_str
from beat.info import version
from beat.models import Stage, estimate_hypers, load_model, sample
from beat.sampler.pt import SamplingHistory
from beat.sampler.smc import sample_factor_final_stage
from beat.sources import MTQTSource, MTSourceWithMagnitude
from beat.utility import list2string, string2slice

logger = logging.getLogger("beat")


km = 1000.0


def d2u(d):
    return dict((k.replace("-", "_"), v) for (k, v) in d.items())


subcommand_descriptions = {
    "init": "create a new EQ model project, use only event"
    " name to skip catalog search",
    "import": "import data or results, from external format or "
    "modeling results, respectively",
    "update": "update configuration file",
    "sample": "sample the solution space of the problem",
    "build_gfs": "build GF stores",
    "clone": "clone EQ model project into new directory",
    "plot": "plot specified setups or results",
    "check": "check setup specific requirements",
    "summarize": "collect results and create statistics",
    "export": "export waveforms and displacement maps of" " specific solution(s)",
}

subcommand_usages = {
    "init": 'init <event_name> <event_date "YYYY-MM-DD"> ' "[options]",
    "import": "import <event_name> [options]",
    "update": "update <event_name> [options]",
    "sample": "sample <event_name> [options]",
    "build_gfs": "build_gfs <event_name> [options]",
    "clone": "clone <event_name> <cloned_event_name> [options]",
    "plot": "plot <event_name> <plot_type> [options]",
    "check": "check <event_name> [options]",
    "summarize": "summarize <event_name> [options]",
    "export": "export <event_name> [options]",
}

subcommands = list(subcommand_descriptions.keys())
subcommand_descriptions["version"] = version

program_name = "beat"

usage = (
    program_name
    + """ <subcommand> <arguments> ... [options]
BEAT: Bayesian earthquake analysis tool
 Version %(version)s
author: Hannes Vasyura-Bathke
email: hannes.vasyura-bathke@kaust.edu.sa

Subcommands:

    init            %(init)s
    clone           %(clone)s
    import          %(import)s
    update          %(update)s
    build_gfs       %(build_gfs)s
    sample          %(sample)s
    summarize       %(summarize)s
    export          %(export)s
    plot            %(plot)s
    check           %(check)s

To get further help and a list of available options for any subcommand run:

    beat <subcommand> --help

"""
    % d2u(subcommand_descriptions)
)


nargs_dict = {
    "init": 2,
    "clone": 2,
    "plot": 2,
    "import": 1,
    "update": 1,
    "build_gfs": 1,
    "sample": 1,
    "check": 1,
    "summarize": 1,
    "export": 1,
}

mode_choices = [geometry_mode_str, ffi_mode_str]

supported_geodetic_formats = ["matlab", "ascii", "kite"]
supported_geodetic_types = ["SAR", "GNSS"]
supported_samplers = ["SMC", "Metropolis", "PT"]


def add_common_options(parser):
    parser.add_option(
        "--loglevel",
        action="store",
        dest="loglevel",
        type="choice",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
        help="set logger level to "
        '"critical", "error", "warning", "info", or "debug". '
        'Default is "%default".',
    )


def get_project_directory(args, options, nargs=1, popflag=False):

    largs = len(args)

    if largs == nargs - 1:
        project_dir = os.getcwd()
    elif largs == nargs:
        if popflag:
            name = args.pop(0)
        else:
            name = args[0]
        project_dir = pjoin(os.path.abspath(options.main_path), name)
    else:
        project_dir = os.getcwd()

    return project_dir


def process_common_options(options, project_dir):
    utility.setup_logging(project_dir, options.loglevel, logfilename="BEAT_log.txt")


def die(message, err=""):
    sys.exit("%s: error: %s \n %s" % (program_name, message, err))


def cl_parse(command, args, setup=None, details=None):
    usage = subcommand_usages[command]
    descr = subcommand_descriptions[command]

    if isinstance(usage, str):
        usage = [usage]

    susage = "%s %s" % (program_name, usage[0])
    for s in usage[1:]:
        susage += "\n%s%s %s" % (" " * 7, program_name, s)

    description = descr[0].upper() + descr[1:] + "."

    if details:
        description = description + " %s" % details

    parser = OptionParser(usage=susage, description=description)

    if setup:
        setup(parser)

    add_common_options(parser)
    (options, args) = parser.parse_args(args)
    project_dir = get_project_directory(args, options, nargs_dict[command])

    if command != "init":
        process_common_options(options, project_dir)
    return parser, options, args


def list_callback(option, opt, value, parser):
    out = [ival.lstrip() for ival in value.split(",")]
    if out == [""]:
        out = []
    setattr(parser.values, option.dest, out)


def get_sampled_slip_variables(config):
    slip_varnames = config.problem_config.get_slip_variables()
    rvs, fixed_rvs = config.problem_config.get_random_variables()

    varnames = list(set(slip_varnames).intersection(set(list(rvs.keys()))))
    return varnames


def command_init(args):

    command_str = "init"

    def setup(parser):

        parser.add_option(
            "--min_mag",
            dest="min_mag",
            type=float,
            default=6.0,
            help="Minimum Mw for event, for catalog search." ' Default: "6.0"',
        )

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) for creating directory structure."
            "  Default: current directory ./",
        )

        parser.add_option(
            "--datatypes",
            default=["geodetic"],
            type="string",
            action="callback",
            callback=list_callback,
            help='Datatypes to include in the setup; "geodetic, seismic".',
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--source_type",
            dest="source_type",
            choices=bconfig.source_names,
            default="RectangularSource",
            help="Source type to solve for; %s"
            '. Default: "RectangularSource"'
            % ('", "'.join(name for name in bconfig.source_names)),
        )

        parser.add_option(
            "--n_sources",
            dest="n_sources",
            type="int",
            default=1,
            help="Integer Number of sources to invert for. Default: 1",
        )

        parser.add_option(
            "--waveforms",
            type="string",
            action="callback",
            callback=list_callback,
            default=["any_P", "any_S"],
            help='Waveforms to include in the setup; "any_P, any_S, slowest".',
        )

        parser.add_option(
            "--sampler",
            dest="sampler",
            choices=supported_samplers,
            default="SMC",
            help="Sampling algorithm to sample the solution space of the"
            " general problem; %s. "
            'Default: "SMC"' % list2string(supported_samplers),
        )

        parser.add_option(
            "--hyper_sampler",
            dest="hyper_sampler",
            type="string",
            default="Metropolis",
            help="Sampling algorithm to sample the solution space of the"
            ' hyperparameters only; So far only "Metropolis" supported.'
            'Default: "Metropolis"',
        )

        parser.add_option(
            "--use_custom",
            dest="use_custom",
            action="store_true",
            help="If set, a slot for a custom velocity model is being created"
            " in the configuration file.",
        )

        parser.add_option(
            "--individual_gfs",
            dest="individual_gfs",
            action="store_true",
            help="If set, Green's Function stores will be created individually"
            " for each station!",
        )

    parser, options, args = cl_parse("init", args, setup=setup)

    la = len(args)

    if la > 2 or la < 1:
        logger.error("Wrong number of input arguments!")
        parser.print_help()
        sys.exit(1)

    if la == 2:
        name, date = args
    elif la == 1:
        logger.info("Doing no catalog search for event information!")
        name = args[0]
        date = None

    project_dir = pjoin(os.path.abspath(options.main_path), name)

    util.ensuredir(project_dir)
    process_common_options(options, project_dir)
    return bconfig.init_config(
        name,
        date,
        main_path=options.main_path,
        min_magnitude=options.min_mag,
        datatypes=options.datatypes,
        mode=options.mode,
        source_type=options.source_type,
        n_sources=options.n_sources,
        waveforms=options.waveforms,
        sampler=options.sampler,
        hyper_sampler=options.hyper_sampler,
        use_custom=options.use_custom,
        individual_gfs=options.individual_gfs,
    )


def command_import(args):

    from pyrocko import io

    command_str = "import"

    data_formats = io.allowed_formats("load")[2::]

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--results",
            dest="results",
            type="string",
            default="",
            help="Import results from previous modeling step" " of given project path",
        )

        parser.add_option(
            "--datatypes",
            default=["geodetic"],
            type="string",
            action="callback",
            callback=list_callback,
            help='Datatypes to import; "geodetic, seismic".',
        )

        parser.add_option(
            "--geodetic_format",
            dest="geodetic_format",
            type="string",
            default=["kite"],
            action="callback",
            callback=list_callback,
            help='Data format to be imported; %s Default: "kite"'
            % list2string(supported_geodetic_formats),
        )

        parser.add_option(
            "--seismic_format",
            dest="seismic_format",
            default="mseed",
            choices=data_formats,
            help="Data format to be imported;"
            'Default: "mseed"; Available: %s' % list2string(data_formats),
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem results to import; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--import_from_mode",
            dest="import_from_mode",
            choices=mode_choices,
            default=ffi_mode_str,
            help="The mode to import estimation results"
            ' from; %s Default: "%s"' % (list2string(mode_choices), ffi_mode_str),
        )

        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="Overwrite existing files",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    if not options.results:
        c = bconfig.load_config(project_dir, options.mode)

        if "seismic" in options.datatypes:
            sc = c.seismic_config
            logger.info("Attempting to import seismic data from %s" % sc.datadir)

            seismic_outpath = pjoin(c.project_dir, bconfig.seismic_data_name)
            if not os.path.exists(seismic_outpath) or options.force:
                # TODO datahandling multi event based ...
                stations = model.load_stations(pjoin(sc.datadir, "stations.txt"))

                if options.seismic_format == "autokiwi":

                    data_traces = inputf.load_data_traces(
                        datadir=sc.datadir, stations=stations, divider="-"
                    )

                elif options.seismic_format in data_formats:
                    data_traces = inputf.load_data_traces(
                        datadir=sc.datadir,
                        stations=stations,
                        divider=".",
                        data_format=options.seismic_format,
                    )

                else:
                    raise TypeError(
                        "Format: %s not implemented yet." % options.seismic_format
                    )

                logger.info("Rotating traces to RTZ wrt. event!")
                data_traces = inputf.rotate_traces_and_stations(
                    data_traces, stations, c.event
                )

                logger.info("Pickle seismic data to %s" % seismic_outpath)
                utility.dump_objects(seismic_outpath, outlist=[stations, data_traces])
                logger.info(
                    "Successfully imported traces for" " %i stations!" % len(stations)
                )
            else:
                logger.info("%s exists! Use --force to overwrite!" % seismic_outpath)

        if "geodetic" in options.datatypes:
            gc = c.geodetic_config

            geodetic_outpath = pjoin(c.project_dir, bconfig.geodetic_data_name)
            if not os.path.exists(geodetic_outpath) or options.force:

                gtargets = []
                for typ, config in gc.types.items():
                    logger.info(
                        "Attempting to import geodetic data for typ %s from %s"
                        % (typ, config.datadir)
                    )
                    if typ == "SAR":
                        if "matlab" in options.geodetic_format:
                            gtargets.extend(
                                inputf.load_SAR_data(config.datadir, config.names)
                            )
                        elif "kite" in options.geodetic_format:
                            gtargets.extend(config.load_data())
                        else:
                            raise ImportError(
                                "Format %s not implemented yet for SAR data."
                                % options.geodetic_format
                            )

                    elif typ == "GNSS":
                        if "ascii" in options.geodetic_format:
                            targets = config.load_data(campaign=False)
                            gtargets.extend(targets)
                        else:
                            raise ImportError(
                                "Format %s not implemented yet for GNSS data."
                                % options.geodetic_format
                            )

                    else:
                        raise TypeError(
                            'Geodetic datatype "%s" is not supported! '
                            "Supported types are: %s "
                            % (typ, list2string(supported_geodetic_types))
                        )
                if len(gtargets) > 0:
                    logger.info("Pickleing geodetic data to %s" % geodetic_outpath)
                    utility.dump_objects(geodetic_outpath, outlist=gtargets)
                else:
                    raise ImportError(
                        "Data import failed-found no data! " "Please check filepaths!"
                    )
            else:
                logger.info("%s exists! Use --force to overwrite!" % geodetic_outpath)

    else:  # import results
        from pandas import read_csv

        logger.info(
            "Attempting to load results with mode %s to config_%s.yaml"
            " from directory: %s"
            % (options.import_from_mode, options.mode, options.results)
        )
        c = bconfig.load_config(project_dir, options.mode)

        _, ending = os.path.splitext(options.results)

        if not ending:
            # load results from mode optimization
            problem = load_model(
                options.results, options.import_from_mode, hypers=False, build=False
            )
            priors = set(list(problem.config.problem_config.priors.keys()))
            rvs = set(problem.varnames)
            source_params = list(priors.intersection(rvs))
            logger.info(
                "Importing priors for variables:" " %s" % list2string(source_params)
            )

            stage = Stage(
                homepath=problem.outfolder,
                backend=problem.config.sampler_config.backend,
            )
            stage.load_results(
                varnames=problem.varnames,
                model=problem.model,
                stage_number=-1,
                load="trace",
                chains=[-1],
            )

            point = plotting.get_result_point(stage.mtrace, "max")
            summarydf = read_csv(pjoin(problem.outfolder, "summary.txt"), sep="\s+")

        else:
            # load kite model
            from kite import sandbox_scene

            kite_model = load(filename=options.results)
            n_sources = len(kite_model.sources)

            reference_sources = bconfig.init_reference_sources(
                kite_model.sources,
                n_sources,
                c.problem_config.source_type,
                c.problem_config.stf_type,
                event=c.event,
            )

            if "geodetic" in options.datatypes:
                c.geodetic_config.gf_config.reference_sources = reference_sources
            if "seismic" in options.datatypes:
                c.seismic_config.gf_config.reference_sources = reference_sources

            bconfig.dump_config(c)
            logger.info("Successfully imported kite model to ffi source geometry!")
            sys.exit(1)

        # import geodetic hierarchicals
        logger.info(
            "Importing hierarchicals for "
            "datatypes: %s " % list2string(options.datatypes)
        )
        logger.info("---------------------------------------------\n")
        if "geodetic" in options.datatypes:
            logger.info("Geodetic datatype listed-importing ...")
            gc = problem.composites["geodetic"]
            if c.geodetic_config.corrections_config.has_enabled_corrections:

                logger.info("Importing correction parameters ...")
                new_bounds = OrderedDict()

                for var in c.geodetic_config.get_hierarchical_names(
                    datasets=gc.datasets
                ):
                    if var in point:
                        logger.info("Importing correction for %s" % var)
                        new_bounds[var] = (point[var], point[var])
                    else:
                        logger.warning(
                            "Correction %s was fixed in previous run!"
                            " Importing fixed values!" % var
                        )
                        tpoint = problem.config.problem_config.get_test_point()
                        new_bounds[var] = (tpoint[var], tpoint[var])

                c.problem_config.set_vars(new_bounds, attribute="hierarchicals")
            else:
                logger.info("No geodetic corrections enabled, nothing to import!")
        else:
            logger.info("geodetic datatype not listed-not importing ...")

        if "seismic" in options.datatypes:
            logger.info("seismic datatype listed-importing ...")
            sc = problem.composites["seismic"]
            if c.seismic_config.station_corrections:
                logger.info("Importing station corrections ...")
                new_bounds = OrderedDict()
                for wmap in sc.wavemaps:
                    param = wmap.time_shifts_id

                    new_bounds[param] = extract_bounds_from_summary(
                        summarydf, varname=param, shape=(wmap.hypersize,), roundto=0
                    )
                    new_bounds[param].append(point[param])

                c.problem_config.set_vars(
                    new_bounds, attribute="hierarchicals", init=True
                )

            else:
                logger.info("No station_corrections enabled, nothing to import.")
        else:
            logger.info("seismic datatype not listed-not importing ...")

        if options.mode == ffi_mode_str:
            n_sources = problem.config.problem_config.n_sources
            if options.import_from_mode == geometry_mode_str:
                logger.info("Importing non-linear source geometry results!")

                for param in list(point.keys()):
                    if param not in source_params:
                        point.pop(param)

                point = utility.adjust_point_units(point)
                source_points = utility.split_point(point)

                reference_sources = bconfig.init_reference_sources(
                    source_points,
                    n_sources,
                    c.problem_config.source_type,
                    c.problem_config.stf_type,
                    event=c.event,
                )

                if "geodetic" in options.datatypes:
                    c.geodetic_config.gf_config.reference_sources = reference_sources
                if "seismic" in options.datatypes:
                    c.seismic_config.gf_config.reference_sources = reference_sources

                if "seismic" in problem.config.problem_config.datatypes:

                    new_bounds = {}
                    for param in ["time"]:
                        new_bounds[param] = extract_bounds_from_summary(
                            summarydf, varname=param, shape=(n_sources,), roundto=0
                        )
                        new_bounds[param].append(point[param])

                    c.problem_config.set_vars(new_bounds, attribute="priors")

            elif options.import_from_mode == ffi_mode_str:
                n_patches = problem.config.problem_config.mode_config.npatches
                logger.info("Importing distributed slip results!")

                new_bounds = {}
                for param in source_params:
                    if param in dist_vars:
                        shape = (n_patches,)
                    else:
                        shape = (n_sources,)

                    new_bounds[param] = extract_bounds_from_summary(
                        summarydf, varname=param, shape=shape, roundto=1
                    )
                    new_bounds[param].append(point[param])

                c.problem_config.set_vars(new_bounds, attribute="priors")

        elif options.mode == geometry_mode_str:
            if options.import_from_mode == geometry_mode_str:
                n_sources = problem.config.problem_config.n_sources
                logger.info("Importing non-linear source geometry results!")

                new_source_params = set(list(c.problem_config.priors.keys()))
                old_source_params = set(source_params)

                common_source_params = list(
                    new_source_params.intersection(old_source_params)
                )

                new_bounds = {}
                for param in common_source_params:
                    try:
                        new_bounds[param] = extract_bounds_from_summary(
                            summarydf, varname=param, shape=(n_sources,), roundto=0
                        )
                        new_bounds[param].append(point[param])
                    except KeyError:
                        logger.info(
                            "Parameter {} was fixed, not importing " "...".format(param)
                        )

                c.problem_config.set_vars(new_bounds, attribute="priors")

            elif options.import_from_mode == ffi_mode_str:
                err_str = "Cannot import results from %s mode to %s mode!" % (
                    options.import_from_mode,
                    options.mode,
                )
                logger.error(err_str)
                raise TypeError(err_str)

        bconfig.dump_config(c)
        logger.info("Successfully updated config file!")


def command_update(args):

    command_str = "update"

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--parameters",
            default=["structure"],
            type="string",
            action="callback",
            callback=list_callback,
            help='Parameters to update; "structure, hypers, hierarchicals". '
            'Default: ["structure"] (config file-structure only)',
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--diff",
            dest="diff",
            action="store_true",
            help="create diff between normalized old and new versions",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    config_file_name = "config_" + options.mode + ".yaml"

    config_fn = os.path.join(project_dir, config_file_name)

    from beat import upgrade

    upgrade.upgrade_config_file(config_fn, diff=options.diff, update=options.parameters)


def command_clone(args):

    command_str = "clone"

    from beat.config import _datatype_choices as datatype_choices

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--datatypes",
            default=["geodetic", "seismic"],
            type="string",
            action="callback",
            callback=list_callback,
            help='Datatypes to clone; "geodetic, seismic".',
        )

        parser.add_option(
            "--source_type",
            dest="source_type",
            choices=bconfig.source_names,
            default=None,
            help="Source type to replace in config; %s"
            '. Default: "dont change"'
            % ('", "'.join(name for name in bconfig.source_names)),
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--copy_data",
            dest="copy_data",
            action="store_true",
            help="If set, the imported data will be copied into the cloned"
            " directory.",
        )

        parser.add_option(
            "--sampler",
            dest="sampler",
            choices=supported_samplers,
            default=None,
            help="Replace sampling algorithm in config to sample "
            "the solution space of the general problem; %s."
            ' Default: "dont change"' % list2string(supported_samplers),
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    if not len(args) == 2:
        parser.print_help()
        sys.exit(1)

    name, cloned_name = args

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    cloned_dir = pjoin(os.path.dirname(project_dir), cloned_name)

    util.ensuredir(cloned_dir)

    mode = options.mode
    config_fn = pjoin(project_dir, "config_" + mode + ".yaml")

    if os.path.exists(config_fn):
        logger.info("Cloning %s problem config." % mode)
        c = bconfig.load_config(project_dir, mode)

        c.name = cloned_name
        c.project_dir = cloned_dir

        new_datatypes = []
        for datatype in datatype_choices:
            if datatype in options.datatypes:
                if datatype not in c.problem_config.datatypes:
                    logger.warn(
                        "Datatype %s to be cloned is not"
                        " in config! Adding to new config!" % datatype
                    )
                    c[datatype + "_config"] = bconfig.datatype_catalog[datatype](
                        mode=options.mode
                    )
                    re_init = True
                else:
                    re_init = False

                new_datatypes.append(datatype)

                if datatype != "polarity":
                    files = [datatype + "_data.pkl"]
                else:
                    marker_files = [
                        basename(fname) for fname in glob(pjoin(project_dir, "*.pf"))
                    ]
                    files = ["stations.txt"] + marker_files

                for file in files:
                    data_path = pjoin(project_dir, file)
                    if os.path.exists(data_path) and options.copy_data:
                        logger.info("Cloning data ... %s " % data_path)
                        cloned_data_path = pjoin(cloned_dir, file)
                        logger.info("Cloned data path: %s", cloned_data_path)
                        shutil.copyfile(data_path, cloned_data_path)
            else:
                if datatype in c.problem_config.datatypes:
                    logger.warning(
                        'Removing datatype "%s" ' "from cloned config!" % datatype
                    )
                    c[datatype + "_config"] = None

        c.problem_config.datatypes = new_datatypes

        if options.copy_data and mode == ffi_mode_str:
            ffi_dir_name = pjoin(project_dir, mode)
            linear_gf_dir_name = pjoin(ffi_dir_name, bconfig.linear_gf_dir_name)
            cloned_ffi_dir_name = pjoin(cloned_dir, mode)
            util.ensuredir(cloned_ffi_dir_name)
            if os.path.exists(linear_gf_dir_name):
                logger.info("FFI mode - attempting to clone linear GF libraries ...")
                cloned_linear_gf_dir_name = pjoin(
                    cloned_ffi_dir_name, bconfig.linear_gf_dir_name
                )
                if os.path.exists(cloned_linear_gf_dir_name):
                    logger.warning("Linear GFs exist already! Removing ...")
                    shutil.rmtree(cloned_linear_gf_dir_name)

                shutil.copytree(linear_gf_dir_name, cloned_linear_gf_dir_name)
                logger.info("Successfully cloned linear GF libraries.")

        if options.source_type is None:
            old_priors = copy.deepcopy(c.problem_config.priors)

            new_priors = c.problem_config.select_variables()
            for prior in new_priors:
                if prior in list(old_priors.keys()):
                    c.problem_config.priors[prior] = old_priors[prior]

        else:
            logger.info('Replacing source with "%s"' % options.source_type)
            c.problem_config.source_type = options.source_type
            c.problem_config.init_vars()
            c.problem_config.set_decimation_factor()
            re_init = False

        if re_init:
            logger.info(
                "Re-initialized priors because of new datatype!"
                " Please check prior bounds!"
            )
            c.problem_config.init_vars()

        old_hypers = copy.deepcopy(c.problem_config.hyperparameters)

        c.update_hypers()
        for hyper in old_hypers.keys():
            c.problem_config.hyperparameters[hyper] = old_hypers[hyper]

        if options.sampler:
            c.sampler_config = bconfig.SamplerConfig(name=options.sampler)

        c.regularize()
        c.validate()
        bconfig.dump_config(c)

    else:
        raise IOError("Config file: %s does not exist!" % config_fn)


def command_sample(args):

    command_str = "sample"

    def setup(parser):
        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--hypers",
            dest="hypers",
            action="store_true",
            help="Sample hyperparameters only.",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    problem = load_model(project_dir, options.mode, options.hypers)

    step = problem.init_sampler(hypers=options.hypers)

    if options.hypers:
        estimate_hypers(step, problem)
    else:
        sample(step, problem)


def result_check(mtrace, min_length):
    if len(mtrace.chains) < min_length:
        raise IOError("Result traces do not exist. Previously deleted?")


def command_summarize(args):

    from numpy import hstack, ravel, split, vstack
    from pymc3 import summary
    from pyrocko.gf import RectangularSource
    from pyrocko.moment_tensor import MomentTensor

    command_str = "summarize"

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="Overwrite existing files",
        )

        parser.add_option(
            "--calc_derived",
            dest="calc_derived",
            action="store_true",
            help="Calculate derived variables (e.g. alternative MT params)",
        )

        parser.add_option(
            "--stage_number",
            dest="stage_number",
            type="int",
            default=None,
            help='Int of the stage number "n" of the stage to be summarized.'
            " Default: all stages up to last complete stage",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    logger.info("Loading problem ...")
    problem = load_model(project_dir, options.mode)
    problem.plant_lijection()

    stage = Stage(
        homepath=problem.outfolder, backend=problem.config.sampler_config.backend
    )
    stage_numbers = stage.handler.get_stage_indexes(options.stage_number)
    logger.info("Summarizing stage(s): %s" % list2string(stage_numbers))
    if len(stage_numbers) == 0:
        raise ValueError("No stage result found where sampling completed!")

    sc = problem.config.sampler_config
    sc_params = problem.config.sampler_config.parameters
    sampler_name = problem.config.sampler_config.name
    if hasattr(sc_params, "rm_flag"):
        if sc_params.rm_flag:
            logger.info("Removing sampled chains!!!")
            input("Sure? Press enter! Otherwise Ctrl + C")
            rm_flag = True
        else:
            rm_flag = False
    else:
        rm_flag = False

    for stage_number in stage_numbers:

        stage_path = stage.handler.stage_path(stage_number)
        logger.info("Summarizing stage under: %s" % stage_path)

        trace_name = "chain--1.{}".format(problem.config.sampler_config.backend)
        result_trace_path = pjoin(stage_path, trace_name)
        if not os.path.exists(result_trace_path) or options.force:
            # trace may exist by forceing
            if os.path.exists(result_trace_path):
                os.remove(result_trace_path)

            stage.load_results(
                model=problem.model, stage_number=stage_number, load="trace"
            )

            if sampler_name == "SMC":
                result_check(stage.mtrace, min_length=2)
                if stage_number == -1:
                    # final stage factor
                    n_steps = sc_params.n_steps * sample_factor_final_stage
                    idxs = range(
                        len(
                            thin_buffer(
                                list(range(n_steps)),
                                sc.buffer_thinning,
                                ensure_last=True,
                            )
                        )
                    )
                else:
                    idxs = [-1]

                draws = sc_params.n_chains * len(idxs)
                chains = stage.mtrace.chains
            elif sampler_name == "PT":
                result_check(stage.mtrace, min_length=1)
                unthinned_idxs = list(
                    range(
                        int(floor(sc_params.n_samples * sc_params.burn)),
                        sc_params.n_samples,
                        sc_params.thin,
                    )
                )
                idxs = range(
                    len(
                        thin_buffer(
                            unthinned_idxs, sc.buffer_thinning, ensure_last=True
                        )
                    )
                )

                chains = [0]
                draws = len(idxs)
            elif sampler_name == "Metropolis":
                result_check(stage.mtrace, min_length=1)

                unburned_idxs = thin_buffer(
                    list(range(sc_params.n_steps)), sc.buffer_thinning, ensure_last=True
                )

                n_ubidxs = len(unburned_idxs)
                idxs = range(
                    len(
                        unburned_idxs[
                            int(floor(n_ubidxs * sc_params.burn)) :: sc_params.thin
                        ]
                    )
                )

                chains = stage.mtrace.chains
                draws = sc_params.n_chains * len(idxs)
            else:
                raise NotImplementedError(
                    "Summarize function still needs to be implemented "
                    "for %s sampler" % problem.config.sampler_config.name
                )

            rtrace = backend_catalog[sc.backend](
                stage_path,
                model=problem.model,
                buffer_size=sc.buffer_size,
                progressbar=False,
            )

            pc = problem.config.problem_config
            reference = pc.get_test_point()

            if options.calc_derived:
                varnames, shapes = pc.get_derived_variables_shapes()
                rtrace.add_derived_variables(varnames, shapes)
                splitinds = range(1, len(varnames))

            rtrace.setup(draws=draws, chain=-1, overwrite=True)

            if "seismic" in problem.config.problem_config.datatypes:
                composite = problem.composites["seismic"]
            elif "polarity" in problem.config.problem_config.datatypes:
                composite = problem.composites["polarity"]
            else:
                composite = problem.composites["geodetic"]

            target = composite.targets[0]
            if hasattr(composite, "sources"):
                source = problem.sources[0]
                sources = composite.sources
                store = composite.engine.get_store(target.store_id)
            else:
                source = composite.load_fault_geometry()
                sources = [source]
                engine = LocalEngine(
                    store_superdirs=[composite.config.gf_config.store_superdir]
                )
                store = engine.get_store(target.store_id)

            for chain in tqdm(chains):
                for idx in idxs:
                    point = stage.mtrace.point(idx=idx, chain=chain)
                    reference.update(point)
                    # normalize MT source, TODO put into get_derived_params
                    if isinstance(source, MTSourceWithMagnitude):
                        composite.point2sources(point)
                        ldicts = []
                        for source in sources:
                            ldicts.append(source.scaled_m6_dict)

                        jpoint = utility.join_points(ldicts)
                        reference.update(jpoint)
                        del jpoint, ldicts

                    derived = []
                    # BEAT sources calculate derived params
                    if options.calc_derived:
                        composite.point2sources(point)
                        if hasattr(source, "get_derived_parameters"):
                            for source in sources:
                                deri = source.get_derived_parameters(
                                    point=reference,  # need to pass correction params
                                    store=store,
                                    target=target,
                                    event=problem.config.event,
                                )
                                derived.append(deri)

                        # pyrocko Rectangular source, TODO use BEAT RS ...
                        elif isinstance(source, RectangularSource):
                            for source in sources:
                                source.magnitude = None
                                derived.append(
                                    source.get_magnitude(store=store, target=target)
                                )

                    lpoint = problem.model.lijection.d2l(point)

                    if derived:
                        lpoint.extend(
                            map(ravel, split(vstack(derived).T, splitinds, axis=0))
                        )

                    # TODO: in PT with large buffer sizes somehow memory leak
                    rtrace.write(lpoint, draw=chain)
                    del lpoint, point

                if rm_flag:
                    # remove chain
                    logger.info("Removing sampled traces ...")
                    os.remove(stage.mtrace._straces[chain].filename)

            rtrace.record_buffer()
        else:
            logger.info("Summarized trace exists! Use force=True to overwrite!")

    final_stage = -1
    if final_stage in stage_numbers:
        stage.load_results(model=problem.model, stage_number=final_stage, chains=[-1])
        rtrace = stage.mtrace

        if len(rtrace) == 0:
            raise ValueError(
                "Trace collection previously failed. Please rerun"
                ' "beat summarize <project_dir> --force!"'
            )

        summary_file = pjoin(problem.outfolder, bconfig.summary_name)

        if os.path.exists(summary_file) and options.force:
            os.remove(summary_file)

        if not os.path.exists(summary_file) or options.force:
            logger.info("Writing summary to %s" % summary_file)
            df = summary(rtrace, alpha=0.01)
            with open(summary_file, "w") as outfile:
                df.to_string(outfile)
        else:
            logger.info("Summary exists! Use force=True to overwrite!")


def command_build_gfs(args):

    command_str = "build_gfs"

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--datatypes",
            default=["geodetic"],
            type="string",
            action="callback",
            callback=list_callback,
            help='Datatypes to calculate the GFs for; "geodetic, seismic".'
            ' Default: "geodetic"',
        )

        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="Overwrite existing files",
        )

        parser.add_option(
            "--execute",
            dest="execute",
            action="store_true",
            help="Start actual GF calculations. If not set only"
            " configuration files are being created",
        )

        parser.add_option(
            "--plot",
            dest="plot",
            action="store_true",
            help="Plot fault discretization after fault patch"
            " discretization optimization.",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    c = bconfig.load_config(project_dir, options.mode)
    if options.mode in [geometry_mode_str, "interseismic"]:
        for datatype in options.datatypes:
            if datatype == "geodetic":
                gc = c.geodetic_config
                gf = c.geodetic_config.gf_config

                for crust_ind in range(*gf.n_variations):
                    heart.geo_construct_gf(
                        event=c.event,
                        geodetic_config=gc,
                        crust_ind=crust_ind,
                        execute=options.execute,
                        force=options.force,
                    )

            elif datatype == "seismic":
                sc = c.seismic_config
                sf = sc.gf_config

                if sf.reference_location is None:
                    logger.info(
                        "Creating Green's Function stores individually"
                        " for each station!"
                    )
                    seismic_data_path = pjoin(c.project_dir, bconfig.seismic_data_name)

                    stations, _ = utility.load_objects(seismic_data_path)
                    logger.info(
                        "Found stations %s"
                        % list2string([station.station for station in stations])
                    )

                    blacklist = sc.get_station_blacklist()
                    stations = utility.apply_station_blacklist(
                        stations, blacklist=blacklist
                    )
                    logger.info("Blacklisted stations: %s" % list2string(blacklist))

                else:
                    logger.info(
                        "Creating one global Green's Function store, which is "
                        "being used by all stations!"
                    )
                    stations = [sf.reference_location]
                    logger.info("Store name: %s" % sf.reference_location.station)

                for crust_ind in range(*sf.n_variations):
                    heart.seis_construct_gf(
                        stations=stations,
                        event=c.event,
                        seismic_config=sc,
                        crust_ind=crust_ind,
                        execute=options.execute,
                        force=options.force,
                    )

            elif datatype == "polarity":
                from pyrocko.model import load_stations

                polc = c.polarity_config
                polcf = polc.gf_config

                if polcf.reference_location is None:
                    logger.info(
                        "Creating Green's Function stores individually"
                        " for each station!"
                    )
                    polarity_stations_path = pjoin(c.project_dir, bconfig.stations_name)

                    stations = load_stations(polarity_stations_path)
                    logger.info(
                        "Found stations %s"
                        % list2string([station.station for station in stations])
                    )

                else:
                    logger.info(
                        "Creating one global Green's Function store, which is "
                        "being used by all stations!"
                    )
                    stations = [polcf.reference_location]
                    logger.info("Store name: %s" % polcf.reference_location.station)

                for crust_ind in range(*polcf.n_variations):
                    heart.polarity_construct_gf(
                        stations=stations,
                        event=c.event,
                        polarity_config=polc,
                        crust_ind=crust_ind,
                        execute=options.execute,
                        force=options.force,
                    )

            else:
                raise ValueError("Datatype %s not supported!" % datatype)

            if not options.execute:
                logger.info(
                    "%s GF store configs successfully created! "
                    "To start calculations set --execute!" % datatype
                )

    elif options.mode == ffi_mode_str:
        from beat import ffi

        varnames = get_sampled_slip_variables(c)
        outdir = pjoin(c.project_dir, options.mode, bconfig.linear_gf_dir_name)
        util.ensuredir(outdir)

        faultpath = pjoin(outdir, bconfig.fault_geometry_name)
        if not os.path.exists(faultpath) or options.force:
            for datatype in options.datatypes:
                try:
                    gf = c[datatype + "_config"].gf_config
                except AttributeError:
                    raise AttributeError(
                        'Datatype "%s" not existing in config!' % datatype
                    )

            if len(c.problem_config.datatypes) > 1:
                logger.warning(
                    "Found two datatypes! Please be aware that the reference"
                    " fault geometries have to be consistent!"
                )

            for source in gf.reference_sources:
                if source.lat == 0 and source.lon == 0:
                    logger.info(
                        "Reference source is configured without Latitude "
                        "and Longitude! Updating with event information! ..."
                    )
                    source.update(lat=c.event.lat, lon=c.event.lon)

            logger.info("Discretizing reference sources ...")
            fault = ffi.discretize_sources(
                config=gf.discretization_config,
                varnames=varnames,
                sources=gf.reference_sources,
                datatypes=c.problem_config.datatypes,
            )
            mode_c = c.problem_config.mode_config

            if gf.discretization == "uniform":
                logger.info("Fault discretization done! Updating problem_config...")
                logger.info("%s" % fault.__str__())

                c.problem_config.n_sources = fault.nsubfaults
                mode_c.npatches = fault.npatches
                mode_c.subfault_npatches = fault.subfault_npatches

                nucleation_strikes = []
                nucleation_dips = []
                for i in range(fault.nsubfaults):
                    ext_source = fault.get_subfault(i, datatype=options.datatypes[0])

                    nucleation_dips.append(ext_source.width / km)
                    nucleation_strikes.append(ext_source.length / km)

                nucl_start = zeros(fault.nsubfaults)
                new_bounds = {
                    "nucleation_strike": (nucl_start, array(nucleation_strikes)),
                    "nucleation_dip": (nucl_start, array(nucleation_dips)),
                }

                c.problem_config.set_vars(new_bounds)
                bconfig.dump_config(c)
                logger.info("Storing discretized fault geometry to: %s" % faultpath)
                utility.dump_objects(faultpath, [fault])
            else:
                logger.info(
                    "For resolution based discretization GF calculation "
                    "has to be started!"
                )

        elif os.path.exists(faultpath):
            logger.info(
                "Discretized fault geometry exists! Use --force to" " overwrite!"
            )
            logger.info("Loading existing discretized fault")
            fault = utility.load_objects(faultpath)[0]

        if options.execute:
            logger.info("Calculating linear Green's Functions")
            logger.info("------------------------------------\n")
            logger.info("For slip components: %s" % list2string(varnames))

            for datatype in options.datatypes:
                logger.info("for %s data ..." % datatype)

                if datatype == "geodetic":
                    gf = c.geodetic_config.gf_config
                    logger.info("using %i workers ..." % gf.nworkers)
                    geodetic_data_path = pjoin(
                        c.project_dir, bconfig.geodetic_data_name
                    )

                    datasets = utility.load_objects(geodetic_data_path)

                    engine = LocalEngine(store_superdirs=[gf.store_superdir])

                    for crust_ind in range(*gf.n_variations):
                        logger.info("crust_ind %i" % crust_ind)

                        targets = heart.init_geodetic_targets(
                            datasets,
                            earth_model_name=gf.earth_model_name,
                            interpolation=c.geodetic_config.interpolation,
                            crust_inds=[crust_ind],
                            sample_rate=gf.sample_rate,
                        )

                        if not fault.is_discretized and fault.needs_optimization:

                            ffidir = os.path.join(c.project_dir, options.mode)

                            if options.plot:
                                figuredir = os.path.join(ffidir, "figures")
                                util.ensuredir(figuredir)
                            else:
                                figuredir = None

                            fault = ffi.optimize_damping(
                                outdir=ffidir,
                                figuredir=figuredir,
                                config=gf.discretization_config,
                                fault=fault,
                                datasets=datasets,
                                varnames=varnames,
                                engine=engine,
                                crust_ind=crust_ind,
                                targets=targets,
                                event=c.event,
                                force=options.force,
                                nworkers=gf.nworkers,
                            )

                            logger.info(
                                "Storing optimized discretized fault"
                                " geometry to: %s" % faultpath
                            )
                            utility.dump_objects(faultpath, [fault])
                            logger.info(
                                "Fault discretization optimization done! "
                                "Updating problem_config..."
                            )
                            logger.info("%s" % fault.__str__())
                            mode_c.npatches = fault.npatches
                            mode_c.subfault_npatches = fault.subfault_npatches
                            bconfig.dump_config(c)

                        ffi.geo_construct_gf_linear(
                            engine=engine,
                            outdirectory=outdir,
                            event=c.event,
                            crust_ind=crust_ind,
                            datasets=datasets,
                            targets=targets,
                            nworkers=gf.nworkers,
                            fault=fault,
                            varnames=varnames,
                            force=options.force,
                        )

                elif datatype == "seismic":

                    sc = c.seismic_config
                    gf = sc.gf_config
                    pc = c.problem_config
                    logger.info("using %i workers ..." % gf.nworkers)

                    seismic_data_path = pjoin(c.project_dir, bconfig.seismic_data_name)
                    datahandler = heart.init_datahandler(
                        seismic_config=sc, seismic_data_path=seismic_data_path
                    )

                    if "time_shift" in pc.hierarchicals:
                        time_shift = pc.hierarchicals["time_shift"]
                    else:
                        time_shift = None

                    engine = LocalEngine(store_superdirs=[gf.store_superdir])

                    for crust_ind in range(*gf.n_variations):
                        logger.info("crust_ind %i" % crust_ind)
                        sc.gf_config.reference_model_idx = crust_ind

                        for i, wc in enumerate(sc.waveforms):
                            wmap = heart.init_wavemap(
                                waveformfit_config=wc,
                                datahandler=datahandler,
                                event=c.event,
                                mapnumber=i,
                            )

                            ffi.seis_construct_gf_linear(
                                engine=engine,
                                fault=fault,
                                durations_prior=pc.priors["durations"],
                                velocities_prior=pc.priors["velocities"],
                                nucleation_time_prior=pc.priors["time"],
                                varnames=varnames,
                                wavemap=wmap,
                                event=c.event,
                                time_shift=time_shift,
                                nworkers=gf.nworkers,
                                starttime_sampling=gf.starttime_sampling,
                                duration_sampling=gf.duration_sampling,
                                sample_rate=gf.sample_rate,
                                outdirectory=outdir,
                                force=options.force,
                            )
        else:
            logger.info("Did not run GF calculation. Use --execute!")


def command_plot(args):

    command_str = "plot"

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--post_llk",
            dest="post_llk",
            choices=["max", "min", "mean", "all", "None", "test"],
            default="max",
            help='Plot model with specified likelihood; "max", "min", "mean"'
            ' "None" or "all"; Default: "max"',
        )

        parser.add_option(
            "--stage_number",
            dest="stage_number",
            type="int",
            default=None,
            help='Int of the stage number "n" of the stage to be plotted.'
            " Default: all stages up to last complete stage",
        )

        parser.add_option(
            "--varnames",
            default="",
            type="string",
            action="callback",
            callback=list_callback,
            help='Variable names to plot in figures. Example: "strike,dip"'
            " Default: empty string --> all",
        )

        parser.add_option(
            "--nensemble",
            dest="nensemble",
            type="int",
            default=1,
            help="Int of the number of solutions that" " are used for fuzzy plots",
        )

        parser.add_option(
            "--source_idxs",
            action="callback",
            callback=list_callback,
            type="string",
            default="",
            help="Indexes to patches" " of slip distribution to draw marginals for",
        )

        parser.add_option(
            "--format",
            dest="format",
            choices=["display", "pdf", "png", "svg", "eps"],
            default="pdf",
            help='Output format of the plot; "display", "pdf", "png", "svg",'
            '"eps" Default: "pdf"',
        )

        parser.add_option(
            "--plot_projection",
            dest="plot_projection",
            # choices=['latlon', 'local', 'individual'],
            default="local",
            help='Output projection of the plot; "latlon" or "local" for maps - Default: "local";'
            ' "pdf", "cdf" or "kde" for stage_posterior plot - Default: "pdf"',
        )

        parser.add_option(
            "--dpi",
            dest="dpi",
            type="int",
            default=300,
            help="Output resolution of the plots in dpi (dots per inch);"
            ' Default: "300"',
        )

        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="Overwrite existing files",
        )

        parser.add_option(
            "--reference",
            dest="reference",
            action="store_true",
            help="Plot reference (test_point) into stage posteriors.",
        )

        parser.add_option(
            "--hypers",
            dest="hypers",
            action="store_true",
            help="Plot hyperparameter results only.",
        )

        parser.add_option(
            "--build",
            dest="build",
            action="store_true",
            help="Build models during problem loading.",
        )

    import traceback

    plots_avail = plotting.available_plots()

    details = """Available <plot types> are: %s or "all". Multiple plots can be
selected giving a comma separated list.""" % list2string(
        plots_avail
    )

    parser, options, args = cl_parse(command_str, args, setup, details)

    if len(args) < 1:
        parser.error("plot needs at least one argument!")
        parser.help()

    project_dir = get_project_directory(
        args, options, nargs_dict[command_str], popflag=True
    )

    if args[0] == "all":
        c = bconfig.load_config(project_dir, options.mode)
        plotnames = plotting.available_plots(
            options.mode, datatypes=c.problem_config.datatypes
        )
        logger.info('Plotting "all" plots: %s' % list2string(plotnames))
    else:
        plotnames = args[0].split(",")

    for plot in plotnames:
        if plot not in plots_avail:
            raise TypeError(
                "Plot type %s not available! Available plots are:"
                " %s" % (plot, plots_avail)
            )

    logger.info("Loading problem ...")

    problem = load_model(project_dir, options.mode, options.hypers, options.build)

    if len(options.source_idxs) < 1:
        source_idxs = None
    else:
        source_idxs = []
        for str_idx in options.source_idxs:
            try:
                idx = int(str_idx)
            except ValueError:
                idx = string2slice(str_idx)

            source_idxs.append(idx)

    po = plotting.PlotOptions(
        plot_projection=options.plot_projection,
        post_llk=options.post_llk,
        load_stage=options.stage_number,
        outformat=options.format,
        force=options.force,
        dpi=options.dpi,
        varnames=options.varnames,
        nensemble=options.nensemble,
        source_idxs=source_idxs,
    )

    if options.reference:
        if po.post_llk == "test":
            results_path = pjoin(problem.outfolder, bconfig.results_dir_name)
            results_test_point = pjoin(results_path, "solution_test.yaml")
            if os.path.exists(results_test_point):
                logger.info(
                    "Result test point exists at %s !"
                    " Forward modeling ..." % results_test_point
                )
                po.reference = load(filename=results_test_point).point
            else:
                raise ValueError(
                    "Results test point does not exist" " at: %s" % results_test_point
                )
        else:
            try:
                po.reference = problem.model.test_point
                step = problem.init_sampler()
                po.reference["like"] = step.step(problem.model.test_point)[1][-1]
            except AttributeError:
                po.reference = problem.config.problem_config.get_test_point()
    else:
        if po.post_llk == "test":
            raise ValueError('"post_llk=test" only works together with --reference!')
        po.reference = {}

    figure_path = pjoin(problem.outfolder, po.figure_dir)
    util.ensuredir(figure_path)

    for plot in plotnames:
        try:
            plotting.plots_catalog[plot](problem, po)
        # except Exception as err:
        #    pass
        except (TypeError, plotting.ModeError) as err:

            logger.warning(
                "Could not plot %s got Error: %s \n %s"
                % (plot, err, traceback.format_exc())
            )


def command_check(args):

    command_str = "check"
    what_choices = ["stores", "traces", "library", "geometry", "discretization"]

    def setup(parser):
        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--datatypes",
            default=["seismic"],
            type="string",
            action="callback",
            callback=list_callback,
            help='Datatypes to check; "geodetic, seismic".',
        )

        parser.add_option(
            "--what",
            dest="what",
            choices=what_choices,
            default="stores",
            help="Setup item to check; "
            '"%s", Default: "stores"' % list2string(what_choices),
        )

        parser.add_option(
            "--targets",
            default=[0],
            type="string",
            action="callback",
            callback=list_callback,
            help="Indexes to targets/datasets to display.",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    problem = load_model(project_dir, options.mode, hypers=False, build=False)

    tpoint = problem.config.problem_config.get_test_point()
    if options.mode == geometry_mode_str:
        problem.point2sources(tpoint)

    if options.what == "stores":
        corrupted_stores = heart.check_problem_stores(problem, options.datatypes)

        for datatype in options.datatypes:
            store_ids = corrupted_stores[datatype]
            if len(store_ids) > 0:
                logger.warning(
                    "Store(s) with empty traces! : %s for %s datatype!"
                    % (list2string(store_ids), datatype)
                )
            else:
                logger.info("All stores ok for %s datatype!" % datatype)

    elif options.what == "traces":
        sc = problem.composites["seismic"]
        for wmap in sc.wavemaps:
            event = sc.events[wmap.config.event_idx]
            wmap.prepare_data(source=event, engine=sc.engine, outmode="stacked_traces")

            # set location code for setup checking
            for tr in wmap._prepared_data:
                tr.set_location("f")

            snuffle(
                wmap.datasets + wmap._prepared_data,
                stations=wmap.stations,
                events=[event],
            )

    elif options.what == "library":
        if options.mode != ffi_mode_str:
            logger.warning(
                'GF library exists only for "%s" ' "optimization mode." % ffi_mode_str
            )
        else:
            from beat import ffi

            for datatype in options.datatypes:
                for var in get_sampled_slip_variables(problem.config):

                    outdir = pjoin(
                        problem.config.project_dir,
                        options.mode,
                        bconfig.linear_gf_dir_name,
                    )
                    if datatype == "seismic":
                        sc = problem.config.seismic_config
                        scomp = problem.composites["seismic"]

                        for wmap in scomp.wavemaps:
                            filename = ffi.get_gf_prefix(
                                datatype,
                                component=var,
                                wavename=wmap._mapid,
                                crust_ind=sc.gf_config.reference_model_idx,
                            )

                            logger.info(
                                "Loading Greens Functions"
                                " Library %s for %s target"
                                % (filename, list2string(options.targets))
                            )
                            gfs = ffi.load_gf_library(
                                directory=outdir, filename=filename
                            )

                            targets = [int(target) for target in options.targets]
                            trs = gfs.get_traces(
                                targetidxs=targets,
                                patchidxs=list(range(gfs.npatches)),
                                durationidxs=list(range(gfs.ndurations)),
                                starttimeidxs=list(range(gfs.nstarttimes)),
                            )
                            snuffle(trs)

    elif options.what == "discretization":
        from beat.plotting import source_geometry

        if "geodetic" in problem.config.problem_config.datatypes:
            datatype = "geodetic"
            datasets = []
            for i in options.targets:
                dataset = problem.composites[datatype].datasets[int(i)]
                dataset.update_local_coords(problem.config.event)
                datasets.append(dataset)
        else:
            datatype = problem.config.problem_config.datatypes[0]
            datasets = None

        if options.mode == ffi_mode_str:
            from numpy import diag

            fault = problem.composites[datatype].load_fault_geometry()
            reference_sources = problem.config[
                datatype + "_config"
            ].gf_config.reference_sources
            try:
                values = diag(fault.get_model_resolution())
            except ValueError:
                values = None

            source_geometry(
                fault,
                reference_sources,
                values=values,
                event=problem.config.event,
                datasets=datasets,
            )
        else:
            logger.warning(
                "Checking discretization is only for"
                ' "%s" mode available' % ffi_mode_str
            )

    elif options.what == "geometry":
        from pyrocko import orthodrome as otd

        if "geodetic" not in problem.config.problem_config.datatypes:
            raise ValueError("Checking geometry is only available for geodetic data")

        try:
            from kite import SandboxScene
            from kite import sources as ksources
            from kite.scene import Scene, UserIOWarning
            from kite.talpa import Talpa

            talpa_source_catalog = {
                "RectangularSource": ksources.OkadaSource,
                "DCSource": ksources.PyrockoDoubleCouple,
                "MTSource": ksources.PyrockoMomentTensor,
                "RingfaultSource": ksources.PyrockoRingfaultSource,
            }

        except ImportError:
            raise ImportError(
                "Please install the KITE software (www.pyrocko.org)"
                " to enable this feature!"
            )

        if len(options.targets) > 1:
            logger.warning(
                "Targets can be only of length 1 for geometry checking!"
                " Displaying only first target."
            )

        gc = problem.composites["geodetic"]
        gfc = gc.config.gf_config
        dataset = gc.datasets[int(options.targets[0])]
        logger.info("Initialising Talpa Sandbox ...")
        sandbox = SandboxScene()
        if isinstance(dataset, heart.DiffIFG):
            try:
                homepath = problem.config.geodetic_config.types["SAR"].datadir
                scene_path = os.path.join(homepath, dataset.name)
                logger.info("Loading full resolution kite scene: %s" % scene_path)
                sandbox.loadReferenceScene(scene_path)
            except (UserIOWarning, KeyError):
                raise ImportError("Full resolution data could not be loaded!")
        elif isinstance(dataset, heart.GNSSCompoundComponent):
            logger.info("Found GNSS Compound %s, importing to kite..." % dataset.name)
            scene = dataset.to_kite_scene()
            # scene.spool()
            sandbox.setReferenceScene(scene)
        else:
            raise TypeError("Dataset type %s is not supported!" % dataset.__class__)

        from tempfile import mkdtemp

        tempdir = mkdtemp(prefix="beat_geometry_check", dir=None)

        store_dir = pjoin(
            gfc.store_superdir,
            heart.get_store_id(
                "statics",
                heart.get_earth_model_prefix(gfc.earth_model_name),
                gfc.sample_rate,
                gfc.reference_model_idx,
            ),
        )

        if options.mode == ffi_mode_str:
            logger.info("FFI mode: Loading reference sources ...")
            sources = gc.config.gf_config.reference_sources

        elif options.mode == geometry_mode_str:
            logger.info("Geometry mode: Loading Test value sources ...")
            tpoint = problem.config.problem_config.get_test_point()
            gc.point2sources(tpoint)
            sources = gc.sources

        logger.info("Transforming sources to local cartesian system.")
        for source in sources:
            n, e = otd.latlon_to_ne_numpy(
                lat0=gc.event.lat, lon0=gc.event.lon, lat=source.lat, lon=source.lon
            )
            n += source.north_shift
            e += source.east_shift
            source.update(
                lat=gc.event.lat, lon=gc.event.lon, north_shift=n, east_shift=e
            )

        src_class_name = problem.config.problem_config.source_type
        for source in sources:
            source.regularize()
            try:
                sandbox.addSource(
                    talpa_source_catalog[src_class_name].fromPyrockoSource(
                        source, store_dir=store_dir
                    )
                )
            except (AttributeError, KeyError):
                raise ValueError(
                    "%s not supported for display in Talpa!" "" % src_class_name
                )

        filename = pjoin(tempdir, "%s.yml" % dataset.name)
        sandbox.save(filename)
        logger.debug("Saving sandbox to %s" % filename)
        Talpa(filename)
    else:
        raise ValueError("Subject what: %s is not available!" % options.what)


def command_export(args):

    command_str = "export"

    def setup(parser):

        parser.add_option(
            "--main_path",
            dest="main_path",
            type="string",
            default="./",
            help="Main path (absolute) leading to folders of events that"
            ' have been created by "init".'
            " Default: current directory: ./",
        )

        parser.add_option(
            "--mode",
            dest="mode",
            choices=mode_choices,
            default=geometry_mode_str,
            help='Inversion problem to solve; %s Default: "%s"'
            % (list2string(mode_choices), geometry_mode_str),
        )

        parser.add_option(
            "--stage_number",
            dest="stage_number",
            type="int",
            default=-1,
            help='Int of the stage number "n" of the stage to be summarized.'
            " Default: all stages up to last complete stage",
        )

        parser.add_option(
            "--post_llk",
            dest="post_llk",
            choices=["max", "min", "mean", "all"],
            default="max",
            help='Export model with specified likelihood; "max", "min", "mean"'
            ' or "all"; Default: "max"',
        )

        parser.add_option(
            "--reference",
            dest="reference",
            action="store_true",
            help="Export data for test point instead of result point " "(post_llk)",
        )

        parser.add_option(
            "--fix_output",
            dest="fix_output",
            action="store_true",
            help="Fix Network Station Location Codes in case "
            "they are not compliant with mseed",
        )

        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="Overwrite existing files",
        )

    parser, options, args = cl_parse(command_str, args, setup=setup)

    project_dir = get_project_directory(args, options, nargs_dict[command_str])

    logger.info("Loading problem ...")
    problem = load_model(project_dir, options.mode, hypers=False, build=False)
    problem.init_hierarchicals()

    sc = problem.config.sampler_config

    trace_name = "chain--1.{}".format(problem.config.sampler_config.backend)
    results_path = pjoin(problem.outfolder, bconfig.results_dir_name)
    logger.info("Saving results to %s" % results_path)
    util.ensuredir(results_path)

    point = problem.config.problem_config.get_test_point()
    if options.reference:
        options.post_llk = "ref"
    else:
        stage = Stage(
            homepath=problem.outfolder, backend=problem.config.sampler_config.backend
        )

        stage.load_results(
            varnames=problem.varnames,
            model=problem.model,
            stage_number=options.stage_number,
            load="trace",
            chains=[-1],
        )

        res_point = plotting.get_result_point(stage.mtrace, point_llk=options.post_llk)
        point.update(res_point)

        if options.stage_number == -1:
            results_trace = pjoin(stage.handler.stage_path(-1), trace_name)
            shutil.copy(results_trace, pjoin(results_path, trace_name))

    var_reds = problem.get_variance_reductions(point)

    rpoint = heart.ResultPoint(
        point=point, post_llk=options.post_llk, variance_reductions=var_reds
    )
    rpoint.regularize()
    rpoint.validate()

    outpoint_name = pjoin(results_path, "solution_{}.yaml".format(options.post_llk))
    dump(rpoint, filename=outpoint_name)
    logger.info("Dumped %s solution to %s" % (options.post_llk, outpoint_name))

    if options.mode == ffi_mode_str:
        if "seismic" in problem.config.problem_config.datatypes:
            datatype = "seismic"
        elif "geodetic" in problem.config.problem_config.datatypes:
            datatype = "geodetic"
        else:
            logger.info("Rupture geometry only available for static / kinematic data.")

        comp = problem.composites[datatype]
        engine = LocalEngine(store_superdirs=[comp.config.gf_config.store_superdir])

        target = comp.targets[0]
        fault = comp.load_fault_geometry()

        ffi_rupture_table_path = pjoin(
            results_path, "rupture_evolution_{}.yaml".format(options.post_llk)
        )

        try:
            geom = fault.get_rupture_geometry(
                point=point,
                target=target,
                store=engine.get_store(target.store_id),
                event=problem.event,
                datatype=datatype,
            )
        except ImportError:
            logger.info(
                "Need to install the pyrocko sparrow branch for "
                "export of the rupture geometry in 3d!"
            )
            geom = None

        if geom is not None:
            logger.info(
                "Exporting finite rupture evolution" " to %s" % ffi_rupture_table_path
            )
            dump(geom, filename=ffi_rupture_table_path)

    for datatype, composite in problem.composites.items():
        logger.info(
            "Exporting %s synthetics \n"
            "-----------------------------\n"
            ' for "%s" likelihood parameters:' % (datatype, options.post_llk)
        )
        for varname, value in point.items():
            logger.info("%s: %s" % (varname, list2string(value.ravel().tolist())))

        composite.export(
            point,
            results_path,
            stage_number=options.stage_number,
            fix_output=options.fix_output,
            force=options.force,
        )

        stdzd_res = composite.get_standardized_residuals(point)
        if stdzd_res:
            stdzd_res_path = pjoin(
                results_path, "{}_stdzd_residuals.pkl".format(datatype)
            )
            logger.info("Exporting standardized residuals to %s" % stdzd_res_path)
            utility.dump_objects(stdzd_res_path, outlist=stdzd_res)


def main():

    if len(sys.argv) < 2:
        sys.exit("Usage: %s" % usage)

    args = list(sys.argv)
    args.pop(0)
    command = args.pop(0)

    if command in subcommands:
        globals()["command_" + command](args)

    elif command in ("--help", "-h", "help"):
        if command == "help" and args:
            acommand = args[0]
            if acommand in subcommands:
                globals()["command_" + acommand](["--help"])

        sys.exit("Usage: %s" % usage)

    else:
        sys.exit("BEAT: error: no such subcommand: %s" % command)


if __name__ == "__main__":
    main()
