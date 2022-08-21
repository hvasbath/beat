#!/usr/bin/env pyrocko-python

import logging
import math
import os.path as op
import shutil
import sys
import tempfile

try:
    from urllib.error import HTTPError
except:
    from urllib2 import HTTPError

import glob
import pipes
from collections import defaultdict
from optparse import OptionParser

import numpy as num
from pyrocko import (
    automap,
    cake,
    catalog,
    io,
    model,
    orthodrome,
    pile,
    trace,
    util,
    weeding,
)
from pyrocko.client import fdsn
from pyrocko.io import enhanced_sacpz as epz
from pyrocko.io import resp, stationxml

from beat import heart, utility

km = 1000.0

g_sites_available = sorted(fdsn.g_site_abbr.keys())

geofon = catalog.Geofon()
usgs = catalog.USGS(catalog=None)
gcmt = catalog.GlobalCMT()

tfade_factor = 1.0
ffade_factors = 0.5, 1.5

fdsn.g_timeout = 60.0


class starfill(object):
    def __getitem__(self, k):
        return "*"


def nice_seconds_floor(s):
    nice = [
        1.0,
        10.0,
        60.0,
        600.0,
        3600.0,
        3.0 * 3600.0,
        12 * 3600.0,
        24 * 3600.0,
        48 * 3600.0,
    ]
    p = s
    for x in nice:
        if s < x:
            return p

        p = x

    return s


def get_events(time_range, region=None, catalog=geofon, **kwargs):
    if not region:
        return catalog.get_events(time_range, **kwargs)

    events = []
    for (west, east, south, north) in automap.split_region(region):
        events.extend(
            catalog.get_events(
                time_range=time_range,
                lonmin=west,
                lonmax=east,
                latmin=south,
                latmax=north,
                **kwargs
            )
        )

    return events


def cut_n_dump(traces, win, out_path):
    otraces = []
    for tr in traces:
        try:
            otr = tr.chop(win[0], win[1], inplace=False)
            otraces.append(otr)
        except trace.NoData:
            pass

    return io.save(otraces, out_path)


aliases = {
    "2010_haiti": "2010-01-12 21:53:00",
    "2012_emilia": ("2012-05-20 02:03:52", "2012-05-29 07:00:03"),
    "2009_laquila": "2009-04-06 01:32:39",
    "muji": "2016-11-25 14:24:30.000",
}


def get_events_by_name_or_date(event_names_or_dates, catalog=geofon):
    stimes = []
    for sev in event_names_or_dates:
        if sev in aliases:
            if isinstance(aliases[sev], str):
                stimes.append(aliases[sev])
            else:
                stimes.extend(aliases[sev])
        else:
            stimes.append(sev)

    events_out = []
    for stime in stimes:
        if op.isfile(stime):
            events_out.extend(model.Event.load_catalog(stime))
        elif stime.startswith("gfz"):
            event = geofon.get_event(stime)
            events_out.append(event)
        else:
            t = util.str_to_time(stime)
            try:
                events = get_events(time_range=(t - 60.0, t + 60.0), catalog=catalog)
                events.sort(key=lambda ev: abs(ev.time - t))
                event = events[0]
            except IndexError:
                for catalog in [gcmt, usgs]:
                    logger.info("Nothing found in geofon! Trying others!")
                    events = get_events(
                        time_range=(t - 60.0, t + 60.0), catalog=catalog
                    )
                    events.sort(key=lambda ev: abs(ev.time - t))
                    if events:
                        event = events[0]

            events_out.append(event)

    return events_out


class NoArrival(Exception):
    pass


class PhaseWindow(object):
    def __init__(self, model, phases, omin, omax):
        self.model = model
        self.phases = phases
        self.omin = omin
        self.omax = omax

    def __call__(self, time, distance, depth):
        for ray in self.model.arrivals(
            phases=self.phases, zstart=depth, distances=[distance * cake.m2d]
        ):

            return time + ray.t + self.omin, time + ray.t + self.omax

        raise NoArrival


class VelocityWindow(object):
    def __init__(self, vmin, vmax=None, tpad=0.0):
        self.vmin = vmin
        self.vmax = vmax
        self.tpad = tpad

    def __call__(self, time, distance, depth):
        ttmax = (depth + distance) / self.vmin
        if self.vmax is not None:
            ttmin = (depth + distance) / self.vmax
        else:
            ttmin = 0.0

        return time + ttmin - self.tpad, time + ttmax + self.tpad


class FixedWindow(object):
    def __init__(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, time, distance, depth):
        return self.tmin, self.tmax


def dump_commandline(argv, fn):
    s = " ".join([pipes.quote(x) for x in argv])
    with open(fn, "w") as f:
        f.write(s)
        f.write("\n")


g_user_credentials = {}
g_auth_tokens = {}


def get_user_credentials(site):
    user, passwd = g_user_credentials.get(site, (None, None))
    token = g_auth_tokens.get(site, None)
    return dict(user=user, passwd=passwd, token=token)


program_name = "beatdown"
description = """
Download waveforms from FDSN web services and prepare for beat, adapted
from grond (https://github.com/pyrocko/grond)
""".strip()

logger = logging.getLogger("")

usage = """
usage: beatdown directory [options] [--] <YYYY-MM-DD> <HH:MM:SS> <lat> <lon> \\
                               <depth_km> <radius_km> <fmin_hz> \\
                               <sampling_rate_hz> \\
                               <eventname>

       beatdown directory [options] [--] <YYYY-MM-DD> <HH:MM:SS> <radius_km>\\
                                <fmin_hz> <sampling_rate_hz> <eventname>

       beatdown directory [options] [--] <catalog-eventname> <radius_km>\\
                                <fmin_hz> <sampling_rate_hz> <eventname>

       beatdown directory [options] --window="<YYYY-MM-DD HH:MM:SS, \\
                                YYYY-MM-DD HH:MM:SS>" \\
                                [--] <lat> <lon> <radius_km> <fmin_hz> \\
                                <sampling_rate_hz> <eventname>
""".strip()


def main():
    parser = OptionParser(usage=usage, description=description)

    parser.add_option(
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="allow recreation of output <directory>",
    )

    parser.add_option(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="print debugging information to stderr",
    )

    parser.add_option(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="show available stations/channels and exit " "(do not download waveforms)",
    )

    parser.add_option(
        "--continue",
        dest="continue_",
        action="store_true",
        default=False,
        help="continue download after a accident",
    )

    parser.add_option(
        "--local-data",
        dest="local_data",
        action="append",
        help="add file/directory with local data",
    )

    parser.add_option(
        "--local-stations",
        dest="local_stations",
        action="append",
        help="add local stations file",
    )

    parser.add_option(
        "--selection",
        dest="selection_file",
        action="append",
        help="add local stations file",
    )

    parser.add_option(
        "--local-responses-resp",
        dest="local_responses_resp",
        action="append",
        help="add file/directory with local responses in RESP format",
    )

    parser.add_option(
        "--local-responses-pz",
        dest="local_responses_pz",
        action="append",
        help="add file/directory with local pole-zero responses",
    )

    parser.add_option(
        "--local-responses-stationxml",
        dest="local_responses_stationxml",
        help="add file with local response information in StationXML format",
    )

    parser.add_option(
        "--window",
        dest="window",
        default="full",
        help='set time window to choose [full, p, "<time-start>,<time-end>"'
        "] (time format is YYYY-MM-DD HH:MM:SS)",
    )

    parser.add_option(
        "--out-components",
        choices=["enu", "rtu"],
        dest="out_components",
        default="rtu",
        help="set output component orientations to radial-transverse-up [rtu] "
        "(default) or east-north-up [enu]",
    )

    parser.add_option(
        "--out-units",
        choices=["M", "M/S", "M/S**2"],
        dest="output_units",
        default="M",
        help='set output units to displacement "M" (default),'
        ' velocity "M/S" or acceleration "M/S**2"',
    )

    parser.add_option(
        "--padding-factor",
        type=float,
        default=3.0,
        dest="padding_factor",
        help="extend time window on either side, in multiples of 1/<fmin_hz> "
        "(default: 5)",
    )

    parser.add_option(
        "--zero-padding",
        dest="zero_pad",
        action="store_true",
        default=False,
        help="Extend traces by zero-padding if clean restitution requires"
        "longer windows",
    )

    parser.add_option(
        "--credentials",
        dest="user_credentials",
        action="append",
        default=[],
        metavar="SITE,USER,PASSWD",
        help="user credentials for specific site to access restricted data "
        "(this option can be repeated)",
    )

    parser.add_option(
        "--token",
        dest="auth_tokens",
        metavar="SITE,FILENAME",
        action="append",
        default=[],
        help="user authentication token for specific site to access "
        "restricted data (this option can be repeated)",
    )

    parser.add_option(
        "--sites",
        dest="sites",
        metavar="SITE1,SITE2,...",
        default="geofon,iris,orfeus",
        help='sites to query (available: %s, default: "%%default"'
        % ", ".join(g_sites_available),
    )

    parser.add_option(
        "--band-codes",
        dest="priority_band_code",
        metavar="V,L,M,B,H,S,E,...",
        default="B,H",
        help="select and prioritize band codes (default: %default)",
    )

    parser.add_option(
        "--instrument-codes",
        dest="priority_instrument_code",
        metavar="H,L,G,...",
        default="H,L",
        help="select and prioritize instrument codes (default: %default)",
    )

    parser.add_option(
        "--radius-min",
        dest="radius_min",
        metavar="VALUE",
        default=0.0,
        type=float,
        help="minimum radius [km]",
    )

    parser.add_option(
        "--nstations-wanted",
        dest="nstations_wanted",
        metavar="N",
        type=int,
        help="number of stations to select initially",
    )

    (options, args) = parser.parse_args(sys.argv[1:])

    print("Parsed arguments:", args)
    if len(args) not in (10, 7, 6):
        parser.print_help()
        sys.exit(1)

    if options.debug:
        util.setup_logging(program_name, "debug")
    else:
        util.setup_logging(program_name, "info")

    if options.local_responses_pz and options.local_responses_resp:
        logger.critical(
            "cannot use local responses in PZ and RESP " "format at the same time"
        )
        sys.exit(1)

    n_resp_opt = 0
    for resp_opt in (
        options.local_responses_pz,
        options.local_responses_resp,
        options.local_responses_stationxml,
    ):

        if resp_opt:
            n_resp_opt += 1

    if n_resp_opt > 1:
        logger.critical(
            "can only handle local responses from either PZ or "
            "RESP or StationXML. Cannot yet merge different "
            "response formats."
        )
        sys.exit(1)

    if options.local_responses_resp and not options.local_stations:
        logger.critical(
            "--local-responses-resp can only be used " "when --stations is also given."
        )
        sys.exit(1)

    try:
        ename = ""
        magnitude = None
        mt = None
        if len(args) == 10:
            time = util.str_to_time(args[1] + " " + args[2])
            lat = float(args[3])
            lon = float(args[4])
            depth = float(args[5]) * km
            iarg = 6

        elif len(args) == 7:
            if args[2].find(":") == -1:
                sname_or_date = None
                lat = float(args[1])
                lon = float(args[2])
                event = None
                time = None
            else:
                sname_or_date = args[1] + " " + args[2]

            iarg = 3

        elif len(args) == 6:
            sname_or_date = args[1]
            iarg = 2

        if len(args) in (7, 6) and sname_or_date is not None:
            events = get_events_by_name_or_date([sname_or_date], catalog=geofon)
            if len(events) == 0:
                logger.critical("no event found")
                sys.exit(1)
            elif len(events) > 1:
                logger.critical("more than one event found")
                sys.exit(1)

            event = events[0]
            time = event.time
            lat = event.lat
            lon = event.lon
            depth = event.depth
            ename = event.name
            magnitude = event.magnitude
            mt = event.moment_tensor

        radius = float(args[iarg]) * km
        fmin = float(args[iarg + 1])
        sample_rate = float(args[iarg + 2])

        eventname = args[iarg + 3]
        cwd = str(sys.argv[1])
        event_dir = op.join(cwd, "data", "events", eventname)
        output_dir = op.join(event_dir, "waveforms")
    except:
        raise
        parser.print_help()
        sys.exit(1)

    if options.force and op.isdir(event_dir):
        if not options.continue_:
            shutil.rmtree(event_dir)

    if op.exists(event_dir) and not options.continue_:
        logger.critical(
            'directory "%s" exists. Delete it first or use the --force option'
            % event_dir
        )
        sys.exit(1)

    util.ensuredir(output_dir)

    if time is not None:
        event = model.Event(
            time=time,
            lat=lat,
            lon=lon,
            depth=depth,
            name=ename,
            magnitude=magnitude,
            moment_tensor=mt,
        )

    if options.window == "full":
        if event is None:
            logger.critical("need event for --window=full")
            sys.exit(1)

        low_velocity = 1500.0
        timewindow = VelocityWindow(low_velocity, tpad=options.padding_factor / fmin)

        tmin, tmax = timewindow(time, radius, depth)

    elif options.window == "p":
        if event is None:
            logger.critical("need event for --window=p")
            sys.exit(1)

        phases = list(map(cake.PhaseDef, "P p".split()))
        emod = cake.load_model()

        tpad = options.padding_factor / fmin
        timewindow = PhaseWindow(emod, phases, -tpad, tpad)

        arrivaltimes = []
        for dist in num.linspace(0, radius, 20):
            try:
                arrivaltimes.extend(timewindow(time, dist, depth))
            except NoArrival:
                pass

        if not arrivaltimes:
            logger.error("required phase arrival not found")
            sys.exit(1)

        tmin = min(arrivaltimes)
        tmax = max(arrivaltimes)

    else:
        try:
            stmin, stmax = options.window.split(",")
            tmin = util.str_to_time(stmin.strip())
            tmax = util.str_to_time(stmax.strip())

            timewindow = FixedWindow(tmin, tmax)

        except ValueError:
            logger.critical('invalid argument to --window: "%s"' % options.window)
            sys.exit(1)

    if event is not None:
        event.name = eventname

    tfade = tfade_factor / fmin

    tpad = tfade

    tmin -= tpad
    tmax += tpad

    tinc = None

    priority_band_code = options.priority_band_code.split(",")
    for s in priority_band_code:
        if len(s) != 1:
            logger.critical("invalid band code: %s" % s)

    priority_instrument_code = options.priority_instrument_code.split(",")
    for s in priority_instrument_code:
        if len(s) != 1:
            logger.critical("invalid instrument code: %s" % s)

    station_query_conf = dict(
        latitude=lat,
        longitude=lon,
        minradius=options.radius_min * km * cake.m2d,
        maxradius=radius * cake.m2d,
        channel=",".join("%s??" % s for s in priority_band_code),
    )

    target_sample_rate = sample_rate

    fmax = target_sample_rate

    # target_sample_rate = None
    # priority_band_code = ['H', 'B', 'M', 'L', 'V', 'E', 'S']

    priority_units = ["M/S", "M", "M/S**2"]

    # output_units = 'M'

    sites = [x.strip() for x in options.sites.split(",") if x.strip()]

    for site in sites:
        if site not in g_sites_available:
            logger.critical("unknown FDSN site: %s" % site)
            sys.exit(1)

    for s in options.user_credentials:
        try:
            site, user, passwd = s.split(",")
            g_user_credentials[site] = user, passwd
        except ValueError:
            logger.critical('invalid format for user credentials: "%s"' % s)
            sys.exit(1)

    for s in options.auth_tokens:
        try:
            site, token_filename = s.split(",")
            with open(token_filename, "r") as f:
                g_auth_tokens[site] = f.read()
        except (ValueError, OSError, IOError):
            logger.critical("cannot get token from file: %s" % token_filename)
            sys.exit(1)

    fn_template0 = (
        "data_%(network)s.%(station)s.%(location)s.%(channel)s_%(tmin)s.mseed"
    )

    fn_template_raw = op.join(output_dir, "raw", fn_template0)
    fn_stations_raw = op.join(output_dir, "stations.raw.txt")
    fn_template_rest = op.join(output_dir, "rest", fn_template0)
    fn_commandline = op.join(output_dir, "beatdown.command")

    ftap = (ffade_factors[0] * fmin, fmin, fmax, ffade_factors[1] * fmax)

    # chapter 1: download

    sxs = []
    for site in sites:
        try:
            extra_args = {"iris": dict(matchtimeseries=True)}.get(site, {})

            extra_args.update(station_query_conf)

            if site == "geonet":
                extra_args.update(starttime=tmin, endtime=tmax)
            else:
                extra_args.update(
                    startbefore=tmax,
                    endafter=tmin,
                    includerestricted=(
                        site in g_user_credentials or site in g_auth_tokens
                    ),
                )

            logger.info("downloading channel information (%s)" % site)
            sx = fdsn.station(site=site, format="text", level="channel", **extra_args)

        except fdsn.EmptyResult:
            logger.error("No stations matching given criteria. (%s)" % site)
            sx = None

        if sx is not None:
            sxs.append(sx)

    if all(sx is None for sx in sxs) and not options.local_data:
        sys.exit(1)

    nsl_to_sites = defaultdict(list)
    nsl_to_station = {}

    if options.selection_file:
        logger.info("using stations from stations file!")
        stations = []
        for fn in options.selection_file:
            stations.extend(model.load_stations(fn))

        nsls_selected = set(s.nsl() for s in stations)
    else:
        nsls_selected = None

    for sx, site in zip(sxs, sites):
        site_stations = sx.get_pyrocko_stations()
        for s in site_stations:
            nsl = s.nsl()

            nsl_to_sites[nsl].append(site)
            if nsl not in nsl_to_station:
                if nsls_selected:
                    if nsl in nsls_selected:
                        nsl_to_station[nsl] = s
                else:
                    nsl_to_station[nsl] = s  # using first site with this station

        logger.info("number of stations found: %i" % len(nsl_to_station))

    # station weeding
    if options.nstations_wanted:
        nsls_selected = None
        stations_all = [nsl_to_station[nsl_] for nsl_ in sorted(nsl_to_station.keys())]

        for s in stations_all:
            s.set_event_relative_data(event)

        stations_selected = weeding.weed_stations(
            stations_all, options.nstations_wanted
        )[0]

        nsls_selected = set(s.nsl() for s in stations_selected)
        logger.info("number of stations selected: %i" % len(nsls_selected))

    if tinc is None:
        tinc = 3600.0

    have_data = set()

    if options.continue_:
        fns = glob.glob(fn_template_raw % starfill())
        p = pile.make_pile(fns)
    else:
        fns = []

    have_data_site = {}
    could_have_data_site = {}
    for site in sites:
        have_data_site[site] = set()
        could_have_data_site[site] = set()

    available_through = defaultdict(set)
    it = 0
    nt = int(math.ceil((tmax - tmin) / tinc))
    for it in range(nt):
        tmin_win = tmin + it * tinc
        tmax_win = min(tmin + (it + 1) * tinc, tmax)
        logger.info(
            "time window %i/%i (%s - %s)"
            % (it + 1, nt, util.tts(tmin_win), util.tts(tmax_win))
        )

        have_data_this_window = set()
        if options.continue_:
            trs_avail = p.all(tmin=tmin_win, tmax=tmax_win, load_data=False)
            for tr in trs_avail:
                have_data_this_window.add(tr.nslc_id)
        for site, sx in zip(sites, sxs):
            if sx is None:
                continue

            selection = []
            channels = sx.choose_channels(
                target_sample_rate=target_sample_rate,
                priority_band_code=priority_band_code,
                priority_units=priority_units,
                priority_instrument_code=priority_instrument_code,
                timespan=(tmin_win, tmax_win),
            )

            for nslc in sorted(channels.keys()):
                if nsls_selected is not None and nslc[:3] not in nsls_selected:
                    continue

                could_have_data_site[site].add(nslc)

                if nslc not in have_data_this_window:
                    channel = channels[nslc]
                    if event:
                        lat_, lon_ = event.lat, event.lon
                    else:
                        lat_, lon_ = lat, lon
                    try:
                        dist = orthodrome.distance_accurate50m_numpy(
                            lat_, lon_, channel.latitude.value, channel.longitude.value
                        )
                    except:
                        dist = orthodrome.distance_accurate50m_numpy(
                            lat_, lon_, channel.latitude, channel.longitude
                        )

                    if event:
                        depth_ = event.depth
                        time_ = event.time
                    else:
                        depth_ = None
                        time_ = None

                    tmin_, tmax_ = timewindow(time_, dist, depth_)

                    tmin_this = tmin_ - tpad
                    tmax_this = float(tmax_ + tpad)

                    tmin_req = max(tmin_win, tmin_this)
                    tmax_req = min(tmax_win, tmax_this)
                    if channel.sample_rate:
                        try:
                            deltat = 1.0 / int(channel.sample_rate.value)
                        except:
                            deltat = 1.0 / int(channel.sample_rate)
                    else:
                        deltat = 1.0

                    if tmin_req < tmax_req:
                        logger.debug("deltat %f" % deltat)
                        # extend time window by some samples because otherwise
                        # sometimes gaps are produced
                        # apparently the WS are only sensitive to full seconds
                        # round to avoid gaps, increase safetiy window
                        selection.append(
                            nslc
                            + (
                                math.floor(tmin_req - deltat * 20.0),
                                math.ceil(tmax_req + deltat * 20.0),
                            )
                        )
            if options.dry_run:
                for (net, sta, loc, cha, tmin, tmax) in selection:
                    available_through[net, sta, loc, cha].add(site)

            else:
                neach = 100
                i = 0
                nbatches = ((len(selection) - 1) // neach) + 1
                while i < len(selection):
                    selection_now = selection[i : i + neach]
                    f = tempfile.NamedTemporaryFile()
                    try:
                        sbatch = ""
                        if nbatches > 1:
                            sbatch = " (batch %i/%i)" % ((i // neach) + 1, nbatches)

                        logger.info("downloading data (%s)%s" % (site, sbatch))
                        data = fdsn.dataselect(
                            site=site,
                            selection=selection_now,
                            **get_user_credentials(site)
                        )

                        while True:
                            buf = data.read(1024)
                            if not buf:
                                break
                            f.write(buf)

                        f.flush()

                        trs = io.load(f.name)
                        for tr in trs:
                            tr.fix_deltat_rounding_errors()
                            logger.debug(
                                "cutting window: %f - %f" % (tmin_win, tmax_win)
                            )
                            logger.debug(
                                "available window: %f - %f, nsamples: %g"
                                % (tr.tmin, tr.tmax, tr.ydata.size)
                            )
                            try:
                                logger.debug("tmin before snap %f" % tr.tmin)
                                tr.snap(interpolate=True)
                                logger.debug("tmin after snap %f" % tr.tmin)
                                tr.chop(
                                    tmin_win,
                                    tmax_win,
                                    snap=(math.floor, math.ceil),
                                    include_last=True,
                                )
                                logger.debug(
                                    "cut window: %f - %f, nsamles: %g"
                                    % (tr.tmin, tr.tmax, tr.ydata.size)
                                )
                                have_data.add(tr.nslc_id)
                                have_data_site[site].add(tr.nslc_id)
                            except trace.NoData:
                                pass

                        fns2 = io.save(trs, fn_template_raw)
                        for fn in fns2:
                            if fn in fns:
                                logger.warn("overwriting file %s", fn)
                        fns.extend(fns2)

                    except fdsn.EmptyResult:
                        pass

                    except HTTPError:
                        logger.warn(
                            "an error occurred while downloading data "
                            "for channels \n  %s"
                            % "\n  ".join(".".join(x[:4]) for x in selection_now)
                        )

                    f.close()
                    i += neach

    if options.dry_run:
        nslcs = sorted(available_through.keys())

        all_channels = defaultdict(set)
        all_stations = defaultdict(set)

        def plural_s(x):
            return "" if x == 1 else "s"

        for nslc in nslcs:
            sites = tuple(sorted(available_through[nslc]))
            logger.info(
                "selected: %s.%s.%s.%s from site%s %s"
                % (nslc + (plural_s(len(sites)), "+".join(sites)))
            )

            all_channels[sites].add(nslc)
            all_stations[sites].add(nslc[:3])

        nchannels_all = 0
        nstations_all = 0
        for sites in sorted(
            all_channels.keys(), key=lambda sites: (-len(sites), sites)
        ):

            nchannels = len(all_channels[sites])
            nstations = len(all_stations[sites])
            nchannels_all += nchannels
            nstations_all += nstations
            logger.info(
                "selected (%s): %i channel%s (%i station%s)"
                % (
                    "+".join(sites),
                    nchannels,
                    plural_s(nchannels),
                    nstations,
                    plural_s(nstations),
                )
            )

        logger.info(
            "selected total: %i channel%s (%i station%s)"
            % (
                nchannels_all,
                plural_s(nchannels_all),
                nstations_all,
                plural_s(nstations_all),
            )
        )

        logger.info("dry run done.")
        sys.exit(0)

    for nslc in have_data:
        # if we are in continue mode, we have to guess where the data came from
        if not any(nslc in have_data_site[site] for site in sites):
            for site in sites:
                if nslc in could_have_data_site[site]:
                    have_data_site[site].add(nslc)

    sxs = {}
    for site in sites:
        selection = []
        for nslc in sorted(have_data_site[site]):
            selection.append(nslc + (tmin - tpad, tmax + tpad))

        if selection:
            logger.info("downloading response information (%s)" % site)
            sxs[site] = fdsn.station(site=site, level="response", selection=selection)

            sxs[site].dump_xml(filename=op.join(output_dir, "stations.%s.xml" % site))

    # chapter 1.5: inject local data

    if options.local_data:
        have_data_site["local"] = set()
        plocal = pile.make_pile(options.local_data, fileformat="detect")
        logger.info(
            "Importing local data from %s between %s (%f) and %s (%f)"
            % (
                options.local_data,
                util.time_to_str(tmin),
                tmin,
                util.time_to_str(tmax),
                tmax,
            )
        )
        for traces in plocal.chopper_grouped(
            gather=lambda tr: tr.nslc_id, tmin=tmin, tmax=tmax, tinc=tinc
        ):

            for tr in traces:
                if tr.nslc_id not in have_data:
                    fns.extend(io.save(traces, fn_template_raw))
                    have_data_site["local"].add(tr.nslc_id)
                    have_data.add(tr.nslc_id)

        sites.append("local")

    if options.local_responses_pz:
        sxs["local"] = epz.make_stationxml(epz.iload(options.local_responses_pz))

    if options.local_responses_resp:
        local_stations = []
        for fn in options.local_stations:
            local_stations.extend(model.load_stations(fn))

        sxs["local"] = resp.make_stationxml(
            local_stations, resp.iload(options.local_responses_resp)
        )

    if options.local_responses_stationxml:
        sxs["local"] = stationxml.load_xml(filename=options.local_responses_stationxml)

    # chapter 1.6: dump raw data stations file

    nsl_to_station = {}
    for site in sites:
        if site in sxs:
            stations = sxs[site].get_pyrocko_stations(timespan=(tmin, tmax))
            for s in stations:
                nsl = s.nsl()
                if nsl not in nsl_to_station:
                    nsl_to_station[nsl] = s

    stations = [nsl_to_station[nsl_] for nsl_ in sorted(nsl_to_station.keys())]

    util.ensuredirs(fn_stations_raw)
    model.dump_stations(stations, fn_stations_raw)

    dump_commandline(sys.argv, fn_commandline)

    # chapter 2: restitution

    if not fns:
        logger.error("no data available")
        sys.exit(1)

    p = pile.make_pile(fns, show_progress=False)
    p.get_deltatmin()
    otinc = None
    if otinc is None:
        otinc = nice_seconds_floor(p.get_deltatmin() * 500000.0)
    otinc = 3600.0
    otmin = math.floor(p.tmin / otinc) * otinc
    otmax = math.ceil(p.tmax / otinc) * otinc
    otpad = tpad * 2

    fns = []
    rest_traces_b = []
    win_b = None
    for traces_a in p.chopper_grouped(
        gather=lambda tr: tr.nslc_id, tmin=otmin, tmax=otmax, tinc=otinc, tpad=otpad
    ):

        rest_traces_a = []
        win_a = None
        for tr in traces_a:
            win_a = tr.wmin, tr.wmax

            if win_b and win_b[0] >= win_a[0]:
                fns.extend(cut_n_dump(rest_traces_b, win_b, fn_template_rest))
                rest_traces_b = []
                win_b = None

            response = None
            failure = []
            for site in sites:
                try:
                    if site not in sxs:
                        continue
                    logger.debug("Getting response for %s" % tr.__str__())
                    response = sxs[site].get_pyrocko_response(
                        tr.nslc_id,
                        timespan=(tr.tmin, tr.tmax),
                        fake_input_units=options.output_units,
                    )

                    break

                except stationxml.NoResponseInformation:
                    failure.append("%s: no response information" % site)

                except stationxml.MultipleResponseInformation:
                    failure.append("%s: multiple response information" % site)

            if response is None:
                failure = ", ".join(failure)

            else:
                failure = ""
                try:
                    if tr.tmin > tmin and options.zero_pad:
                        logger.warning(
                            "Trace too short for clean restitution in "
                            "desired frequency band -> zero-padding!"
                        )
                        tr.extend(tr.tmin - tfade, tr.tmax + tfade, "repeat")

                    rest_tr = tr.transfer(tfade, ftap, response, invert=True)
                    rest_traces_a.append(rest_tr)

                except (trace.TraceTooShort, trace.NoData):
                    failure = "trace too short"

            if failure:
                logger.warn(
                    "failed to restitute trace %s.%s.%s.%s (%s)"
                    % (tr.nslc_id + (failure,))
                )

        if rest_traces_b:
            rest_traces = trace.degapper(
                rest_traces_b + rest_traces_a, deoverlap="crossfade_cos"
            )

            fns.extend(cut_n_dump(rest_traces, win_b, fn_template_rest))
            rest_traces_a = []
            if win_a:
                for tr in rest_traces:
                    try:
                        rest_traces_a.append(
                            tr.chop(win_a[0], win_a[1] + otpad, inplace=False)
                        )
                    except trace.NoData:
                        pass

        rest_traces_b = rest_traces_a
        win_b = win_a

    fns.extend(cut_n_dump(rest_traces_b, win_b, fn_template_rest))

    # chapter 3: rotated restituted traces for inspection

    if not event:
        sys.exit(0)

    fn_template1 = "DISPL.%(network)s.%(station)s.%(location)s.%(channel)s"

    fn_waveforms = op.join(output_dir, "prepared", fn_template1)
    fn_stations = op.join(output_dir, "stations.prepared.txt")
    fn_event = op.join(event_dir, "event.txt")
    fn_event_yaml = op.join(event_dir, "event.yaml")

    nsl_to_station = {}
    for site in sites:
        if site in sxs:
            stations = sxs[site].get_pyrocko_stations(timespan=(tmin, tmax))
            for s in stations:
                nsl = s.nsl()
                if nsl not in nsl_to_station:
                    nsl_to_station[nsl] = s

    p = pile.make_pile(fns, show_progress=False)

    deltat = None
    if sample_rate is not None:
        deltat = 1.0 / sample_rate

    traces_beat = []
    used_stations = []
    for nsl, s in nsl_to_station.items():
        s.set_event_relative_data(event)
        traces = p.all(trace_selector=lambda tr: tr.nslc_id[:3] == nsl)

        if options.out_components == "rtu":
            pios = s.guess_projections_to_rtu(out_channels=("R", "T", "Z"))
        elif options.out_components == "enu":
            pios = s.guess_projections_to_enu(out_channels=("E", "N", "Z"))
        else:
            assert False

        for (proj, in_channels, out_channels) in pios:

            proc = trace.project(traces, proj, in_channels, out_channels)
            for tr in proc:
                tr_beat = heart.SeismicDataset.from_pyrocko_trace(tr)
                traces_beat.append(tr_beat)
                for ch in out_channels:
                    if ch.name == tr.channel:
                        s.add_channel(ch)

            if proc:
                io.save(proc, fn_waveforms)
                used_stations.append(s)

    stations = list(used_stations)
    util.ensuredirs(fn_stations)
    model.dump_stations(stations, fn_stations)
    model.dump_events([event], fn_event)

    from pyrocko.guts import dump

    dump([event], filename=fn_event_yaml)

    utility.dump_objects(
        op.join(cwd, "seismic_data.pkl"), outlist=[stations, traces_beat]
    )
    logger.info("prepared waveforms from %i stations" % len(stations))
