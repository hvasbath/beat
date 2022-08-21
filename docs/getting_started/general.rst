General
-------
The beat internal help can always be executed by typing '--help' in the execution string.
Also, by using the 'tab' button you can use automatic bash completion to see available commands and options.
For example::

    beat init --help

Will display::

    Usage: beat init <event_name> <event_date "YYYY-MM-DD"> [options]

    Create a new EQ model project, use only event name to skip catalog search.

    Options:
      -h, --help            show this help message and exit
      --min_mag=MIN_MAG     Minimum Mw for event, for catalog search. Default:
                            "6.0"
      --main_path=MAIN_PATH
                            Main path (absolute) for creating directory structure.
                            Default: current directory ./
      --datatypes=DATATYPES
                            Datatypes to include in the setup; "geodetic,
                            seismic".
      --mode=MODE           Inversion problem to solve; "geometry", "ffi",
                            "interseismic" Default: "geometry"
      --source_type=SOURCE_TYPE
                            Source type to solve for; ExplosionSource",
                            "RectangularExplosionSource", "DCSource",
                            "CLVDSource", "MTSource", "RectangularSource",
                            "DoubleDCSource", "RingfaultSource. Default:
                            "RectangularSource"
      --n_sources=N_SOURCES
                            Integer Number of sources to invert for. Default: 1
      --waveforms=WAVEFORMS
                            Waveforms to include in the setup; "any_P, any_S,
                            slowest".
      --sampler=SAMPLER     Sampling algorithm to sample the solution space of the
                            general problem; "SMC", "Metropolis". Default: "SMC"
      --hyper_sampler=HYPER_SAMPLER
                            Sampling algorithm to sample the solution space of the
                            hyperparameters only; So far only "Metropolis"
                            supported.Default: "Metropolis"
      --use_custom          If set, a slot for a custom velocity model is being
                            created in the configuration file.
      --individual_gfs      If set, Green's Function stores will be created
                            individually for each station!
      --loglevel=LOGLEVEL   set logger level to "critical", "error", "warning",
                            "info", or "debug". Default is "info".
