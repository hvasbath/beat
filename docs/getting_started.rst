.. getting_started:

*************************
Getting started with BEAT
*************************

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


Initialise a new Modeling Project
---------------------------------

Each modeling project is initiated with the "beat init" command. There are many options that define the type of optimization, datatypes to include, sampling algorithm to use, number of sources, velocity model to use for the Greens Function calculations etc ...

For example to optimize for a Full Moment Tensor for the Landers EQ by using seismic data, with station dependend Greens Functions for P and S waves with the default sampling algorithm (Sequential Monte Carlo) run::

    beat init LandersEQ 1992-06-28 --datatypes='seismic' --individual_gfs --n_sources=1 --source_type=MTSource --min_mag=7

This will create project directory called LandersEQ in the current directory.
Within the directoy you will see that there have been two files created:
 - BEAT_log.txt
 - geometry_config.yaml

The first file is a logging file where all the executed comments and outputs are written to. In case something goes wrong this log file helps to find the error.
For now it contains::

    2018-01-04 16:15:06,696 - utility - INFO - Getting relevant events from the gCMT catalog for the dates:1992-06-27 00:00:00.000 - 1992-06-29 00:00:00.000

    2018-01-04 16:15:07,097 - config - INFO - Added hyperparameter h_any_P_Z to config and model setup!
    2018-01-04 16:15:07,097 - config - INFO - Added hyperparameter h_any_S_Z to config and model setup!
    2018-01-04 16:15:07,097 - config - INFO - All hyper-parameters ok!
    2018-01-04 16:15:07,097 - config - INFO - Number of hyperparameters! 2
    2018-01-04 16:15:07,098 - config - INFO - All parameter-priors ok!
    2018-01-04 16:15:07,102 - config - INFO - Project_directory: /home/vasyurhm/BEATS/LandersEQ

The second file is a yaml-configuration file and it is where ALL the changes in parameters and settings have to be done to avoid tinkering with the program itself!
This file can be read as is by the computer, therefore, it is important to keep the syntax clean!
The content of this file is basically the serialised instance of the BEAT config class. At first the amount of content seems overwhelming, but once you are familiar with the variables you will find that there are not too many things that will need to be edited. Also we have to be aware that the problem we are going to try to solve is very complex, ergo a complex parameter file is somewhat understandable.
To find a short explanation to each parameter and its format the reader is refered to the webpage of the `config <https://hvasbath.github.io/beat/_modules/config.html#SeismicConfig>`__ module.

This example configuration file looks like this::

    --- !beat.BEATconfig
    name: LandersEQ
    date: '1992-06-28'
    event: !pf.Event
      lat: 34.65
      lon: -116.65
      time: 1992-06-28 11:57:53
      name: 062892C
      depth: 15000.0
      magnitude: 7.316312340268055
      region: SOUTHERN CALIFORNIA
      catalog: gCMT
      moment_tensor: !pf.MomentTensor
        mnn: -6.12e+19
        mee: 7.001e+19
        mdd: -8.81e+18
        mne: -7.335e+19
        mnd: 3.807e+19
        med: -9.9e+17
        strike1: 247.72308708747312
        dip1: 82.44124210318292
        rake1: -20.30350409572225
        strike2: 340.50937853818954
        dip2: 69.88059010043526
        rake2: -171.9468551048134
        moment: 1.0579582033331939e+20
        magnitude: 7.316312340268055
      duration: 38.4
    project_dir: /home/vasyurhm/BEATS/LandersEQ
    problem_config: !beat.ProblemConfig
      mode: geometry
      source_type: MTSource
      stf_type: HalfSinusoid
      n_sources: 1
      datatypes: [seismic]
      hyperparameters:
        h_any_P_Z: !beat.heart.Parameter
          name: h_any_P_Z
          form: Uniform
          lower: [-20.0]
          upper: [20.0]
          testvalue: [0.0]
        h_any_S_Z: !beat.heart.Parameter
          name: h_any_S_Z
          form: Uniform
          lower: [-20.0]
          upper: [20.0]
          testvalue: [0.0]
      priors:
        depth: !beat.heart.Parameter
          name: depth
          form: Uniform
          lower: [0.0]
          upper: [5.0]
          testvalue: [2.5]
        duration: !beat.heart.Parameter
          name: duration
          form: Uniform
          lower: [0.0]
          upper: [20.0]
          testvalue: [10.0]
        east_shift: !beat.heart.Parameter
          name: east_shift
          form: Uniform
          lower: [-10.0]
          upper: [10.0]
          testvalue: [0.0]
        magnitude: !beat.heart.Parameter
          name: magnitude
          form: Uniform
          lower: [4.0]
          upper: [7.0]
          testvalue: [5.5]
        mdd: !beat.heart.Parameter
          name: mdd
          form: Uniform
          lower: [-1.4142135623730951]
          upper: [1.4142135623730951]
          testvalue: [0.0]
        med: !beat.heart.Parameter
          name: med
          form: Uniform
          lower: [-1.0]
          upper: [1.0]
          testvalue: [0.0]
        mee: !beat.heart.Parameter
          name: mee
          form: Uniform
          lower: [-1.4142135623730951]
          upper: [1.4142135623730951]
          testvalue: [0.0]
        mnd: !beat.heart.Parameter
          name: mnd
          form: Uniform
          lower: [-1.0]
          upper: [1.0]
          testvalue: [0.0]
        mne: !beat.heart.Parameter
          name: mne
          form: Uniform
          lower: [-1.0]
          upper: [1.0]
          testvalue: [0.0]
        mnn: !beat.heart.Parameter
          name: mnn
          form: Uniform
          lower: [-1.4142135623730951]
          upper: [1.4142135623730951]
          testvalue: [0.0]
        north_shift: !beat.heart.Parameter
          name: north_shift
          form: Uniform
          lower: [-10.0]
          upper: [10.0]
          testvalue: [0.0]
        time: !beat.heart.Parameter
          name: time
          form: Uniform
          lower: [-3.0]
          upper: [3.0]
          testvalue: [0.0]
    seismic_config: !beat.SeismicConfig
      datadir: ./
      blacklist: [placeholder]
      calc_data_cov: true
      pre_stack_cut: true
      waveforms:
      - !beat.WaveformFitConfig
        include: true
        name: any_P
        channels: [Z]
        filterer: !beat.heart.Filter
          lower_corner: 0.001
          upper_corner: 0.1
          order: 4
        distances: [30.0, 90.0]
        interpolation: multilinear
        arrival_taper: !beat.heart.ArrivalTaper
          a: -15.0
          b: -10.0
          c: 50.0
          d: 55.0
      - !beat.WaveformFitConfig
        include: true
        name: any_S
        channels: [Z]
        filterer: !beat.heart.Filter
          lower_corner: 0.001
          upper_corner: 0.1
          order: 4
        distances: [30.0, 90.0]
        interpolation: multilinear
        arrival_taper: !beat.heart.ArrivalTaper
          a: -15.0
          b: -10.0
          c: 50.0
          d: 55.0
      gf_config: !beat.SeismicGFConfig
        store_superdir: ./
        reference_model_idx: 0
        n_variations: [0, 1]
        error_depth: 0.1
        error_velocities: 0.1
        depth_limit_variation: 600.0
        earth_model_name: ak135-f-average.m
        use_crust2: true
        replace_water: true
        source_depth_min: 0.0
        source_depth_max: 10.0
        source_depth_spacing: 1.0
        source_distance_radius: 20.0
        source_distance_spacing: 1.0
        nworkers: 1
        code: qssp
        sample_rate: 2.0
        rm_gfs: true
    sampler_config: !beat.SamplerConfig
      name: SMC
      progressbar: true
      parameters: !beat.SMCConfig
        n_chains: 1000
        n_steps: 100
        n_jobs: 1
        tune_interval: 10
        coef_variation: 1.0
        stage: 0
        proposal_dist: MultivariateNormal
        check_bnd: true
        update_covariances: false
        rm_flag: false
    hyper_sampler_config: !beat.SamplerConfig
      name: Metropolis
      progressbar: true
      parameters: !beat.MetropolisConfig
        n_jobs: 1
        n_stages: 10
        n_steps: 25000
        stage: 0
        tune_interval: 50
        proposal_dist: Normal
        thin: 2
        burn: 0.5
        rm_flag: false


Each BEAT config consists of some general information, from information collected from the gCMT catalog, of a ProblemConfig, the configurations for each dataset (here only seismic_config) and configurations for the sampling algorithms to use for the optimizations of the general problem (sampler_config) as well as a of an initial guess for the hyperparameters (hyper_sampler_config).

Most of the edits likely will be made in the ProblemConfig, particularly in the priors of the source parameters. For now only uniform priors are available. To change the bounds of the priors simply type other values into the 'upper' and 'lower' fields of each source parameter. Note: The test parameter needs to be within these bounds!
To fix one or the other parameter in the optimizations the upper and lower bounds as well as the test value need to be set equal.


Import Data
-----------
This is the step to import the user data into the program format and setup.


geodetic data
^^^^^^^^^^^^^

InSAR
=====
To use static displacement InSAR measurements you need to prepare your data first with `kite <https://github.com/pyrocko/kite>`__.
Kite handels displacement data from a variety of formats, such as e.g. GMTSAR, ISCE, ROIPAC and GAMMA. After importing the data into kite you
should consider to subsample it and to calculate the data-error-variance-covariance as described in the `kite documentation <https://pyrocko.org/kite/docs/current/>`__.
Once you are satisfied with your specifications please store the kite scenes in its native format as "numpy-npz containers".

In the $project_dir you find the config_geometry.yaml, where the geodetic_config variable 'datadir' points to the location where the data are stored.
Under the 'names' variable, the names of the files of interest have to be entered (without the .npz and .yml suffixes). Afterwards, the following command has to be executed to import the data::

    beat import $project_dir

The data are now accessible to beat as the file geodetic_data.pkl. In case it turns out the pre-processing (subsampling, covariance estimation) had to be repeated, the existing 'geodetic_data.pkl' file can be overwritten by adding the --force option to the import command above.

GNSS
====
The supported format for GNSS data is an ASCII file of the following format::

  DOGG  10.0000   15.6546   -0.61   0.44   3.5900     0.18  0.15  0.7000
  CATT  135.0000  -45.000   0.15    -0.57  1.6100     0.23  0.20  0.9000
  COOW  45.0000   98.0000   12.20   15.01  22.8600    0.93  0.78  3.5700

The columns are in this order: station name, Longitude, Latitude, velocity east component, velocity north component, velocity vertical component, standard-deviaion east component, standard-deviaion north component, standard-deviaion vertical component,
The units for the location and the measurements are [decimal deg] and [mm/yr], respectively.

.. note:: This is the native GAMMIT output file.

seismic data
^^^^^^^^^^^^

So far, unfortunately only the output of `autokiwi <https://github.com/emolch/kiwi>`__ is supported for automatic import of seismic data.
To get other types of data imported the user will have to do some programing.

The following remarks are just bits and pieces that may be followed to write a script to bring the data into the necessary format.

The seismic data may be saved using the package "pickle" as a file "seismic_data.pkl" containing a list of 2 lists: 1. list of "pyrocko.trace.Trace" objects alternating for (Z / T) rotated traces. 2. list of "pyrocko.model.Station" objects in the same order like the data traces.

Pyrocko supports the import of various data formats and all the necessary tools to remove the instrument response and to convert the traces to displacement.
How to do this based on some examples is shown `here <https://pyrocko.org/docs/current/library/examples/trace_handling.html#restitute-to-displacement-using-poles-and-zeros>`__ webpage.

Once you have done this with your data, you only need to create the second list of station objects.
To create this list we provide a helper function 'setup_stations' in the inputf module.
Within an ipython session type::

    from beat import inputf
    inputf.setup_stations?

Will show::

    inputf.setup_stations(lats, lons, names, networks, event)
    Docstring:
    Setup station objects, based on station coordinates and reference event.

    Parameters
    ----------
    lats : :class:`num.ndarray`
        of station location latitude
    lons : :class:`num.ndarray`
        of station location longitude
    names : list
        of strings of station names
    networks : list
        of strings of network names for each station
    event : :class:`pyrocko.model.Event`

    Results
    -------
    stations : list
        of :class:`pyrocko.model.Station`

The event object that may be used in this function is the one shown in the top of the configuration file.
In an ipython session from within the LandersEQ directory execute::

    from pyrocko import guts
    from beat import config

    c = guts.load(filename='config_geometry.yaml')
    print c.event

Will yield::

    --- !pf.Event
    lat: 34.65
    lon: -116.65
    time: 1992-06-28 11:57:53
    name: 062892C
    depth: 15000.0
    magnitude: 7.316312340268055
    region: SOUTHERN CALIFORNIA
    catalog: gCMT
    moment_tensor: !pf.MomentTensor
      mnn: -6.12e+19
      mee: 7.001e+19
      mdd: -8.81e+18
      mne: -7.335e+19
      mnd: 3.807e+19
      med: -9.9e+17
      strike1: 247.72308708747312
      dip1: 82.44124210318292
      rake1: -20.30350409572225
      strike2: 340.50937853818954
      dip2: 69.88059010043526
      rake2: -171.9468551048134
      moment: 1.0579582033331939e+20
      magnitude: 7.316312340268055
    duration: 38.4


Once a list of traces and station objects exists it may be exported to the project directory (here path from example)::

    from beat import utility

    seismic_outpath='/home/vasyurhm/BEATS/LandersEQ/seismic_data.pkl'
    utility.dump_objects(seismic_outpath, outlist=[stations, data_traces])


How to update the configuration files
-------------------------------------
In the course of development in BEAT it happened and may happen in the future that
the structure in the configuration file changes. Thus after a code base upgrade it may happen that
older configuration files cannot be read anymore. The program will raise an Error with the message
that the configuration file has to be updated and how this can be done. However, it may be of interest to know
before the actual update what is going to change. These changes can be displayed with::

    beat update <project_directory> --diff

An update screen that appears may look like this.

  .. image:: _static/beat_update_cut.png

Where omitting the "--diff" option will update the configuration file right away.

An update of the configuration file is also necessary, if some of the hierarchical parameters are affected by some changes in the configuration file. For example by enabling temporal "station_corrections" in seismic setups or enabling "fit_ramp"
for residual ramp estimation in the InSAR data.::

    beat update <project_directory> --diff --parameters='hierarchicals, hypers'


How to setup a Custom Greens Function Store
-------------------------------------------
This section covers how to generate a custom Greens Function store for seismic data at a location of choice.
First a new model project has to be created to generate the configuration file. As we have no specific event in mind
we skip the catalog search by not specifiying the date.::

    beat init Cascadia --datatypes='seismic' --mode='geometry' --use_custom

.. note::
    To use the default ak135 earth-model in combination with crust2 one needs to execute above command without the '--use_custom' flag.

This will create a beat project folder named 'Cascadia' and a configuration file 'config_geometry.yaml'.
In this configuration file we may now edit the reference location and the velocity model to the specific model we
received from a colleague or found in the literature.::

    cd Cascadia
    vi config_geometry.yaml

This will open the configuration file in the vi editor.
In lines 4-6 we find the reference event::

    event: !pf.Event
      lat: 0.0
      lon: 0.0

In lines 160-165 we find the reference location::

    reference_location: !beat.heart.ReferenceLocation
      lat: 10.0
      lon: 10.0
      elevation: 0.0
      depth: 0.0
      station: Store_Name

The distance between these two locations determines the center point of the grid of Greens Functions that we want to calculate.
For our example we set the reference location close to Vancouver, Canada as we want to study the Cascadia subduction zone.
We ignore the 'elevation' and 'depth' attributes but we set the 'station' attribute, which determines the prefix of the name of the
Greens Function store. ::

    reference_location: !beat.heart.ReferenceLocation
      lat: 49.28098
      lon: -123.12244
      elevation: 0.0
      depth: 0.0
      station: Vancouver

The events we want to study are going to be around Vancouver island, and we set the reference event coordinates accordingly::

    event: !pf.Event
      lat: 49.608839
      lon: -125.647683

So far we determined the general location of the store, but now we need to specify the spatial dimensions of the grid.
In lines 154-158 we find the respective parameters::

    source_depth_min: 0.0
    source_depth_max: 10.0
    source_depth_spacing: 1.0
    source_distance_radius: 20.0
    source_distance_spacing: 1.0

We are going to have stations in a distance of 500km and the events we are going to study are going to be located in a depth range of 5-40km depth so we update these values accordingly.::

    source_depth_min: 5.0
    source_depth_max: 40.0
    source_depth_spacing: 1.0
    source_distance_radius: 500.0
    source_distance_spacing: 1.0

To decide on the resolution of the grid and the sample_rate (line 167) depends on the use-case and aim of the study.
A description of the corner points to have in mind is `here <https://pyrocko.org/docs/current/apps/fomosto/tutorial.html#considerations-for-real-world-applications>`__
For example for a regional Full Moment Tensor inversion we want to optimize data up to 0.2 Hz. So a source grid of 1. km step size and 1. Hz 'sample_rate' seems a safe way to go.
As we are in a regional setup we use QSEIS for the calculation of the Greens Functions, which we have to specify in line 166.::

    code: qseis

In the lines 144-153 is the velocity model defined for which the Greens Functions are going to be calculated.::

    custom_velocity_model: |2
          0.             5.8            3.46           2.6         1264.           600.
         20.             5.8            3.46           2.6         1264.           600.
         20.             6.5            3.85           2.9         1283.           600.
         35.             6.5            3.85           2.9         1283.           600.
      mantle
         35.             8.04           4.48           3.58        1449.           600.
         77.5            8.045          4.49           3.5         1445.           600.
         77.5            8.045          4.49           3.5          180.6           75.
        100.             8.048          4.495          3.461        180.3           75.

The columns are 'depth', 'p-wave velocity', 's-wave velocity', 'density', 'Qp', 'Qs'. Here the values may be changed to a custom velociy.
For example if we want to add another layer (from 20-25 km depth) between 20 and 35 km depth we would write::

    custom_velocity_model: |2
          0.             5.8            3.46           2.6         1264.           600.
         20.             5.8            3.46           2.6         1264.           600.
         20.             6.1            3.54           2.7         1280.           600.
         25.             6.1            3.54           2.7         1280.           600.
         25.             6.5            3.85           2.9         1283.           600.
         35.             6.5            3.85           2.9         1283.           600.
      mantle
         35.             8.04           4.48           3.58        1449.           600.
         77.5            8.045          4.49           3.5         1445.           600.
         77.5            8.045          4.49           3.5          180.6           75.
        100.             8.048          4.495          3.461        180.3           75.

Below the specified depth it is going to be combined with the earth-model specified in line 141.

.. note::
    In case the standard earth-model should be used rather than a custom model the 'custom_velocity_model' attribute may be deleted.
    For the shallow crust one may decide to use the implemented crust2 model and to remove (potential) water layers. Lines 141-143::

        earth_model_name: ak135-f-average.m
        use_crust2: false
        replace_water: false

Then, we have to specify under line 135 'store_superdir' the path to the directory where to save the GreensFunction store on disk.
One should have in mind that for large grids with high sample-rate the stores might become several GigaBytes in size and may calculate a very long time!

Lastly, we start the store calculation with::

    beat build_gf Cascadia --execute --datatypes='seismic'


How to setup a Finite Fault Optimization
----------------------------------------

In a finite fault optimization in beat a pre-defined RectangularSource (reference fault) is discretized into sub-patches.
Each of these sub-patches may have up to 4 parameters to be optimized for. In the static case (geodetic data) these are two slip-parameters
perpendicular and parallel to the rake direction of the reference fault. In the kinematic case there is the temporal evolution of the rupture
considered as well. So there are additional parameters: (1) the rupture nucleation point from which the rupture originates and propagates accross the fault
following the eikonal equation, (2) the slip-duration and the rupture velocity accross each sub-patch. Each sub-patch is considered to be active only once.
Optimizing for the rupture nucleation point makes the problem non-linear.

The finite fault optimization in beat is considered to be a follow-up step of the geometry optimization for a RectangularSource. Which is why first, a new project directory to solve for the geometry of a RectangularSource has to be created. If the reader has setup such a problem already and finished the optimization for a the geometry the next command can be skipped.::

    beat init FFIproject <date> --datatypes='seismic' --source_type='RectangularSource' --n_sources=1

If an optimization for the geometry of another source has been done or setup (e.g. MTSource), one can clone this project folder and replace the source object. This saves
time for specification of the inputs. How to setup the configurations for a "geometry" optimization is discussed
`here <https://hvasbath.github.io/beat/examples.html#regional-full-moment-tensor>`__ exemplary on a MomentTensor for regional seismic data.
The "source_type" argument will replace any existing source with the specified source for the new project. With the next project we replace the old source with a RectangularSource.::

    beat clone MTproject FFIproject --datatypes='seismic' --source_type='RectangularSource' --copy_data

Now the Green's Functions store(s) have to be calculated for the "geometry" problem if not done so yet. Instructions on this and what to keep in mind are given `here <https://hvasbath.github.io/beat/examples.html#calculate-greens-functions>`__. For illustration, the user might have done a MomentTensor optimization already on teleseismic data using Green's Functions depth and distance sampling of 1km with 1Hz sampling. This may be accurate enough for this type of optimization, however for a finite fault optimization the aim is to resolve details of the rupture propagation and the slip distribution. So the setup parameters of the "geometry" Green's Functions would need to be changed to higher resolution. A depth and distance sampling of 250m and 4Hz sample rate might be precise enough, if waveforms up to 1Hz are to be used in the optimization. Of course, these parameters depend on the problem setup and have to be adjusted individually for each problem!

If the Green's Functions for the "geometry" have been calculated previously with sufficient accuracy one can continue initialysing the configuration file for the finite fault optimization.::

    beat init FFIproject --mode='ffi' --datatypes='seismic'

This will load the parameters from the "geometry" problem and import them to the "ffi" setup. The configuration file for the "ffi" mode is called "config_ffi.yaml" and should be in the same directory as the "config_geometry.yaml". The parameters that are different in the "ffi" mode are under the "seismic_config.gf_config" of the mentioned configuration file.::

    gf_config: !beat.SeismicLinearGFConfig
      store_superdir: ./
      reference_model_idx: 0
      n_variations: [0, 1]
      earth_model_name: local
      nworkers: 3
      reference_sources:
      - !beat.sources.RectangularSource
        lat: 50.410785
        lon: -150.305465
        elevation: 0.0
        depth: 1000.0
        time: 1970-01-01 00:00:00
        stf: !pf.HalfSinusoidSTF
          duration: 15.0
          anchor: 0.0
        stf_mode: post
        magnitude: 6.0
        strike: 90.0
        dip: 67.5
        rake: 0.0
        width: 5000.0
        length: 10000.0
        slip: 4.05
        opening: 0.0
      patch_width: 2.5
      patch_length: 2.5
      extension_width: 0.0
      extension_length: 0.1
      sample_rate: 10.0
      reference_location: !beat.heart.ReferenceLocation
        lat: 50.0
        lon: -100.0
        elevation: 0.0
        depth: 0.0
        station: Waskahigan_broadband2
      duration_sampling: 1.0
      starttime_sampling: 1.0

In the next step again Green's Functions have to be calculated. What? Again? That's right! This time the geometry of the source needs to be specified. This is defined under the "reference_sources" attribute (see above). The distance units are [m], the angles [deg] and the slip [m]. If an optimization for these "geometry" parameters has been completed, the maximum likelihood result may be imported
with.::

    beat import FFIproject --results --datatypes='seismic' --mode='ffi'

If not, the parameters would need to be adjusted manually based on a-priori information from structural geology, literature or ...
Additionally, the discretization of the subpatches along this reference fault has to be set. The parameters "patch_width" and "patch_length" [km] determine these. So far only square patches are supported. "extension_width" and "extension_length" determine by how much the refernce fault is extended in EACH direction. If this would result in a fault that cuts the surface the intersection with the surface at zero depth is used. Example: 0.1 means that the fault is extended by 10% of its with/length value in each direction and 0. means no extension.

The "store_superdir" to the "geometry" Green's Functions needs to be correct and the "sample_rate" needs to be set likely higher than from the "geometry" setup.
The last two parameters are "duration_sampling" and "starttime_sampling" for the sampling of the source-time-function (STF) for each patch. For efficiency during sampling the STF is convolved for each source patch with the synthetic seismogram. The upper and lower bound for the STF duration and the STF (rupture) starttimes are determined by the optimization parameters in the "priors" under "problem_config".::

    priors:
      durations: !beat.heart.Parameter
        name: durations
        form: Uniform
        lower: [0.5]
        upper: [15.5]
        testvalue: [10.0]
      nucleation_dip: !beat.heart.Parameter
        name: nucleation_dip
        form: Uniform
        lower: [0.0]
        upper: [7.0]
        testvalue: [3.5]
      nucleation_strike: !beat.heart.Parameter
        name: nucleation_strike
        form: Uniform
        lower: [0.0]
        upper: [10.0]
        testvalue: [5.0]
      uparr: !beat.heart.Parameter
        name: uparr
        form: Uniform
        lower: [-0.3]
        upper: [6.0]
        testvalue: [2.85]
      uperp: !beat.heart.Parameter
        name: uperp
        form: Uniform
        lower: [-0.3]
        upper: [4.0]
        testvalue: [1.85]
      velocities: !beat.heart.Parameter
        name: velocities
        form: Uniform
        lower: [0.5]
        upper: [4.2]
        testvalue: [2.35]

For this example the synthetic seismograms ranging from an STF with a slip-duration of 0.5s up to 15.5s with a sampling of 1s would be calculated (0.5, 1.5, 2.5).
The sampling has to be consistent with the start and end durations. For example a duration lower: 0.5, duration upper: 3., with a sampling of 0.4 would result in an error as the sampling steps would be: 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9 but 3. is not included.
The "velocities" parameter is referring to the rupture velocity, which is often considered to be propagating with S-wave velocity. Depending on the velocity model that has been used during the setup of the "geometry" Green's Functions these parameter bounds may be adjusted.

With the following command the reference fault is set up and discretized into patches.::

    beat build_gfs FFIproject --mode='ffi' --datatypes='seismic'

The output might look like this::

    ffi          - INFO     Discretizing seismic source(s)
    ffi          - INFO     uparr slip component
    sources      - INFO     Fault extended to length=12500.000000, width=5000.000000!
    ffi          - INFO     Extended fault(s):
     --- !beat.sources.RectangularSource
    lat: 50.410785
    lon: -150.305465
    elevation: 0.0
    depth: 1000.0
    time: 1970-01-01 00:00:00
    stf: !pf.HalfSinusoidSTF
      duration: 15.0
      anchor: 0.0
    stf_mode: post
    magnitude: 6.0
    strike: 90.0
    dip: 67.5
    rake: 0.0
    width: 5000.0
    length: 12500.0
    slip: 1.0
    opening: 0.0

    ffi          - INFO     uperp slip component
    sources      - INFO     Fault extended to length=12500.000000, width=5000.000000!
    ffi          - INFO     Extended fault(s):
     --- !beat.sources.RectangularSource
    lat: 50.410785
    lon: -150.305465
    elevation: 0.0
    depth: 1000.0
    time: 1970-01-01 00:00:00
    stf: !pf.HalfSinusoidSTF
      duration: 15.0
      anchor: 0.0
    stf_mode: post
    magnitude: 6.0
    strike: 90.0
    dip: 67.5
    rake: -90.0
    width: 5000.0
    length: 12500.0
    slip: 1.0
    opening: 0.0

    beat         - INFO     Storing discretized fault geometry to: /home/vasyurhm/BEATS/Waskahigan2Rect/ffi/linear_gfs/fault_geometry.pkl
    beat         - INFO     Updating problem_config:
    beat         - INFO
    Complex Fault Geometry
    number of subfaults: 1
    number of patches: 10

This shows the new parameters of the extended reference source. The "width" and "length" are rounded to full mutliples of the "patch_length" and "patch_width" parameters.
Also we see here the rake directions of the slip parallel and slip perpendicular directions.
The hypocentral location bounds have been adjusted to be within the bounds of the extended fault dimensions! To allow for potential rupture nucleation all along the reference fault in the example, the priors of "nucleation_strike" and "nucleation_dip" were set to be between (0, 12.5)[km] and (0,5)[km], respectively! Of course, the bounds may be set manually to custom values within the fault dimensions!

Finally, we need to pay attention to the "waveforms" under "seismic_config".::

    waveforms:
    - !beat.WaveformFitConfig
      include: true
      name: any_P
      channels: [Z]
      filterer: !beat.heart.Filter
        lower_corner: 0.001
        upper_corner: 4.0
        order: 4
      distances: [0.0, 5.0]
      interpolation: multilinear
      arrival_taper: !beat.heart.ArrivalTaper
        a: -15.0
        b: -10.0
        c: 30.0
        d: 40.0

"Name" specifies the seismic phase; "channels" the component of the observations to include, "filterer" the bandpass filter the synthetics are filtered to; "distances" the receiver-source interval of receivers to include; and the "arrival_taper" the part of the synthetics with respect to the theoretical arrival time (from ray-tracing).

Once satisfied with the set-up the "nworkers" parameter in "config_ffi.yaml" may be set to make use of parallel calculation of the Green's Functions. Depending on the specifications the amount of Green's Functions to be calculated may-be significant. The resulting matrix will be of size: number_receivers * number_patches * number_durations * number_starttimes * number_trace_samples * float64 (8bytes).

The calculation of the Green's Functions, which may take some hours (depending on the setup and computer hardware) may be started with::

    beat build_gfs FFIproject --mode='ffi' --datatypes='seismic' --execute

For visual inspection of the resulting seismic traces in the "snuffler" waveform browser::

    beat check FFIproject --what='library' --datatypes='seismic' --mode='ffi'

This will load the seismic traces for the first receiver, for all patches, durations, starttimes.

  .. image:: _static/linear_gf_library.png

Here we see the slip parallel traces for patch 0, starttime of 11s (after the hypocentral source time) and slip durations(tau) of 1.5 and 10.5[s].

