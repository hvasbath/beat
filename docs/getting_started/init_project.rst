Initialise a new Modeling Project
---------------------------------

Each modeling project is initiated with the "beat init" command. There are many options that define the type of optimization, datatypes to include, sampling algorithm to use, number of sources, velocity model to use for the Greens Function calculations etc ...

For example to optimize for a Full Moment Tensor for the Landers EQ by using seismic data, with station dependent Greens Functions for P and S waves with the default sampling algorithm (Sequential Monte Carlo) run::

    beat init LandersEQ 1992-06-28 --datatypes='seismic' --individual_gfs --n_sources=1 --source_types=MTSource --min_mag=7

This will create project directory called LandersEQ in the current directory.
Within the directory you will see that there have been two files created:

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
To find a short explanation to each parameter and its format the reader is referred to the webpage of the `config <https://hvasbath.github.io/beat/_modules/config.html#SeismicConfig>`__ module.

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
      source_types: [MTSource]
      stf_type: HalfSinusoid
      n_sources: [1]
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
        filterer:
        - !beat.heart.Filter
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
        filterer:
        - !beat.heart.Filter
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


Initialize modeling project of an unlisted earthquake
-----------------------------------------------------
*Contributed by Carlos Herrera*

To create a customizable moment tensor project for an earthquake not included in any moment tensor
catalog, run::

    beat init newEQ --datatypes='seismic' --mode='geometry' --source_types='MTSource' --waveforms='any_P, any_S, slowest' --use_custom

This creates the folder “newEQ” with a *config_geometry.yaml* file inside. Some parameters should be
manually edited and filled up. These are some suggested initial changes in the configuration file:

 * **event: !pf.Event**: In this block, add manually the following earthquake parameters: lat, lon and time;
   optionally: depth, name, magnitude, and region.
 * **hyperparameters** and **seismic_config: !beat.SeismicConfig**: The *beat init* command in
   the example includes three types of waves for the modelling, which can be adjusted in the
   blocks under **waveforms: !beat.WaveformFitConfig**. By default, the vertical channel (Z) is used for the wave types.
   But radial (R) and transverse (T) channels can also be manually added. Also, the modeling can be done in either
   displacement (default) or velocity - please adjust the *quantity* argument accordingly.
   In this section, the source-station distance range (*distances*) is in degrees.
   Make sure the range does not extend beyond the limits of the chosen Green’s functions for modeling.
 * **gf_config: !beat.SeismicGFConfig**: This block is related to Green’s functions parameters. In
   this case, *beat init* was specified to use the option of custom Green’s functions, which need to be
   calculated just once.:

  1) Depending on the distances used for the project (regional or teleseismic), download and install
     independently the `QSEIS <https://git.pyrocko.org/pyrocko/fomosto-qseis/>`__ code and/or the `QSSP <https://git.pyrocko.org/pyrocko/fomosto-qssp/>`__ code.
  2) Edit the Green’s functions parameters in this block and then calculate the Green’s functions.
     General instructions and suggestions can be found `here <https://pyrocko.org/beat/docs/current/getting_started/custom_gf_store.html>`__.

 * **sampler_config: !beat.SamplerConfig** and **hyper_sampler_config: !beat.SamplerConfig**: Parameters
   in these blocks are related to the sampling method and can be edited depending on
   the user needs. In terms of calculation performance, the “bin” backend is considerably faster
   than “csv” (see `sampling backends <https://pyrocko.org/beat/docs/current/getting_started/backends.html>`__).
   Also, the *progressbar* can be optionally set to “false” for an additional performance improvement.

Waveform data for this modeling project can be downloaded using **beatdown**. Data can be selected by
using the earthquake’s origin time, location, and station distance range (check: **beatdown --help** ). A
simple example is found `here <https://pyrocko.org/beat/docs/current/getting_started/import_data.html>`__.
After considering these initial suggestions, follow the `tutorial <https://pyrocko.org/beat/docs/current/examples/FullMT_regional.html>`__ for detailed model parameter descriptions and instructions to run the estimation.
