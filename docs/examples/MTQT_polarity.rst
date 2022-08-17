Example 1: Regional Full Moment Tensor
--------------------------------------
Clone project
^^^^^^^^^^^^^
This setup is comprised of 25 seismic stations in a distance range of 3 to 162 km with respect to a reference event occurred on 2021-03-11 with the local magnitude, 1.6.
We will explore the solution space of a double couple Moment Tensor with the Tape and Tape 2015 parameterisation, the MTQTSource [TapeTape2015]_.
To copy the example setup (including the data) to a directory outside of the package source directory, please edit the *model path* (referred to as $beat_models from now on) and execute::

    cd /path/to/beat/data/examples/
    beat clone MTQT_polarity /'model path'/MTQT_polarity --copy_data --datatypes=polarity

This will create a BEAT project directory named *MTQT_polarity* with a configuration file *config_geometry.yaml* and a file with information of the seismic stations *stations.txt*.
This directory is going to be referred to as *$project_directory* in the following.


Calculate Greens Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^
For the inference of the moment tensor using polarity data the distances and takeoff-angles of rays from the seismic event towards stations need to be calculated. If the event location is fixed
this needs to be done only once! BEAT supports the estimation of the location of the event, which requires repeated ray tracing. In order to avoid repeated ray-tracing, we pre-calculate look-up interpolation tables of the tabeoff-angles
based on a grid of potential source depths and distances towards the stations. These are stored in a database, which we refer to as Green's Functions (GFs) in the following.

Please open $project_directory/config_geometry.yaml with any text editor (e.g. vi) and search for *store_superdir*. Here, it is written for now ./MTQT_polarity, which is an example path to the directory of Greens Functions.
This path needs to be replaced with the path to where the GFs are supposed to be stored on your computer. This directory is referred to as the $GF_path in the rest of the text. It is strongly recommended to use a separate directory apart from the beat source directory and the $project_directory as the GF databases can become very large, depending on the problem! For real examples, the GF databases may require up to several Gigabyte of free disc space. For our example the database that we are going to create is only around 30 Megabyte.::

    cd $beat_models
    beat build_gfs MTQT_polarity --datatypes='polarity'

This will create an empty Greens Function store named PolarityTest_local_10.000Hz_0 in the $GF_path. Under $GF_path/polarity_local_10.000Hz_0/config it is recommended to cross-check again the velocity model and the specifications of the store (open with any texteditor).
Dependend on the case study there are crucial parameters that often need to be changed from the default values: the spatial grid dimensions.

In the $project_path/config_geometry.yaml under polarity config we find the gf_config, which holds the major parameters for GF calculation::

  gf_config: !beat.PolarityGFConfig
    store_superdir: ./MTQT_polarity
    reference_model_idx: 0
    n_variations:
    - 0
    - 1
    earth_model_name: local
    nworkers: 20
    use_crust2: false
    replace_water: false
    custom_velocity_model: |2
          0.             3.406          2.009          2.215        331.1          147.3
          1.9            3.406          2.009          2.215        331.1          147.3
          1.9            5.545          3.295          2.609        286.5          127.5
          8.             5.545          3.295          2.609        286.5          127.5
          8.             6.271          3.74           2.781        471.7          210.1
         21.             6.271          3.74           2.781        471.7          210.1
         21.             6.407          3.767          2.822        900.           401.6
         40.             6.407          3.767          2.822        900.           401.6
    source_depth_min: 0.1
    source_depth_max: 7.5
    source_depth_spacing: 0.1
    source_distance_radius: 250.0
    source_distance_spacing: 0.1
    error_depth: 0.1
    error_velocities: 0.1
    depth_limit_variation: 600.0
    reference_location: !beat.heart.ReferenceLocation
      lat: 55.89310323984567
      lon: -120.38565188644934
      depth: 1.65
      station: polarity
    sample_rate: 10.0

Here we see that instead of a global velocity model a *custom_velocity_model* is going to be used for all the stations.
Below are the grid definitions of the GFs. The distance grid is accordingly extending up to 250 km.
These grid sampling parameters are of major importance for the accuracy of interpolated takeoff-angels. For specific event-station setups the *distance_spacing* and *depth_spacing* parameters may not be accurate enough. In this case BEAT will warn the user and will ask the user to
lower these values.

The *nworkers* variable defines the number of CPUs to use in parallel for the GF calculations, however, in this case only serial calculation is implemented yet.
For our use-case the grid specifications are fine for now. In this case the takeoff-angles are going to be calculated for the P body waves. 
Now the store configuration files have to be updated and the . As we created them before we need to overwrite them! We can do this with the --force option; --execute will start the actual calculation.::

    beat build_gfs MTQT_polarity --datatypes='polarity' --force --execute

Now we can inspect the calculated takeoff-angle table ::

  cd $store_superdir
  fomosto satview polarity_local_10.000Hz_0 any_P

.. image:: ../_static/example8/takeoff_angles_table.png

The top plot shows depth vs distance and the respective takeoff-angle in color. The black boxes are adaptively calculated based on the gradient of takeoff-angles, where grid points falling into one box have the same takeoff-angles.
Thus, we see that at close distances we have small boxes i.e. rapidly changing takeoff-angles, which is not the case for larger distances. Just at rays close to velocity model layer changes these become finer again.

The lower plot shows the takeoff-angle at the depth of 3.8km for all the distances, i.e. a horizontal profile through the top plot.

We can also plot the station map with::

  beat plot MTQT_polarity station_map

.. image:: ../_static/example8/station_map_polarity.png


Optimization setup (PolarityConfig)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once we are confident that the GFs are reasonable we may continue to define the optimization specific setup variables.
In this case we want to optimize the whole polarity from first arrivals of P-wave (pwfarrival). Based on the selected seismic wave, channels should be set up. In our case, it's vertical component (Z).
The input will be a choice between text and a binary file. There is a flag in config named "binary_input" which can be used to enter data into BEAT through seismic binary file "seismic_data.pkl". If this flag is set to false, the input will be read from config file, just like in our case:

  stations_polarities:
  - BCH1A -1.0
  - BCH2A -1.0
  - MONT1 -1.0
  - MONT2 1.0
  - MONT3 1.0
  - MONT7 -1.0
  - MONT8 -1.0
  - MONT9 -1.0
  - MONTA -1.0
  - BMTB -1.0
  - NBC4 1.0
  - NBC7 1.0
  - NBC8 -1.0
  - BDMTA -1.0
  - FAIRA 1.0
  - WTMTA -1.0
  - MG01 1.0
  - MG03 -1.0
  - MG05 -1.0
  - MG07 -1.0
  - MG08 1.0
  - MG09 1.0
  - MG10 -1.0
  - MG11 -1.0

This list includes station names and polarities. There will be no *blacklist* for text-type input. Besides, station information like coordinate specification, azimuths, and distances will be imported into BEAT from "seismic_data.pkl" file.

Finally, we need to check *priors* and *hyperparameters*:

  hyperparameters:
    h_any_P_pol_Z: !beat.heart.Parameter
      name: h_any_P_pol_Z
      form: Uniform
      lower:
      - -5.0
      upper:
      - 8.0
      testvalue:
      - 1.5
  priors:
    depth: !beat.heart.Parameter
      name: depth
      form: Uniform
      lower:
      - 0.5
      upper:
      - 6.0
      testvalue:
      - 1.0
    duration: !beat.heart.Parameter
      name: duration
      form: Uniform
      lower:
      - 0.0
      upper:
      - 1.0
      testvalue:
      - 0.2
    east_shift: !beat.heart.Parameter
      name: east_shift
      form: Uniform
      lower:
      - -5.0
      upper:
      - 5.0
      testvalue:
      - -4.0
    h: !beat.heart.Parameter
      name: h
      form: Uniform
      lower:
      - 0.0
      upper:
      - 1.0
      testvalue:
      - 0.2
    kappa: !beat.heart.Parameter
      name: kappa
      form: Uniform
      lower:
      - 0.0
      upper:
      - 6.283185307179586
      testvalue:
      - 1.2566370614359172
    magnitude: !beat.heart.Parameter
      name: magnitude
      form: Uniform
      lower:
      - 1.0
      upper:
      - 2.5
      testvalue:
      - 2.0
    v: !beat.heart.Parameter
      name: v
      form: Uniform
      lower:
      - -0.3333333333333333
      upper:
      - 0.3333333333333333
      testvalue:
      - -0.26666666666666666
    w: !beat.heart.Parameter
      name: w
      form: Uniform
      lower:
      - -1.1780972450961724
      upper:
      - 1.1780972450961724
      testvalue:
      - 0.0
    north_shift: !beat.heart.Parameter
      name: north_shift
      form: Uniform
      lower:
      - -5.0
      upper:
      - 5.0
      testvalue:
      - -4.0
    peak_ratio: !beat.heart.Parameter
      name: peak_ratio
      form: Uniform
      lower:
      - 0.0
      upper:
      - 0.0
      testvalue:
      - 0.0
    sigma: !beat.heart.Parameter
      name: sigma
      form: Uniform
      lower:
      - -1.5707963267948966
      upper:
      - 1.5707963267948966
      testvalue:
      - -1.2566370614359172
    time: !beat.heart.Parameter
      name: time
      form: Uniform
      lower:
      - -3.0
      upper:
      - 3.0
      testvalue:
      - -2.4

Based on the *problem_config* (source specification) we selected for our inversion:

problem_config: !beat.ProblemConfig
  mode: geometry
  source_type: MTQTSource
  stf_type: Triangular
  n_sources: 1
  datatypes:
  - polarity

we specify priors. In our case, we consider MTQTSource, then we need set up h, kappa, sigma, w, and v source parameters (Tape & Tape 2015). There are some common source parameters between different type of sources such as east_shift, north_shift, duration, etc we need to adjust with respect to our specific problem and case. 

Now that we checked the optimization setup we are good to go.


Sample the solution space
^^^^^^^^^^^^^^^^^^^^^^^^^

Firstly, we fix the source parameters to some random value and only optimize for the noise scaling or hyperparameters (HPs).
The configuration of the hyper parameter sampling, is determined by the hyper_sampler_config parameters.::

    hyper_sampler_config: !beat.SamplerConfig
      name: Metropolis
      backend: csv
      progressbar: true
      buffer_size: 5000
      buffer_thinning: 1
      parameters: !beat.MetropolisConfig
        tune_interval: 50
        proposal_dist: Normal
        check_bnd: true
        rm_flag: false
        n_jobs: 1
        n_steps: 25000
        n_chains: 20
        thin: 5
        burn: 0.5

Here we use an adaptive Metropolis algorithm to sample the solution space.
How many different random source parameters are chosen and how often the sampling is repeated is controlled by *n_chains* (default:20).
In case there are several CPUs available the *n_jobs* parameter determines how many processes (Markov Chains (MCs)) are sampled in parallel.
Each MC will contain 25k samples (*n_steps*) and every 50 samples the step-size will be adjusted (*tune_interval*).
You may want to increase that now! To start the sampling please run ::

    beat sample MTQT_polarity --hypers

This reduces the initial search space from 40 orders of magnitude to usually 5 to 10 orders. Checking the $project_directory/config_geometry.yaml,
the HPs parameter bounds show something like::

  hyperparameters:
    h_any_P_pol_Z: !beat.heart.Parameter
      name: h_any_P_pol_Z
      form: Uniform
      lower:
      - -5.0
      upper:
      - 8.0
      testvalue:
      - 1.5


Now that we have an initial guess on the hyperparameters we can run the optimization using the default sampling algorithm, a Sequential Monte Carlo sampler.
The sampler can effectively exploit the parallel architecture of nowadays computers. The *n_jobs* number should be set to as many CPUs as possible in the configuration file.::

sampler_config: !beat.SamplerConfig
  name: SMC
  backend: csv
  progressbar: false
  buffer_size: 1000
  buffer_thinning: 10
  parameters: !beat.SMCConfig
    tune_interval: 50
    check_bnd: true
    rm_flag: true
    n_jobs: 4
    n_steps: 200
    n_chains: 300
    coef_variation: 1.0
    stage: 0
    proposal_dist: MultivariateCauchy
    update_covariances: false

.. note:: *n_chains* divided by *n_jobs* MUST yield a *Integer* number! An error is going to be thrown if this is not the case!

Here we use 4 cpus (n_jobs) - you can change this according to your systems specifications.
Finally, we sample the solution space with::

    beat sample MTQT_polarity

.. note:: The reader might have noticed the two different *backends* that have been specified in the *SamplerConfigs*, "csv" and "bin". `Here <https://hvasbath.github.io/beat/getting_started/backends.html#sampling-backends>`__ we refer to the backend section that describe these further.


Summarize the results
^^^^^^^^^^^^^^^^^^^^^
The sampled chain results of the SMC sampler are stored in seperate files and have to be summarized.

.. note::
    Only for MomentTensor MTSource: The moment tensor components have to be normalized again with respect to the magnitude.

To summarize all the stages of the sampler please run the summarize command.::

    beat summarize MTQT_polarity


If the final stage is included in the stages to be summarized also a summary file with the posterior quantiles will be created.
If you check the summary.txt file (path then also printed to the screen)::

    vi $project_directory/geometry/summary.txt

For example for the first 4 entries (mee, med, posterior like-lihood, north-shift), the posterior pdf quantiles show::

                             mean        sd  mc_error       hpd_2.5      hpd_97.5
    mee__0             -0.756400  0.001749  0.000087     -0.759660     -0.752939
    med__0             -0.256697  0.000531  0.000024     -0.257759     -0.255713
    like__0         89855.787301  2.742033  0.155631  89849.756559  89859.893765
    north_shift__0     19.989398  0.010010  0.000496     19.970455     20.008629

As this is a synthetic case with only little noise it is not particularly surprising to get such steeply peaked distributions.


Plotting
^^^^^^^^
To see results of source inversion based on polarity, we need to plot beachball with polarities on it. 

    beat plot MTQT_polarity fuzzy_beachball --nensemble=200
    
nensemble arguement would add uncertainty to the plot.

The following command produces a '.png' file with the final posterior distribution. In the $beat_models run::

    beat plot MTQT_polarity stage_posteriors --reference --stage_number=-1 --format='png'

It may look like this.

 .. image:: ../_static/example1/FullMT_stage_-1_max_variance.png

The vertical black lines are the true values and the vertical red lines are the maximum likelihood values.
We see that the true solution is not comprised within the marginals of all parameters. This may have several reasons. In the next section we will discuss and investigate the influence of the noise characteristics.

To get an image of parameter correlations (including the true reference value in red) of moment tensor components, the location and the magnitude. In the $beat_models run::

    beat plot MTQT_polarity correlation_hist --reference --stage_number=-1 --format='png' --varnames='mee, med, mdd, mnn, mnd, mne, north_shift, east_shift, magnitude'

This will show an image like that.

 .. image:: ../_static/example1/FullMT_corr_hist_ref_variance.png

This shows 2d kernel density estimates (kde) and histograms of the specified model parameters. The darker the 2d kde the higher the probability of the model parameter.
The red dot and the vertical red lines show the true values of the target source in the kde plots and histograms, respectively.

The *varnames* option may take any parameter that has been optimized for. For example one might als want to try --varnames='duration, time, magnitude, north_shift, east_shift'.
If it is not specified all sampled parameters are taken into account.


Clone setup into new project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we want to repeat the sampling with the noise structure set to *non-toeplitz*, but we want to keep the previous results
as well as the configuration files unchanged for keeping track of our work. So we can use again the clone function to clone
the current setup into a new directory.::

  beat clone MTQT_polarity MTQT_polarity_nont --copy_data --datatypes=polarity

References
^^^^^^^^^^
.. [TapeTape2015] A uniform parametrization of moment tensors. Geophysical Journal International, 202(3), 2074–2081. https://doi.org/10.1093/gji/ggv262
.. [Brillinger] Brillinger, D. R. and Udias, A. and Bolt, B. A., A probability model for regional focal mechanism solutions. Bulletin of the Seismological Society of America 1980: doi: https://doi.org/10.1785/BSSA0700010149
