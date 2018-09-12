
Rectangular source, real geodetic displacement data (InSAR)
-----------------------------------------------------------
Clone project
^^^^^^^^^^^^^
The project consist of two static displacements data sets from the 06.04.2009 Mw6.3 L'Aquila earthquake. The data are InSAR displacement maps from ascending
and descending orbits. We will explore the parameter space of a Rectangular Source [Okada1985]_ for this earthquake.
The data has been pre-processed with `kite <https://github.com/pyrocko/kite>`__. For details on the use and data display we refer to tutorial on the website.

.. image:: _static/Static_asc.png

To copy the scenario (including the data) to a directory outside of the package source directory, please edit the 'model path' (referred to as $beat_models now on) and execute::

   cd /path/to/beat/data/examples/
   beat clone RectangularStatic /'model path'/RectangularStatic --copy_data

This will create a BEAT project directory named 'RectangularStatic' with a configuration file (config_geometry.yaml) and an example dataset from real Envisat InSAR data (geodetic_data.pkl).
This directory 'RectangularStatic' is going to be referred to as '$project_directory' in the following.


Calculate Greens Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^
We need to calculate a Greens function store (GF), as done in  the Regional Full Moment Tensor example. However in this case we will only to
calculate a store that holds static displacements. For this we will make use of the PSGRN/PSCMP backend.

Please open $project_directory/config_geometry.yaml with any text editor (e.g. vi) and check the line 144: store_superdir.
This path needs to be replaced with the path to where the GFs are supposed to be stored on your computer.
This directory is referred to as the $GF_path in the rest of the text. It is strongly recommended to use a separate directory
apart from the beat source directory. The statics Green's function stores are not very large, but can be used by several projects in the
future.

   cd $beat_models
   beat build_gfs RectangularStatic

This will create an empty Greens Function store named statics_ak135_0.000Hz_0 in the $GF_path. It will use the AK135 earth model[Kennet]_ in combination with CRUST2.0[Laske]_ for the shallow layers.


In the $project_path/config_geometry.yaml under geodetic_config we find the gf_config, which holds the major parameters for GF calculation::

 gf_config: !beat.GeodeticGFConfig
   store_superdir: $project_directory/
   reference_model_idx: 0
   n_variations: [0, 1]
   earth_model_name: ak135-f-average.m
   nworkers: 6
   use_crust2: true
   replace_water: true
   source_depth_min: 0.0
   source_depth_max: 35.0
   source_depth_spacing: 1.0
   source_distance_radius: 100.0
   source_distance_spacing: 1.
   error_depth: 0.1
   error_velocities: 0.1
   depth_limit_variation: 600.0
   code: psgrn
   sample_rate: 1.1574074074074073e-05
   sampling_interval: 1.0
   medium_depth_spacing: 1.0
   medium_distance_spacing: 1.0

Note that you need to change the variable 'store_superdir' to an **absolute path** to your $project_directory/.
You can also change the number of cores available to your system with the variable 'nworkers' to speed up the calculation of the GFs.
The GF grid spacing is important and can be modified in x,y direction with 'source_distance_spacing' and in depth with 'source_depth_spacing'.
The grid extent can be modified by 'source_distance_radius'. All the units are given in [km].
The GF parameters set for the 2009 L'Aquila static example are good for now. We now build the GF directory, where the GF config will
be further configurable.

For your own projects and needs you can also modify directly the GF velocity model and settings in the file $GF_path/statics_ak135_0.000Hz_0/config before exeuting the building in the next
step. For the 2009 L'Aquila static scenario, or after you are satisfied with you modification of the GF setup, we can next build the GF with:

   beat build_gfs $project_directory --force --execute

This will take some time, depending on how much cores you specified to execute in parallel at 'nworkers'. However, this only has to be done once and
the GF store can be reused for different scenarios if the velocity model does not differ between the different cases.

Optimization setup
^^^^^^^^^^^^^^^^^^
Before further setup we should check that the 'project_dir' variable in the main upper body of the $project_directory/config_geometry file is set correctly to your $project_directory/.
Please also take note of the 'event' variables, which are the GCMT source parameters for the 2009 L'Aquila earthquake in the `pyrocko <https://github.com/pyrocko/pyrocko>`__. event format.
The location and timing parameters of this event are used as the reference point in the setup of the local coordinate system.
We will explore the solution space of a Rectangular Source [Okada1985]_ in an elastic homogeneously layered halfspace. The parameters to explore are the sources east_shift, north_shift, depth, strike, rake, dip, length, width and slip.
The unit for slip is [m] and for all the other length measures (length, width, depth etc...) it is [km]. The angles (strike, dip and rake) are given in [degree].
Another option is to estimate an additional linear trend ('ramp' in InSAR terminology) to each dataset. This can be turned on and off with the variable 'fit_plane' in the geodetic_config section.
If you did so, the config_geometry.yaml has to be updated with the additional ramp parameters::

 beat update RectangularStatic --parameters="hierarchicals"

Often there the user has some apriori knowledge about the parameters of the Rectangular Source. These can be defined under the "priors" dictionary in the problem_config section.  
Here is an example::

   priors:
     rake: !beat.heart.Parameter
       name: rake
       form: Uniform
       lower: [-180.0]
       upper: [0.0]
       testvalue: [-110.0]

 .. Note: The "testvalue" has to be within the upper and lower bounds!

However, for the L'Aquila example we are now satisfied with the pre-set priors, in the config_geometry.yaml file. These are chosen with broad bounds around the reference solution, demonstrating a case where some prior knowledge is available. This allows for a faster search of the solution space.

The 'decimation_factor' variable controls how detailed the displacement from the source should be calculated. 
High numbers allow for faster calculation through each forward calculation, but the accuracy will be lower.
The sub variable 'geodetic' controls the decimation for the geodetic data only.
As the datasets for the L'Aquila earthquake example consist of subsampled datasets at a low resolution, we can set the decimation_factor to 7.


Sample the solution space
^^^^^^^^^^^^^^^^^^^^^^^^^
Please refer to the 'Sample the solution space section' of the `FullMT <https://hvasbath.github.io/beat/examples.html#sample-the-solution-space>`__ scenario for a more detailed description of the sampling and associated parameters.

Firstly, we only optimize for the noise scaling or hyperparameters (HPs)::

   beat sample $project_directory --hypers

Checking the $project_directory/config_geometry.yaml, the HPs parameter bounds show something like::

   hyperparameters:
   h_SAR: !beat.heart.Parameter
     name: h_SAR
     form: Uniform
     lower: [-1.0]
     upper: [5.0]
     testvalue: [2.0]

The 'n_jobs' number should be set to as many CPUs as the user can spare under the sampler_config. The number of sampled MarkovChains and the number of steps for each chain of the SMC sampler has been reduced for this example to allow for a fast result, at the cost of a more thorough exploration of the parameter space.
After the determination of the hyperparameter we can now start the sampling with::

   beat sample $project_directory

 .. note::  For more detailed search of the solution space please modify the parameters 'n_steps' and 'n_chains' for the SMC sampler in the $project_directory/config_geometry.yaml file to higher numbers. Depending on these specifications and the available hardware the sampling may take several hours.


Summarize and plotting
^^^^^^^^^^^^^^^^^^^^^^
After the sampling successfully finished, the final stage results can be summarized with::

 beat summarize $project_directory --stage_number=-1

After that several figures illustrating the results can be created.  For a comparison between data, synthetic displacements and residuals for the two InSAR tracks please run::

 beat plot $project_directory scene_fits

The plot should show something like this:
 .. image:: _static/Static_scene_fits.png


To plot the posterior marginal distributions of the source parameters, please run::

   beat plot $project_directory stage_posteriors --stage_number=-1


These plots are stored under your $project_directory folder under geometry/figures.
 .. image:: _static/Static_stage_-1_max.png

The solution should be comparable to results from published studies. E.g. [Walters2009]_.


References
^^^^^^^^^^

Kennet
Laske
Okada1985
Walters2009