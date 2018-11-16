
Scenario III: kinematic finite-fault optimization
-------------------------------------------------

It is a requirement to have Scenario I and II completed in order to follow the instructions and commands given in this scenario.
The data is the exact same from `Scenario I <https://hvasbath.github.io/beat/examples/Rectangular.html#>`__, where the overall geometry of the fault plane was estimated.
In `Scenario II <https://hvasbath.github.io/beat/examples/FFO_static.html#>`__ we solved for variable slip on the optimum fault gometry from I by using static InSAR data.
We will use the posterior marginals from Scenario II and use them as priors in this scenario. Here we will determine a kinematic variable slip distribution including rupture propagation for the L'aquila 2009 earthquake by using static InSAR data **jointly** with teleseismic displacement waveforms.

Please make sure that you are one level above the Laquila project folder (created earlier).::

  cd $beat_models_path


Clone config
^^^^^^^^^^^^
We want to use the setup that we used for Scenario II, but additionally we want to include the teleseismic data.
So we generate a new project folder Laquila_kinematic cloning the previous *config_ffo.yaml* from the *Laquila* project_directory with::

  beat clone Laquila Laquila_kinematic --mode=ffo --datatypes=geodetic,seismic

The new *config_ffo.yaml* will have an additional *seismic_config* and the *problem_config* includes priors for the kinematic rupture properties: velocities, durations, nucleation_strike, nucleation_dip, time as well as additional noise scalings for the seismic data in the hyperparameters.

Import results
^^^^^^^^^^^^^^
In this step we want to import the results from the previous two optimizations to the configuration file.
Firstly, we want to import the results from Scenario I(geometry optimization). But this time also the *seismic_config* is updated::

  beat import Laquila_kinematic --results=Laquila --mode=geometry --datatypes=geodetic,seismic

Now as we have two datatypes there are two *reference_sources* arguments in the config_ffo.yaml, one under *geodetic_config.gf_config* and another under *seismic_config.gf_config*. However, as you can see by the *yaml* coding (**id001**) these two are referring to the same object::

    ...
    gf_config: !beat.GeodeticLinearGFConfig
      store_superdir: /home/vasyurhm/GF/Laquila/
      reference_model_idx: 0
      n_variations: [0, 1]
      earth_model_name: ak135-f-continental.m
      nworkers: 4
      reference_sources:
      - &id001 !beat.sources.RectangularSource
        lat: 42.29
        lon: 13.35
        north_shift: 5542.073672733207
        east_shift: 10698.176839272524
        elevation: 0.0
        depth: 2926.702988951863
        time: 2009-04-06 01:32:49.190000
        stf: !pf.HalfSinusoidSTF
          duration: 0.0
          anchor: -1.0
        stf_mode: post
        strike: 144.48588106798306
        dip: 54.804317242125464
        rake: -114.58259929068664
        length: 12219.56636799338
        width: 9347.802691161543
        velocity: 3500.0
        slip: 0.5756726498300268
    ...
    gf_config: !beat.SeismicLinearGFConfig
      store_superdir: /home/vasyurhm/GF/Laquila/
      reference_model_idx: 0
      n_variations: [0, 1]
      earth_model_name: ak135-f-continental.m
      nworkers: 4
      reference_sources:
      - *id001
      patch_widths: [2.0]
      patch_lengths: [2.0]
      ...

Editing the *reference_sources* this way ensures consistent geometry for all datatypes (maybe in the future there are more to be added...).
Now please make sure that also the arguments *patch_widths* and *patch_lengths* as well as the *extension_widths* and *extension_lengths* are consistent!
The discretization at this point could be changed of course. However, this would then not allow to import the results from Scenario II(ffo optimization), which we want to do next::

  beat import Laquila_kinematic --results=Laquila --mode=ffo --datatypes=geodetic,seismic

You will notice now that the lower and upper bounds of the slip parameters *uparr* and *uperp* have been updated. Each sub-patch has indiviudal bounds where the index in the array refers to the patch number in the geometry and discretization figure shown in Scenario II. As a short recap here again.

.. image:: ../_static/scenario2/Laquila_FaultGeometry.png

.. note:: We could not reproduce the plot yet right away as we did not create the *fault_geometry.pkl* object yet.


Optimization setup
^^^^^^^^^^^^^^^^^^

TODO:  arrival taper, filterer, wavemaps

#check data and filter settings
beat check LaquilaJointPonlyUPDATE_wide_kin --what=traces



Calculate Green's Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Elementary GFs
==============
Now the Green's Functions store(s) have to be calculated again for the "geometry" problem with higher resolutions. Please remember `Scenario I <https://hvasbath.github.io/beat/examples/Rectangular.html#calculate-greens-functions>`__. There the optimization was run using Green's Functions depth and distance sampling of 4km with 0.5Hz sampling. This may be accurate enough for the *geometry* type of optimization, however, for a finite fault optimization the aim is to resolve details of the rupture propagation and the slip distribution. So the setup parameters of the "geometry" Green's Functions would need to be changed to higher resolution. In this case we want to use wavelengths of up to 0.5Hz ergo a depth and distance sampling of 1 km and 2Hz sample rate may be precise enough. Of course, these parameters depend on the problem setup and have to be adjusted individually for each problem! So please open the *Laquila/config_geometry.yaml* and edit the parameters accordingly.
Running this calculation will take a long time depending on the number of CPUs at hand. (With 25 CPUs the calculation took approximately 15Hrs)::

  beat build_gfs Laquila --datatypes='seismic' --execute

GF Library
==========
In the next step again Green's Functions have to be calculated. What? Again? That's right! Now they need to be calculated for the fixed fault geometry (remember Scenario II). Firstly, we create the discretized fault, this time for seismic and geodetic data::

  beat build_gfs Laquila_kinematic --datatypes=seismic,geodetic --mode=ffo

Parts of the output might look like::

    2018-11-03 15:28:00,164 - ffo.fault - INFO - Discretizing seismic source(s)
    2018-11-03 15:28:00,164 - ffo.fault - INFO - uparr slip component
    2018-11-03 15:28:00,164 - sources - INFO - Fault extended to length=22000.000000, width=22000.000000!
    2018-11-03 15:28:00,164 - sources - INFO - Fault would intersect surface! Setting top center to 0.!
    2018-11-03 15:28:00,165 - ffo.fault - INFO - Extended fault(s): 
     --- !beat.sources.RectangularSource
    lat: 42.29
    lon: 13.35
    north_shift: 6741.193145771676
    east_shift: 12378.404380741504
    elevation: 0.0
    depth: 0.0
    time: 2009-04-06 01:32:49.19
    stf: !pf.HalfSinusoidSTF
      duration: 0.0
      anchor: -1.0
    stf_mode: post
    strike: 144.48588106798306
    dip: 54.804317242125464
    rake: -114.58259929068664
    length: 22000.0
    width: 22000.0
    velocity: 3500.0
    slip: 1.0

Here we see that due to the extension parameters got extended to 22.0 times 22.0 [km].

For the geodetic GF *library* these from the Laquila project_directory could be also copied/linked, but for now we just recalculate it.::

  beat build_gfs Laquila_kinematic --datatypes=geodetic --execute --mode=ffo

For the seismic GF *library* we need to specify temporal parameter bounds of the source(s).
As the forward calculation has to be as fast as possible as much as possible has to be pre-calculated. Therefore, the effects of the source-time-function on the waveforms has to be included in the *library*. The consequence is that we have a *library* that has additional dimensions for the possible source *durations* (risetimes) of each patch.

These can be specified for **all** the patches under the *durations* prior. In order to keep the *library* at feasable sizes these values should be reasonable compared to the size of the earthquake. Example: For a magnitude Mw 6 earthquake we do not expect risetimes of 20s...
Please set the lower and upper bounds of the durations to 0. and 4. seconds, respectively.

Also we need to specify the bounds on the rupture velocities. The shear-wave velocity from the velocity model is a good proxy for that. So please set the lower and upper bounds on the velocities to 2.2 and 4.5 [km/s], respectively. These velocities are sampled for each patch individually and indirectly determine the rupture onset time of each patch depending on the hypocentral location (*nucleation_dip* and *nucleation_strike*). To assure causal rupture propagation starting from the hypocentre the Eikonal equation is solved each forward calculation, which then determines the rupture onset time on each patch [Minson2013]_.



TODO: hypocentral time (shift),



Finally, we are left with specifying the *duration_sampling* and *starttime_sampling* under the *seismic_config.gf_config*. These determine the steps taken between the upper and lower bounds for the *durations* and the discrete starttime-shifts.
Please set the *duration_sampling* to 0.25. As we are using GFs with 2Hz setting the *starttime_sampling* to full discrete time samples of 0.5 is reasonable.

.. note::A duration sampling of 0.25 with a lower bound at 0. and an upper bound at 1. will result in source-time-function (STF) convolutions with the base-seismogram (no STF) at durations of [0., 0.25, 0.5, 0.75, 1.]. (for *each* patch and station). 

The *interpolation* attribute determines the interpolation method that is used to interpolate the GFs at values in between the pre-calculated waveforms. Please use *multilinear* for higher-precission and *nearest_neighbor* if the calculation has to be fast.

Now we are ready to calculate the seismic GF *library*. Depending on the priors and the number of CPUs (*nworkers* you want to specify under the *seismic_config.gf_config*) this calculation may take from few minutes to hour(s).::

  beat build_gfs Laquila_kinematic --datatypes=seismic --execute --mode=ffo

.. warning:: The seismic GF *libraries* can become very fast very big if the prior bounds are set too wide. These matrixes (two, i.e. one for each slip-component) have to be able to fit in the memory of your computer during sampling.

Like for the geodetic GFs this will create two files for each GF *library* in the **$linear_gfs** directory:
 - *seismic_uparr_static_0.traces.npy* a numpy array containing the linear GFs
 - *seismic_uparr_static_0.yaml* a yaml file with the meta information

For visual inspection of the resulting seismic traces in the "snuffler" waveform browser::

    beat check Laquila_kinematic --what='library' --datatypes='seismic' --mode='ffo'

This will load the seismic traces for the first station (target), for all patches, durations and starttimes.

.. image:: ../_static/scenario3/uparr_library_gf.png

Here we see the slip parallel traces for patch 0, at starttime (t0) of -1s (after the hypocentral source time) and slip durations(tau) of 0. and 0.25[s].






beat sample LaquilaJointPonlyUPDATE_wide_kin --mode=ffo

-