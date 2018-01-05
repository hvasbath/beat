
Scenarios in beat/data/examples
-------------------------------
In the following tutorials we will use synthetic example data to get familiar with the basic functionality of BEAT.


Regional Full Moment Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This setup is comprised of 20 seismic stations that are randomly distributed within distances of 40 to 1000 km compared to a reference event.
To copy the scenario (including the data) to a directory outside of the package source directory please edit the target path and execute::

    beat clone FullMT /absolute/path/to/your/directory/FullMT --copy_data

This will create a BEAT project directory with a configuration file and some synthetic example data.
The station-event geometry determines the grid of Greens Functions (GFs) that will need to be calculated next.

In the geometry_config.yaml under: seismic_config gf_config store_superdir- the path to where the Greens Functions are supposed to be stored needs to be filled!
This directory is refered to as the $GF_path in the rest of the text.::

    beat build_gfs FullMT --datatypes='seismic'

This will create an empty Greens Function store named AqabaMT_ak135_1.000Hz_0 in the $GF_path. Under $GF_path/AqabaMT_ak135_1.000Hz_0/config it is always recommended to cross-check again the velocity model and the specificationos of the store.
Dependend on the case study there are crucial parameters that often need to be changed from the default values: the spatial grid dimensions, the sample rate and the wave phases (body waves and/or surface waves) to be calculated.

In the geometry_config.yaml under seismic config we find the gf_config, which holds the major parameters for GF calculation::

  gf_config: !beat.SeismicGFConfig
    store_superdir: /home/vasyurhm/BEATS/GF/
    reference_model_idx: 0
    n_variations: [0, 1]
    error_depth: 0.1
    error_velocities: 0.1
    depth_limit_variation: 600.0
    earth_model_name: ak135-f-average.m
    use_crust2: false
    replace_water: false
    custom_velocity_model: |2
          0.             5.51           3.1            2.6         1264.           600.
          7.2            5.51           3.1            2.6         1264.           600.
          7.2            6.23           3.6            2.8         1283.           600.
         21.64           6.23           3.6            2.8         1283.           600.
      mantle
         21.64           7.95           4.45           3.2         1449.           600.
    source_depth_min: 8.0
    source_depth_max: 8.0
    source_depth_spacing: 1.0
    source_distance_radius: 1000.0
    source_distance_spacing: 1.0
    nworkers: 4
    reference_location: !beat.heart.ReferenceLocation
      lat: 29.07
      lon: 34.73
      elevation: 0.0
      depth: 0.0
      station: AqabaMT
    code: qseis
    sample_rate: 1.0
    rm_gfs: true

Here we see that instead of the global velocity model ak135 a custom velocity model is going to be used for all the stations.
Below are the grid definitions of the GFs. In this example the source depth grid is limited to one depth (8 km) to reduce calculation time.
As we have stations up to a distance of 970km distance to the event, the distance grid is accordingly extending up to 1000km.
These grid sampling parameters as well as the sample rate are of major importance for the overall optimization. How to adjust these parameters
according to other case studies is described `here <https://pyrocko.org/docs/current/apps/fomosto/tutorial.html#considerations-for-real-world-applications>`__.

The 'n_workers' variable defines the number of CPUs to use in parallel for the GF calculations. As these calculations may become very expensive and time-consuming it is of advantage to use as many CPUs as available. To be still able to navigate in your Operating System without crashing the system it is good to leave one CPU work-less.

For our use-case these specifications are fine for now.

Under waveforms in the geometry_config.yaml there are the seismic phases defined the GFs are going to be calculated for.::

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

In this case the GFs are going to be calculated for the P body waves. We can add additional WaveformFitConfig(s) if we want to include more phases. Like in our case of a regional setup we would like to include surface waves. For the build GFs command only the existence of the WaveformFitConfig and the name is of importance and we can ignore the other parameters so far. So lets add this to the geometry_config.yaml file below the any_P WaveformFitConfig. Note: both entries have to have the same indentation!::

      - !beat.WaveformFitConfig
        include: true
        name: slowest
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

Now the store configuration files have to be updated, as they are existing we need to overwrite them! We can do this with the --force option.::

    beat build_gfs FullMT --datatypes='seismic' --force

Checking again the store config under $GF_path/AqabaMT_ak135_1.000Hz_0/config shows the phases that are going to be calculated::

    tabulated_phases:
    - !pf.TPDef
      id: any_P
      definition: p,P,p\,P\
    - !pf.TPDef
      id: slowest
      definition: '0.8'

Finally, we are good to start the GF calculations!

    beat build_gfs FullMT --datatypes='seismic' --force --execute

Depending on the number of CPUs that have been assigned to the job this may take few minutes.




