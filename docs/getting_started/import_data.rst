Import Data
-----------
This is the step to import the user data into the program format and setup.


geodetic data
^^^^^^^^^^^^^

InSAR
=====
To use static displacement InSAR measurements you need to prepare your data first with `kite <https://github.com/pyrocko/kite>`__.
Kite handles displacement data from a variety of formats, such as e.g. GMTSAR, ISCE, ROIPAC and GAMMA. After importing the data into kite you
should consider to subsample it and to calculate the data-error-variance-covariance as described in the `kite documentation <https://pyrocko.org/kite/docs/current/>`__.
Once you are satisfied with your specifications please store the kite scenes in its native format as "numpy-npz containers".

In the $project_dir you find the config_geometry.yaml, where the geodetic_config variable 'datadir' points to the location where the data are stored.
Under the 'names' variable, the names of the files of interest have to be entered (without the .npz and .yml suffixes).
Afterwards, the following command has to be executed to import the data::

  beat import $project_dir --datatypes=geodetic --geodetic_format=kite

The data are now accessible to beat as the file geodetic_data.pkl. In case it turns out the pre-processing (subsampling, covariance estimation) had to be repeated, the existing 'geodetic_data.pkl' file can be overwritten by adding the --force option to the import command above.

GNSS
====
The supported format for GNSS data is an ASCII file of the following format::

  #SUMMARY VELOCITY ESTIMATES FROM GLOBK
  # Long.  Lat.   E & N Rate      E & N Adj.      E & N   +-    RHO    H Rate   H adj.  +-   SITE
  # (deg)  (deg)  (mm/yr)         (mm/yr)         (mm/yr)              (mm/yr)
   30.4    40.6   -14.98   -2.12  -14.98   -2.12  1.34    1.62 -0.042  -2.16   -2.16    5.77 DOGG_FRM
   31.6    42.7   -13.92    2.11  -13.92    2.11  1.98    2.44 -0.012   6.71    6.71    8.91 CATT_FRM
   32.5    41.5    -5.43   -4.25   -5.43   -4.25  1.35    1.67 -0.013  -5.34   -5.34    5.97 COOW_FRM

From that file the following columns are imported: Longitude, Latitude, velocity east component, velocity north component, velocity vertical component, standard-deviaion east component, standard-deviaion north component, standard-deviaion vertical component and site label, which is composed of the station name and the network name.
The units for the location and the measurements are [decimal deg] and [mm/yr], respectively.

.. note:: This is the native GAMMIT-GLOBK output file, and the number of header-lines (first three commented lines) is important. In case some of those lines are missing the first entries might be skipped during import!

The following command has to be executed to import the data::

  beat import $project_dir --datatypes=geodetic --geodetic_format=ascii

GNSS and InSAR
==============
The dataformats specified above apply.
To import both, the GNSS and InSAR data into a beat project run::

  beat import $project_dir --datatypes=geodetic --geodetic_format=ascii,kite

seismic data
^^^^^^^^^^^^
For the import and aquistion of seismic data for beat exist several options. The command beatdown can be used to download a
dataset from available FDSN services. Alternatively, existing files from any custom source may be converted by using the pyrocko framework.

beatdown
========
The command line tool beatdown downloads waveforms from all available FDSN web services and prepares them for BEAT,
including restituting the waveforms to displacements and rotating them into the R,T,Z or E, N,Z coordinate
systems.

The beatdown command for downloading FDSN data can be executed in different formats, e.g. by giving an event time or an event name.
It will download all wanted data in a given radius around the origin. For a complete list of input options
please use::

  beatdown --help

An example line to download and prepare the data for the 2009 L'Aquila earthquake would be::

  beatdown /path/to/project_directory "2009-04-06 01:32:39" 1000. 0.001 5. Laquila

This command downloads the available data for the event at time 2009-04-06 01:32:39 in a
radius of 1000 km restitutes the traces to frequencies between 0.001 and 5. Hz and saves them in the folder
/path/to/project_directory/data/events/Laquila. Additionally it creates a seismic_data.pkl into the "project_directory", which will
be used by BEAT.

It may be desired to automatically pre-selected a subset of stations from all available data
based on data quality and separation of stations. The option --nstations-wanted enables such a station
weeding and tries to find a suitable subset of stations close to the number provided. The actual resulting
station number might vary based on station distribution and quality. For the above
example we might want to use around 60 stations, so the command line for that would look like::

  beatdown /path/to/home_directory "2009-04-06 01:32:39" 1000. 0.001 5. Laquila --nstations-wanted=60


To convert a distance in degree, e.g. 30 and 90 degrees for minimum and maximum data retrieval
radii (rmin and rmax), you can use pyrockos cake function inbuilt converter factor d2m.
The conversion than could look like this to retrieve rmin and rmax radii in km::

  from pyrocko import cake
  km = 1000.
  rmin = 30.*cake.d2m/km
  rmax = 90.*cake.d2m/km


Data import
===========

The output of `autokiwi <https://github.com/emolch/kiwi>`__ is supported for automatic import of seismic data.

To see a list of the supported data types ($ending) please see: `Trace Handling <https://pyrocko.org/docs/current/library/examples/trace_handling.html>`__
or type.::

    beat import --help

The traces should be named in the format 'network.station..channel.$ending'
In addition to these an ascii text file with the station information is needed of the format::

    #network_name.station_name.location_name latitude[deg] longitude[deg] elevation[m] depth[m]
    IU.TSUM.10            -19.20220       17.58380         1260.0            0.0
      BHE             90              0              1   # channel name azimuth[deg] dip[deg] gain \n
      BHN              0              0              1
      BHZ              0            -90              1
    IU.RCBR.00             -5.82740      -35.90140          291.0          109.0
      BH1             48              0              1
      BH2            138              0              1
      BHZ              0            -90              1
    ...

To ease the creation of this textfile we refer the user to investigate the pyrocko module: model (Function: dump_stations).


Custom Data import
==================
To get other types of data imported the user will have to do some programming.

The following remarks are just bits and pieces that may be followed to write a script to bring the data into the necessary format.

The seismic data may be saved using the package "pickle" as a file "seismic_data.pkl" containing a list of 2 lists: 1. list of "pyrocko.trace.Trace" objects alternating for (R T Z) rotated traces. 2. list of "pyrocko.model.Station" objects in the same order like the data traces.

Pyrocko supports the import of various data formats and all the necessary tools to remove the instrument response and to convert the traces to displacement.
How to do this based on some examples is shown `here <https://pyrocko.org/docs/current/library/examples/trace_handling.html#restitute-to-displacement-using-poles-and-zeros>`__ webpage.

For import from obspy you can checkout the `obspy_compat <https://pyrocko.org/docs/current/library/reference/obspy_compat.html#pyrocko.obspy_compat.plant>`__
pyrocko module to convert your obspy data into pyrocko data and obspy inventories to pyrocko stations.
Once you have done this the standard pyrocko traces will need to be converted to beat trace objects, this is done simply, assuming that "traces"
is a list of pyrocko trace objects, by::

    from beat import heart
    traces_beat = []
    for tr in traces:
        tr_beat= heart.SeismicDataset.from_pyrocko_trace(tr)
        traces_beat.append(tr_beat)

Once a list of traces and station objects exists it may be exported to the project directory (here path from example)::

    from beat import utility

    seismic_outpath='/home/vasyurhm/BEATS/LandersEQ/seismic_data.pkl'
    utility.dump_objects(seismic_outpath, outlist=[stations, data_traces])
