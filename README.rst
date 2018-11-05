.. image:: docs/_static/LOGO_BEAT.png?raw=true
    :align: center
    :alt: BEAT logo

Bayesian Earthquake Analysis Tool
---------------------------------

Based on pyrocko, theano and pymc3

Documentation (under construction) can be found here:
https://hvasbath.github.io/beat/

This repository is still beta version and under development!
There might be future changes in the API that cause previous versions to break.

Tutorials
=========
Step by step points on how to use the tool for several scenarios can be found here:
`Scenarios <https://hvasbath.github.io/beat/examples.html#>`__

Data import
===========
Geodetic
^^^^^^^^
We recommend to prepare the SAR data (subsampling, data covariance estimation) using KITE (www.pyrocko.org).
kite supports import of ISCE, GAMMA, ROI_Pac and GMTSAR processed interferograms. BEAT then supports import of the native KITE format.

Seismic
^^^^^^^
To see a list of the supported data types please see: `Trace Handeling <https://pyrocko.org/docs/current/library/examples/trace_handling.html>`__
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

To ease the creation of this textfile we refer the user to investigate the pyrocko module: model (Function: dump_stations)

In addition to these, BEAT supports the output of the autokiwi package.

Work is in progress to support obspy saved stream and inventory files, as well as stationxml stay tuned ...

Alternatively the seismic data may be saved using the package "pickle" as a file "seismic_data.pkl"
containing a list of 2 lists:
1. list of "pyrocko.trace.Trace" objects alternating for (Z / T / R) rotated traces.
2. list of "pyrocko.model.Station" objects in the same order like the data traces.

We invite the users to propose data formats or outputs of specific programs that they would 
like to see implemented. 

Contributions
=============
This is an open source project and contributions to the repository are welcome!
