.. image:: docs/_static/LOGO_BEAT.png?raw=true
    :align: center
    :alt: BEAT logo

Bayesian Earthquake Analysis Tool
---------------------------------

Based on pyrocko, theano and pymc3

Documentation (under construction) can be found here:
https://hvasbath.github.io/beat/

The first version 1.0 of BEAT is released!

License 
=======
GNU General Public License, Version 3, 29 June 2007

Copyright © 2017 Hannes Vasyura-Bathke

BEAT is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
BEAT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.


Citation
========
If your work results in an publication where you used BEAT we kindly ask you to consider citing the BEAT software package and the related article.:

 > Vasyura-Bathke, Hannes; Dettmer, Jan; Steinberg, Andreas; Heimann, Sebastian; Isken, Marius; Zielke, Olaf; Mai, Paul Martin; Sudhaus, Henriette; Jónsson, Sigurjón (under Revision): The Bayesian Earthquake Analysis Tool. Seismological Research Letters 

 > Vasyura-Bathke, Hannes; Dettmer, Jan; Steinberg, Andreas; Heimann, Sebastian; Isken, Marius; Zielke, Olaf; Mai, Paul Martin; Sudhaus, Henriette; Jónsson, Sigurjón (2019): BEAT - Bayesian Earthquake Analysis Tool. V. 1.0. GFZ Data Services. http://doi.org/10.5880/fidgeo.2019.024

[![DOI](https://img.shields.io/badge/DOI-10.5880%2Ffidgeo.2019.024-blue)] (http://pmd.gfz-potsdam.de/panmetaworks/review/fa98c2af48960efa5c4dd9265591eccc8dce92bfb41ad3a0ffcf39aa0b847b3a)


Tutorials
=========
Step by step points on how to use the tool for several scenarios can be found here:
`Examples <https://hvasbath.github.io/beat/examples/index.html>`__

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

Support
=======
For substantial issues please use the "issues" tab here in the repository.
For smaller issues or short clarifications there is a support chat `here <https://hive.pyrocko.org/pyrocko-support/channels/beat>`__ . This is provided by the pyrocko project and is accessible after a short account creation.

Finally, there is the option to write an email to:

Hannes Vasyura-Bathke
hvasbath@uni-potsdam.de

Andreas Steinberg
andreas.steinberg@ifg.uni-kiel.de

Contributions
=============
This is an open source project and contributions to the repository are welcome!
