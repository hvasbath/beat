
![BEAT logo](https://pyrocko.org/beat/docs/current/_images/LOGO_BEAT.png)

# Bayesian Earthquake Analysis Tool

If you came looking for the beat package calculating internet time you can find it [here](https://github.com/tonyskapunk/beat).

Based on pyrocko, theano and pymc3

14 February 2023
Version 1.2.4 is released. Details in the [changelog](https://github.com/hvasbath/beat/blob/master/CHANGELOG.md).

20 November 2022
Version 1.2.3 is released. Bugfix-release.

28 October 2022
Version 1.2.2 is released. Details in the [changelog](https://github.com/hvasbath/beat/blob/master/CHANGELOG.md).

14 September 2022
Version 1.2.1 is released, improvements on the stage_posterior plot. Updated [Example 8](https://pyrocko.org/beat/docs/current/examples/MTQT_polarity.html#plotting) to show-case its use.

21 August 2022
Version 1.2.0 is released introducing waveform polarity inversion and inversion in spectral domain for seismic data.
Checkout the [changelog](https://github.com/hvasbath/beat/blob/master/CHANGELOG.md) for all the details.
New [tutorial](https://pyrocko.org/beat/docs/current/examples/MTQT_polarity.html) on moment tensor inversion with P wave polarity picks.
Now PyPI packages are provided and the installation via pip is recommended.

6 January 2022
A new version 1.1.1 is released vastly improving resolution based fault discretization. Minor changes.
Checkout the [changelog](https://github.com/hvasbath/beat/blob/master/CHANGELOG.md) for all the details.
New [tutorial](https://pyrocko.org/beat/docs/current/examples/FFI_static_resolution.html) on resolution based discretization of the fault surface.

12 April 2021
A new version 1.1.0 is released adding support for multi-segmented fault setups and tensile dislocations.
Checkout the [changelog](https://github.com/hvasbath/beat/blob/master/CHANGELOG.md) for all the details.

Documentation of the current version moved to the pyrocko server can be found here:
https://pyrocko.org/beat/

New [tutorial](https://pyrocko.org/beat/docs/current/examples/Rectangular_tensile.html) on tensile dislocation modeling.

**HELP wanted!**

The new release contains a lot of undocumented features a list of these can be found here:
https://github.com/hvasbath/beat/issues/69
However, as this project is mostly the work of a single author it is becoming increasingly difficult to also
write extended pages of tutorials that take days and days of writing. However, as this work is not acknowledged by the
current academic system, I had to decide for a delayed release of the documentation, whenever it will
be available provided by someone. Thus, in case you are willing to contribute I would be more than happy to guide/ support
you in writing parts of the documentation for a particular feature-if you want to try it out.

The legacy documentation of beat v1.0. can be found under: https://hvasbath.github.io/beat/

## License
GNU General Public License, Version 3, 29 June 2007

Copyright © 2017 Hannes Vasyura-Bathke

BEAT is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
BEAT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.


## Citation
If your work results in an publication where you used BEAT we kindly ask you to consider citing the BEAT software package and the related article.:

 > Vasyura-Bathke, Hannes; Dettmer, Jan; Steinberg, Andreas; Heimann, Sebastian; Isken, Marius; Zielke, Olaf; Mai, Paul Martin; Sudhaus, Henriette; Jónsson, Sigurjón: The Bayesian Earthquake Analysis Tool. Seismological Research Letters. https://doi.org/10.1785/0220190075

 > Vasyura-Bathke, Hannes; Dettmer, Jan; Steinberg, Andreas; Heimann, Sebastian; Isken, Marius; Zielke, Olaf; Mai, Paul Martin; Sudhaus, Henriette; Jónsson, Sigurjón (2019): BEAT - Bayesian Earthquake Analysis Tool. V. 1.0. GFZ Data Services. http://doi.org/10.5880/fidgeo.2019.024


## Tutorials
Step by step points on how to use the tool for several scenarios can be found here:
[Examples](https://hvasbath.github.io/beat/examples/index.html)

## Data import
### Geodetic
We recommend to prepare the SAR data (subsampling, data covariance estimation) using KITE (www.pyrocko.org).
kite supports import of ISCE, GAMMA, ROI_Pac and GMTSAR processed interferograms. BEAT then supports import of the native KITE format.

### Seismic
To see a list of the supported data types please see: [Trace Handling](https://pyrocko.org/docs/current/library/examples/trace_handling.html)
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

## Support
For substantial issues please use and check the "issues" tab here in the repository.
For common issues please check out the BEAT [FAQ](https://hvasbath.github.io/beat/faq.html).
For smaller issues or short clarifications there is a support [chat](https://hive.pyrocko.org/pyrocko-support/channels/beat). This is provided by the pyrocko project and is accessible after a short account creation.

Finally, there is the option to write an email to:

Hannes Vasyura-Bathke
hvasbath@uni-potsdam.de

Andreas Steinberg
andreas.steinberg@ifg.uni-kiel.de

## Contributions
This is an open source project and contributions to the repository are welcome!
