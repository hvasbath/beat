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
Step by step points on how to use the tool are in preparation and will be online soon! Stay tuned!

Data import
===========
Geodetic
^^^^^^^^
We recommend to prepare the SAR data (subsampling, data covariance estimation) using KITE (www.pyrocko.org).
kite supports import of ISCE, GAMMA, ROI_Pac and GMTSAR processed interferograms. BEAT then supports import of the native KITE format.

Seismic
^^^^^^^
Unfortunately, the import of seismic data is so far restricted to only one format:
 autokiwi output

Alternatively the seismic data may be saved using the package "pickle" as a file "seismic_data.pkl"
containing a list of 2 lists:
1. list of "pyrocko.trace.Trace" objects alternating for (Z / T) rotated traces.
2. list of "pyrocko.model.Station" objects in the same order like the data traces.
How to import the data into pyrocko format we refer to the webpage: https://pyrocko.org/docs/current/library/examples/trace_handling.html

We invite the users to propose data formats or outputs of specific programs that they would 
like to see implemented. 

Contributions
=============
This is an open source project and contributions to the repository are welcome!
