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

Data import
===========
The import of data (geodetic, seismic) is so far restricted to only two formats.
seismic - autokiwi output
geodetic - matlab, ascii (GAMIT output)

An alternative:
The geodetic data may be saved using the package "pickle" as a file "geodetic_data.pkl"
containing a list of "GeodeticTarget", especially "CompoundGPS" or "DiffIFG" objects. Please see the heart.py module for specifics.

The seismic data may be saved using the package "pickle" as a file "seismic_data.pkl"
containing a list of 2 lists:
1. list of "pyrocko.trace.Trace" objects alternating for (Z / T) rotated traces.
2. list of "pyrocko.model.Station" objects in the same order like the data traces.

We invite the users to propose data formats or outputs of specific programs that they would 
like to see implemented. 

Contributions
=============
This is an open source project and contributions to the repository are welcome!
