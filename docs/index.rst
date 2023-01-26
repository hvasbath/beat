.. beat documentation master file, created by
   sphinx-quickstart on Thu Oct  6 21:48:50 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to beat's documentation!
================================

.. image :: _static/LOGO_BEAT.png

Author: Hannes Vasyura-Bathke


Citing BEAT
-----------
The development of BEAT lead to several publications that describe theory and methods in detail.
If your work results in an publication where you used BEAT we kindly ask you to consider citing the BEAT software package and the related article(s). Doing so is essential for maintaining and further developing the software.

.. [VasyuraBathke2021] Vasyura-Bathke, Hannes; Dettmer, Jan; Dutta, Rishabh, Mai, Paul Martin; Jónsson, Sigurjón (2021): **Accounting for theory errors with empirical Bayesian noise models in nonlinear centroid moment tensor estimation**. Geophysical Journal International. https://doi.org/10.1093/gji/ggab034

.. [VasyuraBathke2020] Vasyura-Bathke, Hannes; Dettmer, Jan; Steinberg, Andreas; Heimann, Sebastian; Isken, Marius; Zielke, Olaf; Mai, Paul Martin; Sudhaus, Henriette; Jónsson, Sigurjón (2020): **The Bayesian Earthquake Analysis Tool**. Seismological Research Letters. https://doi.org/10.1785/0220190075

.. [VasyuraBathke2019] Vasyura-Bathke, Hannes; Dettmer, Jan; Steinberg, Andreas; Heimann, Sebastian; Isken, Marius; Zielke, Olaf; Mai, Paul Martin; Sudhaus, Henriette; Jónsson, Sigurjón (2019): **BEAT - Bayesian Earthquake Analysis Tool**. V. 1.0. GFZ Data Services. http://doi.org/10.5880/fidgeo.2019.024

.. [Heimann2019] Heimann, Sebastian; Vasyura-Bathke, Hannes; Sudhaus, Henriette; Isken, Marius; Kriegerowski, Marius; Steinberg, Andreas; Dahm, Torsten: 2019. **A Python framework for efficient use of pre-computed Green’s functions in seismological and other physical forward and inverse source problems.** Solid Earth, 2019, 10(6):1921–1935. https://doi.org/10.5194/se-10-1921-2019


Using BEAT
----------
A list of publications implementing BEAT can be found `here <https://pyrocko.org/beat/docs/current/community_references.html>`__ .


Introduction
------------
In crustal deformation studies geophysicists are interested in estimating the
parameters of sources that might be the cause of deformation in the Earth's
crust. These may be for example, movement of fluids (e.g. magma) below a
volcano or the fast movements of one tectonic plate compared to another, also
known as earthquakes. These types of sources can be often approximated by
one or many rectangular dislocations (geometry, position, amount of
dislocation).
With observations at the earth's surface like geodetic data, i.e. deformation
maps from e.g. InSAR or point information from GNSS and seismic data i.e. seismic waveforms
from seismic stations, it is possible to estimate the parameters of these
deformation sources.

 .. image:: _static/defor_sources.png

BEAT is a package that can handle either geodetic and/or seismic data to
estimate source parameters of dislocations in the Earth's crust.


Contents:

.. toctree::
   :maxdepth: 3

   short_installation
   anaconda_installation
   installation
   getting_started/index
   examples/index
   faq
   api
   community_references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
