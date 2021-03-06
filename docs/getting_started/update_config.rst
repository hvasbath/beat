How to update the configuration files
-------------------------------------
In the course of development in BEAT it happened and may happen in the future that
the structure in the configuration file changes. Thus after a code base upgrade it may happen that
older configuration files cannot be read anymore. The program will raise an Error with the message
that the configuration file has to be updated and how this can be done. However, it may be of interest to know
before the actual update what is going to change. These changes can be displayed with::

    beat update <project_directory> --diff

An update screen that appears may look like this.

.. image:: ../_static/getting_started/beat_update_cut.png

Where omitting the "--diff" option will update the configuration file right away.

An update of the configuration file is also necessary, if some of the hierarchical parameters are affected by some changes in the configuration file. For example by enabling temporal "station_corrections" in seismic setups or enabling "fit_ramp"
for residual ramp estimation in the InSAR data.::

    beat update <project_directory> --diff --parameters='hierarchicals, hypers'
