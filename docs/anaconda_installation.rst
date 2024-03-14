
.. anaconda_installation:

**********************************
Anaconda Installation instructions
**********************************

For users that want to use anaconda to install BEAT one cannot follow the short or detailed installation instructions.
A general advice when dealing with anaconda is that the "sudo" command must NOT be used at any time, otherwise things will be installed to the system
instead of the respective anaconda environment.
Below are a series of commands that might be able to get you up and running using anaconda3 (thanks to Rebecca Salvage).

Create and activate a new conda environment e.g. called "beatenv" using python3.11 (minimum required is 3.9)::

  conda create -n beatenv python=3.11
  conda activate beatenv
  cd ~/src  # or wherever you keep the packages

Download and install several required packages::

  conda install -n beatenv libgfortran openblas pytensor numpy

Install pymc and pyrocko packages::

  conda install -n beatenv -c conda-forge pymc
  conda install -n beatenv -c pyrocko pyrocko

Once all the requirements are installed we install *BEAT* with::

  pip install beat

Then for a fast check if beat is running one can start it calling the help::

  beat init --help


Optional: Install MPI for the PT sampler
----------------------------------------
Install openmpi and mpi4py through conda-forge::

  conda install -n openmpi
  conda install -c conda-forge mpi4py


Optional: Install pygmsh and cutde for the BEM module
-----------------------------------------------------
There are optional dependencies that are required in order to use the Boundary Element Method (BEM) module.
For meshing *BEAT* uses the gmsh library and a python wrapper pygmsh::

  [sudo] apt install gmsh
  pip install pygmsh

To calculate synthetic surface displacements for triangular dislocations::

  conda install -c conda-forge cutde

Install and configure your GPU for *cutde* following this `page <https://github.com/tbenthompson/cutde?tab=readme-ov-file#gpu-installation>`__.


Optional: Greens Function calculations
--------------------------------------
To calculate the Greens Functions we rely on modeling codes written by
`Rongjiang Wang <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/staff/profil/rongjiang-wang/>`__.
If you plan to use the GreensFunction calculation framework,
these codes are required and need to be compiled manually.
The original codes are packaged for windows and can be found
`here <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/data-products-services/downloads-software/>`__.

For Unix systems the codes had to be repackaged.

The packages below are also github repositories and you may want to use "git clone" to download::

    git clone <url>

This also enables easy updating for potential future changes.

For configuration and compilation please follow the descriptions provided in each repository respectively.
For now we have no feedback on, if there are specific things to have in mind to install these programs within an anaconda environment.

Seismic synthetics
""""""""""""""""""
* `QSEIS <https://git.pyrocko.org/pyrocko/fomosto-qseis/>`__
* `QSSP <https://git.pyrocko.org/pyrocko/fomosto-qssp/>`__


Geodetic synthetics
"""""""""""""""""""
* `PSGRN_PSCMP <https://git.pyrocko.org/pyrocko/fomosto-psgrn-pscmp>`__
