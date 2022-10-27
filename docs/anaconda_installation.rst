
.. anaconda_installation:

**********************************
Anaconda Installation instructions
**********************************

For users that want to use anaconda to install BEAT one cannot follow the short or detailed installation instructions.
A general advice when dealing with anaconda is that the "sudo" command must NOT be used at any time, otherwise things will be installed to the system
instead of the respective anaconda environment.
Below are a series of commands that might be able to get you up and running using anaconda3 (thanks to Rebecca Salvage).

Create and activate a new conda environment e.g. called "beat" using python3.8 (minimum required is 3.7)::

  conda create -n beat python=3.8
  conda activate beat
  cd ~/src  # or wherever you keep the packages

Download the beat source package from github (requires git to be installed on your machine)::

  git clone https://github.com/hvasbath/beat

Download and install several required packages::

  conda install -n beat libgfortran openblas theano pygpu openmpi pandas numpy openmpi

Install mpi4py through conda-forge::

  conda install -c conda-forge mpi4py

Configure theano to find your libraries by creating a file ".theanorc" in your home directory containing::

  [blas]
  ldflags = -L/path/to/your/anaconda/environments/beat/lib -lopenblas -lgfortran

  [nvcc]
  fastmath = True

  [global]
  device = cpu
  floatX = float64

For testing if numpy and theano installations worked fine::

  cd ~/src/beat
  python3 test/numpy_test.py
  THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3 test/gpu_test.py

Install pymc3 and pyrocko packages::

  conda install -n beat -c conda-forge pymc3=3.4.1
  conda install -n beat -c pyrocko pyrocko

Once all the requirements are installed we install BEAT with::

  cd ~/src/beat
  pip3 install .

Then for a fast check if beat is running one can start it calling the help::

  beat init --help

Greens Function calculations
----------------------------

To calculate the Greens Functions we rely on modeling codes written by
`Rongjiang Wang <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/staff/profil/rongjiang-wang/>`__.
If you plan to use the GreensFunction calculation framework,
these codes are required and need to be compiled manually.
The original codes are packaged for windows and can be found
`here <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/data-products-services/downloads-software/>`__.

For Unix systems the codes had to be repackaged.

The packages below are also github repositories and you may want to use "git clone" to download:

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
