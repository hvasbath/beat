.. short_installation:

*******************************
Short Installation instructions
*******************************

Starting BEAT v2.0.0 can be installed on any Unix based system with python>=3.9 that supports its prerequisites.
Earlier versions must be installed on python3.8!

Please consider to use `virtual environments <https://docs.python.org/3/tutorial/venv.html>`__ to lower the risk of package conflicts.


Install and create a virtual environment
----------------------------------------
Install the virtual environment package::

    sudo apt install python-venv

Create a directory *virtualenvs* where you want to keep your virtual environments, e.g. in user home::

    cd ~
    mkdir virtualenvs
    cd virtualenvs

Create new environment e.g. *beat_env* using python3.11 (for other version just change the number e.g.: python3.9) and activate it::

    python3.11 -m venv beat_env
    source ~/virtualenvs/beat_env/bin/activate

The environment can be (later) deactivated NOT NOW!, with::

    deactivate

Now we have created the *beat_env* environment into which we will install all the needed packages. Thus, we can avoid potential versioning conflicts
with other packages.


Install beat, latest release
----------------------------

In the activated environment we install the latest release of *beat* through the package manager pip::

    pip3 install beat


Install beat from source (github)
---------------------------------

Please make sure to activate your virtual environment! e.g.::

    source ~/virtualenvs/beat_env/bin/activate

Get the development version through the github repository::

    cd ~/src  # or wherever you keep the packages
    git clone https://github.com/hvasbath/beat
    cd beat
    # get feature branch from online repository and create local branch
    git fetch origin feature_branch_name:feature_branch_name
    # switch to branch "feature_branch_name" locally
    git checkout feature_branch_name
    git pull origin feature_branch_name
    pip3 install -e .

Once the development headers are installed. Only switching between gitbranches- is enough.::

    git checkout $branch_name


Optional Dependencies
---------------------
For using the BEM module
""""""""""""""""""""""""
The new Boundary Element Modeling (BEM) module requires extra dependencies (and dependencies within)::

    pygmsh
    cutde

To install *pygmsh*::

    [sudo] apt install python3-gmsh   # this will be a system wide installation of gmsh
    pip install pygmsh                # this will install the python abstraction library around gmsh

To install *cutde*::

    pip install cutde

This will be sufficient to run *cutde* on the CPU using its C++ backend. However, that will render sampling slow
to the point that it is not useful. In order to use the BEM module of BEAT for sampling a GPU is required.
Install instructions for installing the GPU depending on your system architecture for *cutde* are `here <https://github.com/tbenthompson/cutde?tab=readme-ov-file#gpu-installation>`__.

For using the PT sampler
""""""""""""""""""""""""
For the Parallel Tempering (PT) algorithm OpenMPI and the python
bindings are required. If you do not have any MPI library installed, this needs to be installed first.::

    [sudo] apt install openmpi-bin libopenmpi-dev

Finally, the python wrapper::

    pip3 install mpi4py


If a particular mpi version is required, they can be listed with the command::

    apt-cache madison libopenmpi-dev

To install openmpi for a specific version for example version 2.1.1-8::

    [sudo] apt install openmpi-bin=2.1.1-8 libopenmpi-dev=2.1.1-8 -V


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

Seismic synthetics
""""""""""""""""""
* `QSEIS <https://git.pyrocko.org/pyrocko/fomosto-qseis/>`__
* `QSSP <https://git.pyrocko.org/pyrocko/fomosto-qssp/>`__


Geodetic synthetics
"""""""""""""""""""
* `PSGRN_PSCMP <https://git.pyrocko.org/pyrocko/fomosto-psgrn-pscmp>`__
