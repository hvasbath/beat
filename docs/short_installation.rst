.. short_installation:

*******************************
Short Installation instructions
*******************************

BEAT can be installed on any Unix based system with python>=3.7 that supports its prerequisites.

Please consider to use `virtual environments <https://docs.python.org/3/tutorial/venv.html>`__ to lower the risk of package conflicts.


Install and create a virtual environment
----------------------------------------
Install the virtual environment package::

    sudo apt install python-venv

Create a directory *virtualenvs* where you want to keep your virtual environments, e.g. in user home::

    cd ~
    mkdir virtualenvs
    cd virtualenvs

Create new environment e.g. *beat_env* and activate it::

    python3 -m venv beat_env
    source ~/virtualenvs/beat_env/bin/activate

The environment can be (later) deactivated NOT NOW!, with::

    deactivate

Now we have created the *beat_env* environment into which we will install all the needed packages. Thus, we can avoid potential versioning conflicts
with other packages.


Install beat, latest release
----------------------------

In the activated environment we install the latest release of *beat* through the package manager pip::

    pip3 install beat


Install beat, development version
---------------------------------

Get the development version through the github repository::

    cd ~/src  # or wherever you keep the packages
    git clone https://github.com/hvasbath/beat
    cd beat
    git pull origin master
    pip3 install .


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
