.. short_installation:

*******************************
Short Installation instructions
*******************************

First install pyrocko following this webpage:

 - `pyrocko <http://pyrocko.org/>`__

Then install the following packages::

    sudo pip install pymc3
    sudo pip install pyproj

    cd ~/src  # or whereever you keep the packages
    git clone https://github.com/hvasbath/beat
    cd beat
    sudo python setup.py install

Greens Function calculations
----------------------------

To calculate the Greens Functions we rely on modeling codes written by
`Rongjiang Wang <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/staff/profil/rongjiang-wang/>`__.
If you plan to use the GreensFunction calculation framework,
these codes are required and need to be compiled manually.
The original codes are packaged for windows and can be found 
`here <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/data-products-services/downloads-software/>`__.

For Unix systems the codes had to be repackaged.

Seismic synthetics
""""""""""""""""""
* `QSEIS <http://kinherd.org/fomosto-qseis-2006a.tar.gz>`__
* `QSSP <http://kinherd.org/fomosto-qssp-2010.tar.gz>`__

After unpacking each package, within each folder run::

    autoreconf -i   # only if 'configure' script is missing
    F77=gfortran ./configure
    make
    sudo make install

Geodetic synthetics
"""""""""""""""""""
* PSGRN and PSCMP

These codes are so far included in the beat repository, but will be a part of the pyrocko framework in the future.
In the BEAT folder::

    tar -xvzf fomosto-psgrn-pscmp.tar.gz
    cd fomosto-psgrn-pscmp
    autoreconf -i   # only if 'configure' script is missing
    ./configure

If the number of modelled points is large the FFLAGS flag has to be changed to
FFLAGS=-mcmodel=large, this will result in a long compilation time.::

    FFLAGS=-mcmodel=large ./configure

After configuration::

    make
    sudo make install
