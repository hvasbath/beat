
Installation instructions
-------------------------

BEAT can be installed on any Unix based system that supports its prerequisites.

Prerequisites
-------------
First of all we need a fortran compiler and the python developers library::

    sudo apt-get install git python-dev gfortran

Numerics
^^^^^^^^
BEAT does many intensive calculations, which is why we need to get as much as
possible out of the available libraries in terms of computational efficiency.
There are actually significant speedups possible by not using the standard
distribution packages that are available over tools like `pip` or
`easy_install`.

Although, this process is somewhat tedious and not straight forward for
everybody, it is really worth doing so! If you have done a similar optimization
to your machine's libraries already, you can start at the Main Packages section.

For everyone else I summarized the relevant points below.
For all the heavy details I refer to these links:

https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/
http://www.johnwittenauer.net/configuring-theano-for-high-performance-deep-learning/

OpenBlas
""""""""
If the OpenBlas library is compiled locally it is optimized for your machine
and again speeds up the calculations::

    cd ~/src  # or where you keep your source files
    git clone https://github.com/xianyi/OpenBLAS
    cd OpenBLAS
    make FC=gfortran
    sudo make PREFIX=/usr/local install

Numpy
"""""
Again, I want to note how important it is, to compile your numpy locally
against the previously installed OpenBlas library. For my machine it resulted 
in a speed-up of the numpy related calculations by a factor of at least 3.

    cd ~/src
    git clone https://github.com/numpy/numpy
    cd numpy

Per default, the current developers branch is being installed. We want to
install one of the most recent stable branches::

    git checkout -v1.11.1

Next create a configuration file `site.cfg` that tells numpy where to find the
 previously installed OpenBlas library:

[default]
include_dirs = /usr/local/include
library_dirs = /usr/local/lib

[openblas]
openblas_libs = openblas
library_dirs = /usr/local/lib

[lapack]
lapack_libs = openblas
library_dirs = /usr/local/lib

To make sure everything is working you should run::

    python setup.py config

If everything worked ok, no mention of the ATLAS library run::

    python setup.py build

Finally::

    sudo python setup.py install


Theano
""""""



BEAT relies on 2 main libraries. Detailed installation instructions for each
can be found on the respective websites:
*`pyrocko <http://pyrocko.org/>`
*`pymc3


pyproj

http://pyrocko.org/v0.3/install_details.html
