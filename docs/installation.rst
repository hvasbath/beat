.. installation:

**********************************
Detailed Installation instructions
**********************************

BEAT can be installed on any Unix based system with python2.7
that supports its prerequisites.

BEAT source
-----------
First of all please download the beat source code::

    cd ~/src  # or whereever you keep the packages
    git clone https://github.com/hvasbath/beat

The package includes scripts that help setting up and testing the following
optimizations of your numerics libraries.

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

`Numpy configure <https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/>`__

`Theano configure <http://www.johnwittenauer.net/configuring-theano-for-high-performance-deep-learning/>`__

OpenBlas
""""""""
If the OpenBlas library is compiled locally, it is optimized for your machine
and speeds up the calculations::

    cd ~/src  # or where you keep your source files
    git clone https://github.com/xianyi/OpenBLAS
    cd OpenBLAS
    make FC=gfortran
    sudo make PREFIX=/usr/local install

Now we have to tell the system where to find the new OpenBLAS library.
In the directory /etc/ld.so.conf.d/ should be a file `libc.conf` containing
the line::

    /usr/local/lib

If it is there, fine. If not, you have to create it.

If you decided to install OpenBlas in a totally different directory you have to
create a file `openblas.conf` containing your custom_path::

    /custom_path/lib

In both cases, either only checking if the files are there or creating the new
file with the path; you have to do::

    sudo ldconfig

Alternatively, you could add your /custom_path/lib to the $LD_LIBRARY_PATH in
your .bashrc or .cshrc in the homedirectory::

    export LD_LIBRARY_PATH=/custom_path/lib:$LD_LIBRARY_PATH

Numpy
"""""
Building numpy from source requires cython::

    pip install cython

If you compile numpy locally against the previously installed OpenBlas
library you can gain significant speedup. For my machine it resulted 
in a speed-up of the numpy related calculations by a factor of at least 3.::

    cd ~/src
    git clone https://github.com/numpy/numpy
    cd numpy

Per default, the current developers branch is being installed. We want to
install one of the most recent stable branches::

    git checkout v1.11.1

Next, create a configuration file `site.cfg` that tells numpy where to find the
previously installed OpenBlas library::

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

If everything worked ok,i.e. no mention of the ATLAS library, run::

    python setup.py build

Finally::

    sudo python setup.py install

or if you prefer you can again use pip and it will be cleaner recognized by the
packaging::

    pip install .

Test the performance and if everything works fine::

    cd ~/src/beat
    python src/test/numpy_test.py

Depending on your hardware something around these numbers should be fine!::

    dotted two (1000,1000) matrices in 73.6 ms
    dotted two (4000) vectors in 10.82 us
    SVD of (2000,1000) matrix in 9.939 s
    Eigendecomp of (1500,1500) matrix in 36.625 s


Theano
""""""
Theano is a package that was originally designed for deep learning and enables
to compile the python code into GPU cuda code or CPU C++. Therefore, you can
decide to use the GPU of your computer rather than the CPU, without needing to
reimplement all the codes. Using the GPU is very much useful, if many heavy
matrix multiplications have to be done, which is the case for some of the BEAT
models (static and kinematic optimizers). Thus, it is worth to spent the time
to configure your theano to efficiently use your GPU. Even if you dont plan to
use your GPU, these instructions will help boosting your CPU performance as
well.

For the bleeding edge installation do::

    cd ~/src
    git clone https://github.com/Theano/Theano
    cd Theano
    sudo python setup.py install

For any troubleshooting and detailed installation instructions I refer to the
`Theano <http://deeplearning.net/software/theano/install.html>`__ webpage.

CPU setup
#########

Optional: Setup for libamdm
___________________________
Only for 64-bit machines!
This again speeds up the elemantary operations! Theano will for sure work
without including this, but the performance increase (below)
will convince you to do so ;) .

Download the amdlibm package `here <http://developer.amd.com/tools-and-sdks/
archive/compute/libm/>`__ according to your system.

For Linux based systems if you have admin rights (with $ROOT=/usr) do ::

    tar -xvfz amdlibm-3.1-lin64.tar.gz
    cd amdlibm-3.1-lin64
    cp lib/*/* $ROOT/lib64/
    cp include/amdlibm.h $ROOT/include/

If you do not want to install the library to your system libraries ergo
$ROOT = /custom_path/ you need to add this path again to your environment
variables $LD_LIBRARY_PATH and $LIBRARY_PATH, for example if
$ROOT=/usr/local/ ::

    export LIBRARY_PATH=/usr/local/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
    export C_INCLUDE_PATH=/usr/local/include:$C_INCLUDE_PATH

General
_______
In your home directory create a file `.theanorc`.
The file has to be edited depending on the type of processing unit that is
intended to be used. Set amdlibm = True if you did the optional step! ::

    [blas]
    ldflags = -L/usr/local/lib -lopenblas -lgfortran

    [nvcc]
    fastmath = True

    [global]
    device = cpu
    floatX = float64

    [lib]
    amdlibm = False  # if applicable set True here


GPU setup
#########
For NVIDIA graphics cards there is the CUDA package that needs to be installed.::

    sudo apt-get install nvidia-current
    sudo apt-get install nvdidia-cuda-toolkit

Restart the system.
To check if the installation worked well type::

    nvidia-smi

This should display stats about your graphics card model.

Now we have to tell theano where to find the cuda package.
For doing so we have to add the library folder to the $LD_LIBRARY_PATH and the
CUDA root direct to the $PATH.

In bash you can do it like this, e.g. (depending on the path to your cuda
installation) add to your .bashrc file in the home directory::

    export CUDA_LIB="/usr/local/cuda-5.5/lib64"
    export CUDA_ROOT="/usr/local/cuda-5.5/bin"

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$CUDA_LIB
    export PATH=${PATH}:$CUDA_ROOT

Theano also supports OpenCL, however, I haven't set it up myself so far and
cannot provide instructions on how to do it.

In your home directory create a file `.theanorc` with these settings::

    [blas]
    ldflags = -L/usr/local/lib -lopenblas -lgfortran

    [nvcc]
    fastmath = True

    [global]
    device = gpu
    floatX = float32


Check performance
#################

To check the performance of the CPU or GPU and whether the GPU is being used
as intended::

    cd ~/src/beat

Using the CPU (amdlibm = False)::

    THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python src/test/gpu_test.py 

    [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
    Looping 1000 times took 2.717895 seconds
    Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
      1.62323284]
    Used the cpu

Using the CPU (amdlibm = True)::

    THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python src/test/gpu_test.py 

    [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
    Looping 1000 times took 0.703979 seconds
    Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
      1.62323284]
    Used the cpu

That's a speedup of 3.86! On the ELEMENTARY operations like exp(), log(), cos() ...


Using the GPU::

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python src/test/gpu_test.py 

    Using gpu device 0: Quadro 5000 (CNMeM is disabled, cuDNN not available)
    [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>),
     HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
    Looping 1000 times took 0.841933 seconds
    Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
      1.62323296]
    Used the gpu

Congratulations, you are done with the numerics installations!


Main Packages
^^^^^^^^^^^^^

BEAT relies on 2 main libraries. Detailed installation instructions for each
can be found on the respective websites:

 - `pymc3 <https://github.com/pymc-devs/pymc3>`__
 - `pyrocko <http://pyrocko.org/>`__

pymc3
"""""
Pymc3 is a framework that provides various optimization algorithms allows and
allows to build Bayesian models. For the last stable release::

    pip install pymc3

For the bleeding edge::

    cd ~/src
    git clone https://github.com/pymc-devs/pymc3
    cd pymc3
    sudo python setup.py install

Pyrocko
"""""""
Pyrocko is an extensive library for seismological applications and provides a
framework to efficiently store and access Greens Functions.::

    cd ~/src
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

Pyproj
""""""
Pyproj is the last package and also the most easy one to install::

    pip install pyproj


BEAT
----
After these long and heavy installations, you can setup BEAT itself::

    cd ~/src/beat
    sudo python setup.py install

Greens Function calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
* `QSEIS <https://github.com/pyrocko/fomosto-qseis>`__
* `QSSP <https://github.com/pyrocko/fomosto-qssp>`__


Geodetic synthetics
"""""""""""""""""""
* `PSGRN_PSCMP <https://github.com/pyrocko/fomosto-psgrn-pscmp>`__
