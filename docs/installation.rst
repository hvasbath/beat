.. installation:

*************************
Installation instructions
*************************

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
Buildin numpy from source requires cython::

    pip install cython

If you compile your numpy locally against the previously installed OpenBlas
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
In your home directory create a file `.theanorc`.
The file has to be edited depending on the type of processing unit that is
intended to be used. For CPU::

    [blas]
    ldflags = -L/usr/local/lib -lopenblas -lgfortran

    [nvcc]
    fastmath = True

    [global]
    device = cpu
    floatX = float64

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

To check if the GPU is being actually active and used in the calculations
copy and paste the follwing code and run it::

    from theano import function, config, shared, sandbox
    import theano.tensor as T
    import numpy
    import time

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')

Using the CPU::

    THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python gpu_test.py 

    [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
    Looping 1000 times took 1.311933 seconds
    Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
      1.62323284]
    Used the cpu

Using the GPU::

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gpu_test.py 

    Using gpu device 0: Quadro 5000 (CNMeM is disabled, cuDNN not available)
    [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>),
     HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
    Looping 1000 times took 0.841933 seconds
    Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
      1.62323296]
    Used the gpu

You are done with the numerics installations!


Main Packages
^^^^^^^^^^^^^

BEAT relies on 2 main libraries. Detailed installation instructions for each
can be found on the respective websites:

`pymc3 <https://github.com/pymc-devs/pymc3>`__
`pyrocko <http://pyrocko.org/>`__

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
    cd beat
    sudo python setup.py install

Pyproj
""""""
Pyproj is the last package and also the most easy one to install::

    pip install pyproj



BEAT source
-----------
After these long and heavy installations, BEAT itself is easy and
straight-forward to install::

    cd ~/src
    git clone https://github.com/hvasbath/beat
    cd beat
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
