
Frequently asked questions
--------------------------
| Below you find a list for known errors, that may occur and how to work around:
|
| **1. HDF5surrogate escape**

NameError: global name 'exc' is not defined

| add:
| export LC_ALL=en_AU.utf8
| to your .bashrc
|
| **2. Theano MKL support**

RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.

| add
| export MKL_THREADING_LAYER=GNU
| to your .bashrc
|
| **3. Slow compilation**

No error will be thrown, but during "beat sample" the compilation of the forward model function ḿay take a long time.
In such a case the default compilation flags of theano may be overwritten. This may result in longer runtime.::
  THEANO_FLAGS=optimizer=fast_compile beat sample Projectpath 

| **4. MPI rank always 0**

ValueError: Specified more workers that sample in the posterior "8", than there are total number of workers "0"

| There is an inconsistency in the MPI executable that is used for starting the sampler and the library that was used to compile mpi4py.
| Please make sure to use the same MPI library. One fix would be to remove all installations of MPI from the machine and install a fresh version.


