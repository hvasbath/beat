
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

