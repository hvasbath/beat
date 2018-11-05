
Frequently asked questions
--------------------------
Below you find a list for known errors, that may occur and how to work around:

1. HDF5surrogate escape
NameError: global name 'exc' is not defined

export LC_ALL=en_AU.utf8
to your .bashrc

2. Theano MKL support
RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.
export MKL_THREADING_LAYER=GNU
to your .bashrc

 

