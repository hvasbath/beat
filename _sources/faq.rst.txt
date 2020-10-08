
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

| **5. GF store error BAD_REQUEST**

This means that the geodetic (static) Green's Function store does not cover the full depth-distance range of the source- receiver pairs.
Please make sure that these ranges are wide enough in depths and distances!

| **6. Upgrading BEAT from beta**

Unfortunately, some incompatibility arose from beta to version 1.0. Finite fault projects that have been sampled and created using the beta version will need to undergo some manual changes by the user to be loadable under version 1.0.

1. Rename config_ffo.yaml to config_ffi.yaml
2. Rename "$project_folder/ffo" to "$project_folder/ffi"
3. To update the config file run::

    beat update $project_folder --mode=ffi  # (--diff to display updates first)
4. According to the $datatypes that were included in the project, recreate the fault geometry with::

    beat build_gfs $project_folder --mode=ffi --force --datatypes=$datatypes

| **7. Cannot connect to display while working remotely**

X forwarding needs to be activated in the ssh config! For linux:

1. vi /etc/ssh/ssh_config
2. set "X11Forwarding" to "yes", save and close 
3. in the shell run::

    systemctl restart sshd   # could require sudo rights
