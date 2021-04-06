.. updating:

*************
Updating beat
*************

In the beat main directory run::

    git pull origin master
    sudo python setup.py install


Testing development features
----------------------------

New features are developed and implemented using git branches. These features can be tested and installed
with following commands::

  # get feature data from online repository
  git fetch origin feature_branch_name
  # create branch "feature_branch_name" locally
  git checkout -b feature_branch_name
  git pull origin feature_branch_name
  sudo python setup.py install

Please be aware that compatibility between development versions and released versions is not always given.
Accordingly, some project configurations will only run on the particular branch and it is the responsibility
of the user to keep track of these versions and modeling projects.

The master branch can then be recovered with::

  git checkout master
  sudo python setup.py install
