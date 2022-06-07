.. updating:

*************
Updating beat
*************

For installations into system
-----------------------------
In the beat main directory run::

  git pull origin master
  sudo python setup.py install

For anaconda installations
--------------------------

Activate your environment that you created for the beat installation::

  conda activate beat

In the beat main directory run::

  git pull origin master
  python setup.py install

Testing development features
----------------------------

New features are developed and implemented using git branches. These features can be tested and installed
with following commands in the beat main directory ::

  # get feature branch from online repository and create local branch
  git fetch origin feature_branch_name:feature_branch_name
  # switch to branch "feature_branch_name" locally
  git checkout feature_branch_name
  git pull origin feature_branch_name
  sudo python setup.py install

Please be aware that compatibility between development versions and released versions is not always given.
Accordingly, some project configurations will only run on the particular branch and it is the responsibility
of the user to keep track of these versions and modeling projects.

The master branch can then be recovered with::

  git checkout master
  sudo python setup.py install
