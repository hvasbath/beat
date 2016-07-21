#!/usr/bin/env python

from distutils.core import setup

setup(
    name='beat',
    description='Bayesian Earthquake Analysis Tool',
    version='0.1',
    author='Hannes Vasyuara-Bathke',
    author_email='hannes.vasyura-bathke@kaust.edu.sa',
    packages=['beat'],
    package_dir={'beat': 'src'},
    scripts=['apps/beat'],
    package_data={'beat': []})

