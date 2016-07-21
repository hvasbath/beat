#!/usr/bin/env python

from distutils.core import setup

setup(
    name='BEAT',
    description='Bayesian Earthquake Analysis Tool',
    version='0.1',
    author='Hannes Vasyuara-Bathke',
    author_email='hannes.vasyura-bathke@kaust.edu.sa',
    packages=['BEAT'],
    package_dir={'BEAT': 'src'},
    scripts=['apps/BEAT'],
    package_data={'BEAT': []})

