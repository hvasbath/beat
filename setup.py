#!/usr/bin/env python

import os
from distutils.core import setup, Extension

try:
    import numpy
except ImportError:
    class numpy():
        def __inti__(self):
            pass

        @classmethod
        def get_include(self):
            return ''


subpackages = ['beat.fast_sweeping']

setup(
    name='beat',
    description='Bayesian Earthquake Analysis Tool',
    version='0.1',
    author='Hannes Vasyuara-Bathke',
    author_email='hannes.vasyura-bathke@kaust.edu.sa',
    packages=['beat'] + subpackages,
    package_dir={'beat': 'src'},
    scripts=['apps/beat'],
    package_data={'beat': []},
    ext_modules=[
        Extension('fast_sweep_ext',
            sources=[os.path.join('src/fast_sweeping', 'fast_sweep_ext.c')],
            include_dirs=[numpy.get_include()])
                ]
)
