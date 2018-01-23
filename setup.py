#!/usr/bin/env python

import os
from setuptools import setup, Extension
from setuptools.command.build_py import build_py
import shutil

try:
    import numpy
except ImportError:
    class numpy():
        def __inti__(self):
            pass

        @classmethod
        def get_include(self):
            return ''


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()


class custom_build_py(build_py):
    def run(self):
        build_py.run(self)
        try:
            shutil.copy('extras/beat', '/etc/bash_completion.d/beat')
            print 'Installing beat bash_completion...'
        except IOError as e:
            import errno
            if e.errno in (errno.EACCES, errno.ENOENT):
                print e
            else:
                raise e


subpackages = ['beat.fast_sweeping']

setup(
    cmdclass={
        'build_py': custom_build_py
                },
    name='beat',
    description='Bayesian Earthquake Analysis Tool',
    version='1.0beta',
    author='Hannes Vasyuara-Bathke',
    author_email='hannes.vasyura-bathke@kaust.edu.sa',
    install_requires=install_reqs,
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
