#!/usr/bin/env python

import os
from distutils.core import setup, Extension
from distutils.command.build_py import build_py
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
