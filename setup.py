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


def bash_completions_dir():
    from subprocess import Popen, PIPE

    def q(c):
        return Popen(c, stdout=PIPE).communicate()[0]

    try:
        d = q(['pkg-config', 'bash-completion', '--variable=completionsdir'])
        return d.strip().decode('utf-8')
    except Exception:
        return None


class custom_build_py(build_py):
    def run(self):
        build_py.run(self)
        bd_dir = bash_completions_dir()
        if bd_dir:
            try:
                shutil.copy('extras/beat', bd_dir)
                print('Installing beat bash_completion to "%s"' % bd_dir)
            except Exception:
                print(
                    'Could not install beat bash_completion to "%s" '
                    '(continuing without)'
                    % bd_dir)


subpackages = ['beat.fast_sweeping', 'beat.voronoi']

setup(
    cmdclass={
        'build_py': custom_build_py},
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
        Extension(
            'fast_sweep_ext',
            sources=[os.path.join('src/fast_sweeping', 'fast_sweep_ext.c')],
            include_dirs=[numpy.get_include()])]
)
