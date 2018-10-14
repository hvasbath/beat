#!/usr/bin/env python3
import os
from setuptools import setup, Extension
from setuptools.command.build_py import build_py
import shutil
import time

op = os.path

packname = 'beat'
version = '1.0rc1'


try:
    import numpy
except ImportError:
    class numpy():
        def __init__(self):
            pass

        @classmethod
        def get_include(self):
            return 


project_root = op.dirname(op.realpath(__file__))
REQUIREMENTS_FILE = op.join(project_root, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()


class NotInAGitRepos(Exception):
    pass


def git_infos():
    '''Query git about sha1 of last commit and check if there are local \
       modifications.'''

    from subprocess import Popen, PIPE
    import re

    def q(c):
        return Popen(c, stdout=PIPE).communicate()[0]

    if not op.exists('.git'):
        raise NotInAGitRepos()

    sha1 = q(['git', 'log', '--pretty=oneline', '-n1']).split()[0]
    sha1 = re.sub(br'[^0-9a-f]', '', sha1)
    sha1 = str(sha1.decode('ascii'))
    sstatus = q(['git', 'status'])
    local_modifications = bool(re.search(br'^#\s+modified:', sstatus,
                                         flags=re.M))
    return sha1, local_modifications


def make_info_module(packname, version):
    '''Put version and revision information into file src/info.py.'''

    sha1, local_modifications = None, None
    combi = '%s-%s' % (packname, version)
    try:
        sha1, local_modifications = git_infos()
        combi += '-%s' % sha1
        if local_modifications:
            combi += '-modified'

    except (OSError, NotInAGitRepos):
        pass

    datestr = time.strftime('%Y-%m-%d_%H:%M:%S')
    combi += '-%s' % datestr

    s = '''# This module is automatically created from setup.py
project_root = %s
git_sha1 = %s
local_modifications = %s
version = %s
long_version = %s  # noqa
installed_date = %s
''' % tuple([repr(x) for x in (
        project_root, sha1, local_modifications, version, combi, datestr)])

    try:
        f = open(op.join('src', 'info.py'), 'w')
        f.write(s)
        f.close()
    except Exception:
        pass


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

        make_info_module(packname, version)
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


subpackages = [
    'beat.fast_sweeping',
    'beat.voronoi',
    'beat.sampler',
    'beat.models',
    'beat.ffo',
    'beat.apps']

setup(
    cmdclass={
        'build_py': custom_build_py},
    name='beat',
    description='Bayesian Earthquake Analysis Tool',
    version=version,
    author='Hannes Vasyuara-Bathke',
    author_email='hannes.vasyura-bathke@kaust.edu.sa',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: C',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        ],
    install_requires=install_reqs,
    packages=['beat'] + subpackages,
    package_dir={'beat': 'src'},
    entry_points={
        'console_scripts':
            ['beat = beat.apps.beat:main']
    },
    package_data={'beat': []},
    ext_modules=[
        Extension(
            'fast_sweep_ext',
            sources=[op.join('src/fast_sweeping', 'fast_sweep_ext.c')],
            include_dirs=[numpy.get_include()]),
        Extension(
            'voronoi_ext',
            extra_compile_args=['-lm'],
            sources=[op.join('src/voronoi', 'voronoi_ext.c')],
            include_dirs=[numpy.get_include()])
        ]
)
