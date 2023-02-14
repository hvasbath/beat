#!/usr/bin/env python3
import os
import shutil
import sys
import time
from distutils.sysconfig import get_python_inc

from setuptools import Extension, setup
from setuptools.command.build_py import build_py

op = os.path

packname = "beat"
version = "1.2.4"


try:
    import numpy
except ImportError:

    class numpy:
        def __init__(self):
            ...

        @classmethod
        def get_include(cls):
            return


project_root = op.dirname(op.realpath(__file__))


class NotInAGitRepos(Exception):
    pass


def git_infos():

    from subprocess import PIPE, run

    """Query git about sha1 of last commit and check if there are local \
       modifications."""
    import re

    def q(c):
        return run(c, stdout=PIPE, stderr=PIPE, check=True).stdout

    if not op.exists(".git"):
        raise NotInAGitRepos()

    sha1 = q(["git", "log", "--pretty=oneline", "-n1"]).split()[0]
    sha1 = re.sub(rb"[^0-9a-f]", "", sha1)
    sha1 = str(sha1.decode("ascii"))
    sstatus = q(["git", "status", "--porcelain", "-uno"])
    local_modifications = bool(sstatus.strip())
    return sha1, local_modifications


def make_info_module(packname, version):
    """Put version and revision information into file beat/info.py."""

    from subprocess import CalledProcessError

    sha1, local_modifications = None, None
    combi = "%s-%s" % (packname, version)
    try:
        sha1, local_modifications = git_infos()
        combi += "-%s" % sha1
        if local_modifications:
            combi += "-modified"

    except (OSError, CalledProcessError, NotInAGitRepos):
        print("Failed to include git commit ID into installation.", file=sys.stderr)

    datestr = time.strftime("%Y-%m-%d_%H:%M:%S")
    combi += "-%s" % datestr

    s = """# This module is automatically created from setup.py
project_root = %s
git_sha1 = %s
local_modifications = %s
version = %s
long_version = %s  # noqa
installed_date = %s
""" % tuple(
        [
            repr(x)
            for x in (project_root, sha1, local_modifications, version, combi, datestr)
        ]
    )

    try:
        f = open(op.join("beat", "info.py"), "w")
        f.write(s)
        f.close()
    except Exception:
        pass


def bash_completions_dir():
    from subprocess import PIPE, Popen

    def q(c):
        return Popen(c, stdout=PIPE).communicate()[0]

    try:
        d = q(["pkg-config", "bash-completion", "--variable=completionsdir"])
        return d.strip().decode("utf-8")
    except Exception:
        return None


def make_bash_completion():
    bd_dir = bash_completions_dir()
    if bd_dir:
        try:
            shutil.copy("extras/beat", bd_dir)
            print('Installing beat bash_completion to "%s"' % bd_dir)
        except Exception:
            print(
                'Could not install beat bash_completion to "%s" '
                "(continuing without)" % bd_dir
            )


make_info_module(packname, version)
make_bash_completion()


setup(
    ext_modules=[
        Extension(
            "fast_sweep_ext",
            language="c",
            sources=[op.join("beat/fast_sweeping", "fast_sweep_ext.c")],
            include_dirs=[numpy.get_include(), get_python_inc()],
        ),
        Extension(
            "voronoi_ext",
            extra_compile_args=["-lm"],
            language="c",
            sources=[op.join("beat/voronoi", "voronoi_ext.c")],
            include_dirs=[numpy.get_include(), get_python_inc()],
        ),
    ]
)
