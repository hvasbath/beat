from subprocess import PIPE, Popen
from os.path import join as pjoin
import os
from tempfile import mkdtemp
import logging
import signal
import shutil
import sys
from beat.utility import list2string


logger = logging.getLogger('distributed')

program_bins = {'mpi': 'mpiexec'}
mpiargs_name = 'mpiinput'


__all__ = [
    'enum',
    'run_mpi_sampler',
    'MPIRunner']


def enum(*sequential, **named):
    """
    Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def have_backend():
    have_any = False
    for cmd in [[exe] for exe in program_bins.values()]:
        try:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE)
            (stdout, stderr) = p.communicate()
            have_any = True

        except OSError:
            pass

    return have_any


class NotInstalledError(Exception):
    pass


class MPIError(Exception):
    pass


class MPIRunner(object):

    def __init__(self, tmp=None, keep_tmp=False):
        if have_backend():
            logger.info('MPI is installed!')
        else:
            raise NotInstalledError(
                'could not find "mpiexec" please check the mpi installation!')

        if tmp is not None:
            tmp = os.path.abspath(tmp)

        self.keep_tmp = keep_tmp
        self.tempdir = mkdtemp(prefix='mpiexec-', dir=tmp)
        logger.info('Done initialising mpi runner')

    def run(self, script_path, n_jobs=None):

        if n_jobs is None:
            raise ValueError('n_jobs has to be defined!')

        program = program_bins['mpi']
        args = ' -n %i python %s ' % (n_jobs, script_path)
        commandstr = program + args

        old_wd = os.getcwd()

        os.chdir(self.tempdir)

        interrupted = []

        def signal_handler(signum, frame):
            os.kill(proc.pid, signal.SIGTERM)
            interrupted.append(True)

        original = signal.signal(signal.SIGINT, signal_handler)
        try:
            try:
                proc = Popen(
                    commandstr.split(), stdout=sys.stdout, stderr=sys.stderr)

            except OSError:
                os.chdir(old_wd)
                raise NotInstalledError(
                    '''could not start "%s"''' % program)

            logger.debug('Running mpi with arguments: %s' % args)
            proc.communicate()

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                'mpiexec had a non-zero exit state: %i' % proc.returncode)

        if errmess:
            self.keep_tmp = True

            os.chdir(old_wd)
            raise MPIError('''
mpiexec has been invoked as "%s"
in the directory %s'''.lstrip() % (program, self.tempdir))

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                logger.debug(
                    'removing temporary directory under: "%s"' % self.tempdir)
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn(
                    'not removing temporary directory: %s' % self.tempdir)


samplers = {
    'pt': 'src/sampler/pt.py'
}


def run_mpi_sampler(sampler_name, sampler_args, keep_tmp, n_jobs):
    """
    Execute a sampling algorithm that requires the call of mpiexec
    as it uses MPI for parallelization.

    A run directory is created unter '/tmp/' where the sampler
    arguments are pickled and then reloaded by the MPI sampler
    script.

    Parameters
    ----------
    sampler_name : string
        valid names: %s
    sampler_args : list
        of sampler arguments, order is important
    keep_tmp : boolean
        if true dont remove the run directory after execution
    n_jobs : number of processors to call MPI with
    """ % list2string(samplers.keys())

    from beat.info import project_root
    from beat.utility import dump_objects

    try:
        sampler = samplers[sampler_name]
    except KeyError:
        raise NotImplementedError(
            'Currently only samplers: %s supported!' %
            list2string(samplers.keys()))

    runner = MPIRunner(keep_tmp=keep_tmp)
    args_path = pjoin(runner.tempdir, mpiargs_name)

    dump_objects(args_path, sampler_args)

    samplerdir = pjoin(project_root, sampler)
    logger.info('sampler directory: %s' % samplerdir)
    runner.run(samplerdir, n_jobs=n_jobs)
