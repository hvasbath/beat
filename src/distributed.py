from subprocess import PIPE, Popen
from os.path import join as pjoin
import os
from tempfile import mkdtemp
import logging
import signal
import shutil


logger = logging.getLogger('distributed')

program_bins = {'mpi': 'mpiexec'}


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
                proc = Popen(commandstr.split(), stdout=PIPE, stderr=PIPE)

            except OSError:
                os.chdir(old_wd)
                raise NotInstalledError(
                    '''could not start "%s"''' % program)

            logger.debug('Running mpi with arguments: %s' % args)
            (output_str, error_str) = proc.communicate()

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        logger.debug('===== begin mpiexec output =====\n'
                     '%s===== end mpiexec output =====' % output_str.decode())

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                'mpiexec had a non-zero exit state: %i' % proc.returncode)

        if error_str:
            logger.warn(
                'mpiexec emitted something via stderr:\n\n%s'
                % error_str.decode())

            # errmess.append('qseis emitted something via stderr')

        if output_str.lower().find(b'error') != -1:
            errmess.append("the string 'error' appeared in output")

        if errmess:
            self.keep_tmp = True

            os.chdir(old_wd)
            raise MPIError('''
===== begin mpiexec output =====
%s===== end mpiexec output =====
===== begin mpiexec error =====
%s===== end mpiexec error =====
%s
qseis has been invoked as "%s"
in the directory %s'''.lstrip() % (
                output_str.decode(), error_str.decode(),
                '\n'.join(errmess), program, self.tempdir))

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
