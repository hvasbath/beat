import multiprocessing
import logging
import traceback
import functools
import signal


logger = logging.getLogger('paripool')


def exception_tracer(func):
    """
    Function decorator that returns a traceback if an Error is raised in
    a child process of a pool.
    """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            print('Exception in ' + func.__name__)
            raise type(e)(msg)

    return wrapped_func


class TimeoutException(Exception):
    """
    Exception raised if a per-task timeout fires.
    """
    def __init__(self, jobstack=[]):
        super(TimeoutException, self).__init__()
        self.jobstack = jobstack


# http://stackoverflow.com/questions/8616630/time-out-decorator-on-a-multprocessing-function
def overseer(timeout):
    """
    Function decorator that raises a TimeoutException exception
    after timeout seconds, if the decorated function did not return.
    """

    def decorate(func):
        def timeout_handler(signum, frame):
            raise TimeoutException(traceback.format_stack())

        @functools.wraps(func)
        def wrapped_f(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            result = func(*args, **kwargs)

            # Old signal handler is restored
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)  # Alarm removed
            return result

        wrapped_f.__name__ = func.__name__
        return wrapped_f

    return decorate


class WatchedWorker(object):
    """
    Wrapper class for parallel execution of a task.

    Parameters
    ----------
    task : function to execute
    work : List
        of arguments to specified function
    timeout : int
        time [s] after which worker is fired, default 65536s
    """

    def __init__(self, task, work, timeout=0xFFFF):
        self.function = task
        self.work = work
        self.timeout = timeout

    def run(self):
        """
        Start working on the task!
        """
        try:
            return self.function(*self.work)
        except TimeoutException:
            logger.warn('Worker timed out! Fire him! Returning: None!')
            return None


def _pay_worker(worker):
    """
    Wrapping function for the pool start instance.
    """
    return overseer(worker.timeout)(worker.run)()


def paripool(function, workpackage, nprocs=None, chunksize=1, timeout=0xFFFF,
    initmessage=True):
    """
    Initialises a pool of workers and executes a function in parallel by
    forking the process. Does forking once during initialisation.

    Parameters
    ----------
    function : function
        python function to be executed in parallel
    workpackage : list
        of iterables that are to be looped over/ executed in parallel usually
        these objects are different for each task.
    nprocs : int
        number of processors to be used in paralell process
    chunksize : int
        number of work packages to throw at workers in each instance
    timeout : int
        time [s] after which processes are killed, default: 65536s
    initmessage : bool
        log status message during initialisation, default: true
    """

    def start_message():
        logger.debug('Starting %s' % multiprocessing.current_process().name)

    def callback(result):
        logger.info('Feierabend! Done with the work!')

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    if not initmessage:
        start_message = None

    if chunksize is None:
        chunksize = 1

    if nprocs == 1:
        for work in workpackage:
            yield [function(*work)]

    else:
        pool = multiprocessing.Pool(
            processes=nprocs,
            initializer=start_message)

        logger.info('Worker timeout after %i second(s)' % timeout)

        workers = [
            WatchedWorker(function, work, timeout) for work in workpackage]

        pool_timeout = int(len(workpackage) * timeout / nprocs)

        logger.info('Overseer timeout after %i second(s)' % pool_timeout)

        try:
            yield pool.map_async(
                _pay_worker, workers,
                chunksize=chunksize, callback=callback).get(pool_timeout)
        except multiprocessing.TimeoutError:
            logger.error('Overseer fell asleep. Fire everyone!')
            pool.terminate()
        except KeyboardInterrupt:
            logger.error('Got Ctrl + C')
            traceback.print_exc()
            pool.terminate()
        else:
            pool.close()
            pool.join()
