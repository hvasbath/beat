import multiprocessing
import logging
import traceback
import functools
import signal
import time

logger = logging.getLogger('paripool')


def exception_tracer(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            logger.warn('Exception in ' + func.__name__)
            traceback.print_exc()
    return wrapped_func


def paripool(function, work, nprocs=None, chunksize=1, initmessage=True):
    """
    Initialises a pool of workers and executes a function in parallel by
    forking the process. Does forking once during initialisation.

    Parameters
    ----------
    function : function
        python function to be executed in parallel
    work : list
        of iterables that are to be looped over/ executed in parallel usually
        these objects are different for each task.
    nprocs : int
        number of processors to be used in paralell process
    initmessage : bool
        log status message during initialisation, default: true
    chunksize : int
        number of work packages to throw at workers in each instance
    """

    def start_message():
        logger.debug('Starting %s' % multiprocessing.current_process().name)

    def callback():
        logger.info('Done with the work!')

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    if not initmessage:
        start_message = None

    if chunksize is None:
        chunksize = 1

    if nprocs == 1:
        for work_item in work:
            yield function(work_item)

    else:
        osig_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        pool = multiprocessing.Pool(
            processes=nprocs,
            initializer=start_message)

        signal.signal(signal.SIGINT, osig_handler)

        try:
            yield pool.map_async(
                function, work,
                chunksize=chunksize, callback=callback).get(0xFFFF)

        except KeyboardInterrupt:
            logger.error('Got Ctrl + C')
            pool.terminate()
        else:
            pool.close()
