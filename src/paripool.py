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
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            print('Exception in ' + func.__name__)
            raise type(e)(msg)

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

    def callback(result):
        logger.debug('Done with the work! {}'.format(result))

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
        pool = multiprocessing.Pool(
            processes=nprocs,
            initializer=start_message)

        try:
            yield pool.map_async(
                function, work,
                chunksize=chunksize, callback=callback).get(0xFFFF)

        except KeyboardInterrupt:
            logger.error('Got Ctrl + C')
            traceback.print_exc()
            pool.terminate()
        else:
            pool.close()
