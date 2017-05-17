import multiprocessing
import logging


logger = logging.getLogger('paripool')


def start_message():
    logger.info('Starting', multiprocessing.current_process().name)


def paripool(function, work, nprocs=None, chunksize=1, initmessage=False):
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
        log status message during initialisation, default: false
    chunksize : int
        number of work packages to throw at workers in each instance
    """

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
            yield pool.map(function, work, chunksize=chunksize)
        finally:
            pool.terminate()
