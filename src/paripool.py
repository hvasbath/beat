import multiprocessing
import logging


logger = logging.getLogger('paripool')


def start_message():
    logger.info('Starting', multiprocessing.current_process().name)


def paripool(function, work, **kwargs):
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

    nprocs = kwargs.get('nprocs', None)
    initmessage = kwargs.get('initmessage', False)
    chunksize = kwargs.get('chunksize', None)

    if not initmessage:
        start_message = None

    if chunksize is None:
        chunksize = 1

    if nprocs == 1:
        def pack_one_worker(*work):
            iterables = map(iter, work)
            return iterables

        iterables = pack_one_worker(work)

        while True:
            args = [next(it) for it in iterables]
            kwargs = {}
            yield function(*args, **kwargs)

        return

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=nprocs,
                                initializer=start_message)

    try:
        result = pool.imap_unordered(function, work, chunksize=chunksize)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.warn('User interrupt! Ctrl + C')
        pool.terminate()
        pool.join()

    return
