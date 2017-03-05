import multiprocessing
import logging


logger = logging.getLogger('paripool')


def start_message():
    logger.info('Starting', multiprocessing.current_process().name)


def paripool(function, work, **kwargs):

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
