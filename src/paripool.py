import multiprocessing
import logging
from tqdm import tqdm
import itertools

logger = logging.getLogger('paripool')


def workbench(function, work, pshared):
    kwargs = {}
    if pshared is not None:
        kwargs['pshared'] = pshared

    print 'On workbench'
    result = function(*work, **kwargs)
    return result

def start_message():
    logger.info('Starting', multiprocessing.current_process().name)


def paripool(function, work, **kwargs):

    nprocs = kwargs.get('nprocs', None)
    initmessage = kwargs.get('initmessage', False)
    pshared = kwargs.get('pshared', None)

    if not initmessage:
        start_message = None

    if nprocs == 1:
        iterables = map(iter, iterables)
        kwargs = {}

        while True:
            args = [next(it) for it in iterables]
            yield function(*args, **kwargs)

        return

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=nprocs,
                                initializer=start_message)

    try:
        result = pool.imap_unordered(function, work, chunksize=nprocs)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.warn('User interrupt! Ctrl + C')
        pool.terminate()
        pool.join()

    return
