import time

import numpy as num

from beat.parallel import paripool


def add(x, y):
    time.sleep(x)
    return x + y


def test_pool_with_timeout():
    factors = num.array([0, 1, 2, 3, 2, 1, 0])
    arguments = [[k, 1] for k in factors]

    results = []
    for chunk in paripool(add, arguments, chunksize=2, nprocs=4, timeout=3):
        results.extend(chunk)

    # the task that sleeps as long as the timeout is not returned
    ref_values = (factors + 1).tolist()
    ref_values[3] = None
    assert results == ref_values
