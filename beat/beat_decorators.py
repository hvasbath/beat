from functools import wraps
from timeit import Timer


def time_method(loop=10000):
    def timer_decorator(func):
        @wraps(func)
        def wrap_func(*args, **kwargs):
            total_time = Timer(lambda: func(*args, **kwargs)).timeit(number=loop)
            print("Method {name} run {loop} times".format(name=func.__name__, loop=loop))
            print("It took: {time} s, Mean: {mean_time} s".
                  format(mean_time=total_time / loop, time=total_time))
            # return func(*args, **kwargs)
        return wrap_func
    return timer_decorator
