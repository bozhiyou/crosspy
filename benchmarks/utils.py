import sys
import os

sys.path.append(os.path.dirname(__file__) + '/..')

from functools import wraps
from time import time


def timing(test):
    @wraps(test)
    def wrap(*args, **kw):
        t = time()
        res = test(*args, **kw)
        tt = time()
        print("%2.4fs\t%r" % (tt - t, test.__name__))
        return res

    return wrap