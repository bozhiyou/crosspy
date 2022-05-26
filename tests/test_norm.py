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


DIM = int(16_000) # 256 million


if __name__ == '__main__':
    @timing
    def test_norm(npa, cpa):
        print(cp.linalg.norm(cp.asarray(npa)))
        print(np.linalg.norm(cp.asnumpy(cpa)))


    @timing
    def test_norm_xp(xpa):
        # TODO xp.linalg.norm(xpa)
        print(np.linalg.norm(xpa[:DIM].to(0)))
        print(cp.linalg.norm(xpa[DIM:].to(-1)))


    import numpy as np
    import cupy as cp

    np_arr = np.random.rand(DIM, DIM)
    print(np_arr.shape)
    cp_arr = cp.random.rand(DIM, DIM)
    print(cp_arr.shape)

    import crosspy as xp
    xp_arr = xp.array([np_arr.copy(), cp_arr.copy()])
    print(xp_arr.shape)

    test_norm(np_arr, cp_arr)
    test_norm_xp(xp_arr)