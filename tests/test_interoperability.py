import numpy as np
import cupy as cp
import crosspy as xp



DIM = int(100)


def test_norm(xpa, npa, cpa):
    print(cp.linalg.norm(cpa))
    print(cp.linalg.norm(xpa))
    print(np.linalg.norm(npa))
    print(np.linalg.norm(xpa))


if __name__ == '__main__':
    np_arr = np.random.rand(DIM)
    cp_arr = cp.random.rand(DIM)
    xp_arr = xp.array([np_arr.copy(), cp_arr.copy()], axis=0)
    test_norm(xp_arr, np_arr, cp_arr)