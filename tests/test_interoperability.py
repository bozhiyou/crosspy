from .conftest import *

def test_norm(np_arr, cp_arr, xp_arr):
    print(cp.linalg.norm(cp_arr))
    print(cp.linalg.norm(xp_arr))
    print(np.linalg.norm(np_arr))
    print(np.linalg.norm(xp_arr))


if __name__ == '__main__':
    # test_norm()
    pass