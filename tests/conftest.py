import pytest

import numpy, numpy as np
import cupy, cupy as cp
import crosspy, crosspy as xp

DIM = int(100)

@pytest.fixture
def np_arr():
    return np.random.rand(DIM)

@pytest.fixture
def cp_arr():
    return cp.random.rand(DIM)

@pytest.fixture
def xp_arr(np_arr, cp_arr):
    return xp.array([np_arr.copy(), cp_arr.copy()], axis=0)