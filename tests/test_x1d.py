import crosspy as xp
import numpy as np
from crosspy import cupy as cp

MAX_BLK_SZ = 10
NUM_AVAIL_GPUS = 2

def _random_npa():
    return np.random.rand(np.random.randint(1, MAX_BLK_SZ))

def _random_cpa(d):
    with cp.cuda.Device(d):
        return cp.random.rand(cp.random.randint(1, MAX_BLK_SZ).item())

def test_x1d():
    from crosspy.core.x1darray import X1D
    for _ in range(5):
        num_blocks = np.random.randint(1, 10)
        devices = np.random.randint(-1, NUM_AVAIL_GPUS, size=num_blocks)
        blocks = []
        for d in devices:
            if d < 0:
                blocks.append(_random_npa())
            else:
                blocks.append(_random_cpa(d))
        x1d = X1D(blocks, axis=0)

        i = 0
        for b in blocks:
            for x in b:
                assert x1d[i] == x
                i += 1

if __name__ == '__main__':
    test_x1d()