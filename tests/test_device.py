import numpy as np
import crosspy as xp
from crosspy import cupy as cp, gpu, cpu
from crosspy.device import _GPUDevice

def test_asnumpy():
    x = xp.random.rand([2,3,4], device=[gpu(1), cpu(0), gpu(0)])
    a = np.asarray(x)
    assert isinstance(a, np.ndarray)
    assert (a.shape == x.shape)

def test_cupy_cuda_device():
    assert isinstance(cp.cuda.Device(0), _GPUDevice)

if __name__ == '__main__':
    test_asnumpy()
    test_cupy_cuda_device()