import numpy
from crosspy import cupy as cp
from crosspy.device import _GPUDevice

def test_cupy_cuda_device():
    assert isinstance(cp.cuda.Device(0), _GPUDevice)

if __name__ == '__main__':
    test_cupy_cuda_device()