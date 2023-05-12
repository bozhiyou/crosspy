import numpy as np
import cupy as cp

from timeit import timeit
from warnings import warn

GB32 = 2**(30-2)
GB64 = 2**(30-3)

def make_data(device_id, size):
    if device_id >=0:
        with cp.cuda.Device(device_id) as device:
            data = cp.ones(size, dtype=cp.float32)
            device.synchronize()
    else:
        data = np.ones(size, dtype=np.float32)
    return data

def copy(array, src_dev_id, tgt_dev_id):
    # Assume device >= 0 means the device is a GPU, device < 0 means the device is a CPU.

    if src_dev_id >= 0 and tgt_dev_id < 0:
        warn("Copy GPU to CPU")
        with cp.cuda.Device(src_dev_id):
            with cp.cuda.Stream(non_blocking=True) as stream:
                membuffer = cp.asnumpy(array, stream=stream)
                stream.synchronize()
        return membuffer

    if src_dev_id < 0 and tgt_dev_id >= 0:
        warn("Copy CPU to GPU")
        with cp.cuda.Device(tgt_dev_id):
            with cp.cuda.Stream(non_blocking=True) as stream:
                membuffer = cp.empty(array.shape, dtype=array.dtype)
                membuffer.set(array, stream=stream)
                stream.synchronize()
        return membuffer

    if src_dev_id >= 0 and tgt_dev_id >= 0:
        warn("Copy GPU to GPU")
        with cp.cuda.Device(tgt_dev_id):
            with cp.cuda.Stream(non_blocking=True) as stream:
                membuffer = cp.empty(array.shape, dtype=array.dtype)
                membuffer.data.copy_from_device_async(array.data, array.nbytes, stream=stream)
                stream.synchronize()
        return membuffer

    if src_dev_id < 0 and tgt_dev_id < 0:
        warn("Copy CPU to CPU")
        return np.copy(array)

    raise Exception("I'm not sure how you got here. But we don't support this device combination")

def estimate(source, destination, size=1*GB32, samples=20):
    print(
        timeit("copy(array, source, destination)",
               setup="array = make_data(source, size)",
               number=samples,
               globals={
                "make_data": make_data,
                "copy": copy,
                "destination": destination,
                "source": source,
                "size": size
               }
            ) / samples
        )

    
if __name__ == '__main__':
    estimate(-1, -1)

    estimate(-1, 0)
    estimate(-1, 1)

    estimate(0, -1)
    estimate(1, -1)

    estimate(0, 1)
    estimate(1, 0)