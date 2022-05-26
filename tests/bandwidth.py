import numpy as np
import cupy as cp

from timeit import timeit
from warnings import warn

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

def estimate(source, destination, size=10**6, samples=20):
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
    
def movement():
    # warmup
    for i in range(2):
        with cp.cuda.Device(i):
            cp.cuda.get_current_stream().synchronize()

    array = make_data(0, 10**6)
    import time
    t = time.perf_counter()
    with cp.cuda.Device(1):
        with cp.cuda.Stream(non_blocking=True) as stream:
            membuffer = cp.empty(array.shape, dtype=array.dtype)
            membuffer.data.copy_from_device_async(array.data, array.nbytes, stream=stream)
            stream.synchronize()
    tt = time.perf_counter()
    print(tt-t)

    t = time.perf_counter()
    with cp.cuda.Device(1):
        with cp.cuda.Stream(non_blocking=True) as stream:
            membuffer = cp.empty(array.shape, dtype=array.dtype)
            membuffer.data.copy_from_device_async(array[0:array.shape[0]].data, array[0:array.shape[0]].nbytes, stream=stream)
            stream.synchronize()
    tt = time.perf_counter()
    print(tt-t)

    t = time.perf_counter()
    with cp.cuda.Device(0):
        with cp.cuda.Stream(non_blocking=True) as stream:
            a = cp.arange(len(array))
            stream.synchronize()
    tt = time.perf_counter()
    print(tt-t)

    t = time.perf_counter()
    with cp.cuda.Device(0):
        with cp.cuda.Stream(non_blocking=True) as stream:
            all_index = array[a]
            stream.synchronize()
    tt = time.perf_counter()
    print(tt-t)

    t = time.perf_counter()
    with cp.cuda.Device(1):
        with cp.cuda.Stream(non_blocking=True) as stream:
            membuffer = cp.empty(array.shape, dtype=array.dtype)
            membuffer.data.copy_from_device_async(all_index.data, all_index.nbytes, stream=stream)
            stream.synchronize()
    tt = time.perf_counter()
    print(tt-t)

    
if __name__ == '__main__':
    movement()
    raise

    estimate(-1, -1)

    estimate(-1, 0)

    estimate(0, -1)

    estimate(0, 1)
    estimate(1, 0)