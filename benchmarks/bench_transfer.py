import time
import cupy as cp

GB32 = 2**(30-2)
GB64 = 2**(30-3)

SIZE = int(2 * GB64)

with cp.cuda.Device(0):
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    stream3 = cp.cuda.Stream(non_blocking=True)
    a = cp.random.rand(SIZE)
    b = cp.random.rand(SIZE)
    c = cp.random.rand(SIZE)
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)

def sync_test():
    t = time.perf_counter()
    with cp.cuda.Device(1):
        aa = cp.asarray(a)
        bb = cp.asarray(b)
        cc = cp.asarray(c)
    cp.cuda.Stream.null.synchronize()
    tt = time.perf_counter()
    print("copy:", tt - t)

def async_test():
    t = time.perf_counter()
    with cp.cuda.Device(1):
        with cp.cuda.Stream(non_blocking=True) as stream1:
            aa = cp.asarray(a)
        with cp.cuda.Stream(non_blocking=True) as stream2:
            bb = cp.asarray(b)
    stream1.synchronize()
    stream2.synchronize()
    tt = time.perf_counter()
    print("copy w/ stream:", tt - t)

def copy_from_test():
    t = time.perf_counter()
    with cp.cuda.Device(1):
            aa = cp.empty(a.shape, dtype=a.dtype)
            aa.data.copy_from_device(a.data, a.nbytes)
            bb = cp.empty(b.shape, dtype=b.dtype)
            bb.data.copy_from_device(b.data, b.nbytes)
    cp.cuda.Stream.null.synchronize()
    tt = time.perf_counter()
    print("copy_from_device", tt - t)

def copy_from_async_test():
    t = time.perf_counter()
    with cp.cuda.Device(1):
        aa = cp.empty(a.shape, dtype=a.dtype)
        bb = cp.empty(b.shape, dtype=b.dtype)
        with cp.cuda.Stream(non_blocking=True) as stream3:
            aa.data.copy_from_device_async(a.data, a.nbytes, stream=stream1)
        bb.data.copy_from_device_async(b.data, b.nbytes, stream=stream2)
        stream1.synchronize()
        stream2.synchronize()
        stream3.synchronize()
    tt = time.perf_counter()
    print("copy_from_device_async", tt - t)

copy_from_async_test()
# copy_from_test()
# sync_test()
# async_test()