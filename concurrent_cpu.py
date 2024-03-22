import cupy
import numpy as np
from time import perf_counter as time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', default=4, type=int)
args = parser.parse_args()


N = 2**23
d = 128
A = np.random.rand(N, d)

pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

A_pinned = _pin_memory(A)
A_gpu = []
B_gpu = []

streams = []

for i in range(args.ngpus):
    with cupy.cuda.Device(i):
        A_gpu.append(cupy.ndarray(A.shape, dtype=A.dtype))
        streams.append(cupy.cuda.Stream(non_blocking=True))
        cupy.cuda.Device().synchronize()

t = time()
for i in range(args.ngpus):
    with cupy.cuda.Device(i):
        with streams[i]:
            A_gpu[i].set(A_pinned)

for j in range(args.ngpus):
    i = args.ngpus - j - 1
    with cupy.cuda.Device(i):
        streams[i].synchronize()

t = time() - t
print("Time: ", t)
