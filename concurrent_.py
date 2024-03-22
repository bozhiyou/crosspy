import cupy
import numpy as np
from time import perf_counter as time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', default=4, type=int)
args = parser.parse_args()


N = 2**22
d = 128
A = np.random.rand(N, d)

# pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
# cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def _gpu_version(A, stream):
    B = cupy.empty_like(A)
    with stream:
        B.set(A, stream=stream)

A_pinned = _pin_memory(A)
A_gpu = []
B_gpu = []

streams = []

for i in range(args.ngpus):
    with cupy.cuda.Device(i):
        A_gpu.append(cupy.ndarray(A.shape, dtype=A.dtype))
        B_gpu.append(cupy.ndarray(A.shape, dtype=A.dtype))
        streams.append(cupy.cuda.Stream(non_blocking=True))
        cupy.cuda.Device().synchronize()

t = time()
for i in range(args.ngpus):
    with cupy.cuda.Device(i):
        with streams[i]:
            nlocal = int(len(A)/args.ngpus)
            local_source_array = A_gpu[i][i*nlocal:(i+1)*nlocal]
            local_target_array = B_gpu[0][i*nlocal:(i+1)*nlocal]
            local_target_array.data.copy_from_device_async(local_source_array.data, local_source_array.nbytes, stream=streams[i])

for i in range(args.ngpus):
    with cupy.cuda.Device(i):
        streams[i].synchronize()

t = time() - t
print("Time: ", t)
