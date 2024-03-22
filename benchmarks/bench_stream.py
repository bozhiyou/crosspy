import numpy as np
import cupy as cp

size = 1024 * 1024 // 8  # 64-bit
num_gpus = 2

def _pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

on_cpu = np.random.rand(size)
pinned_on_cpu = _pin_memory(on_cpu)

streams = []
for i in range(num_gpus):
    with cp.cuda.Device(i):
        streams.append(cp.cuda.Stream(non_blocking=True))

def sync():
    for s in streams:
        s.synchronize()

def fission(src):
    dsts = []
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            with streams[i]:
                dsts.append(cp.empty_like(src))

    for i in range(num_gpus):
        with cp.cuda.Device(i):
            with streams[i]:
                dsts[i].set(src)
    sync()
    return dsts

def fusion(src):
    dsts = []
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            with streams[i]:
                dsts.append(cp.empty_like(src))
                dsts[i].set(src)
    sync()
    return dsts

res1 = fission(pinned_on_cpu)
res2 = fusion(pinned_on_cpu)
