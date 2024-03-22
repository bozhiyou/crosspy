import cupy as cp
import numba
from numba import cuda

@cuda.jit
def k_masked_sum(xr, lr, sr, nr):
    sum = cuda.local.array(100, dtype=numba.float64)
    num = cuda.local.array(10, dtype=numba.int64)
    for i in range(num.shape[0]):
        num[i] = 0
    for t in range(cuda.grid(1), xr.shape[0], cuda.gridsize(1)):
        l = lr[t]
        num[l] += 1
        start = l * sr.shape[1]
        for i in range(xr.shape[1]):
            sum[start + i] += xr[t, i]

    for l in range(sr.shape[0]):
        start = l * sr.shape[1]
        for i in range(sr.shape[1]):
            cuda.atomic.add(sr, (l, i), sum[start + i])
    for l in range(nr.shape[0]):
        cuda.atomic.add(nr, l, num[l])


if __name__ == '__main__':
    import cupy as cp
    import math
    import time
    nk = 3
    points = cp.ones((2**25, 10))
    labels = cp.random.randint(nk, size=points.shape[0], dtype=cp.int64)
    dp = cp.cuda.runtime.getDeviceProperties(0)
    MAX_THREADS_PER_BLOCK = dp['maxThreadsPerBlock']
    MAX_BLOCKS_PER_SM = dp['maxBlocksPerMultiProcessor']
    NUM_SM = dp['multiProcessorCount']
    print(MAX_BLOCKS_PER_SM, NUM_SM)
    tpb = min(len(points), MAX_THREADS_PER_BLOCK)
    bpg = (len(points) + tpb - 1) // tpb
    print(tpb, bpg)
    for repeat in range(3):
        sum = cp.zeros((nk, 10))
        num = cp.zeros(nk, dtype=cp.int64)
        t = time.perf_counter()
        k_masked_sum[192, 32](points, labels, sum, num)
        t = time.perf_counter() - t
    print(cp.sum(labels == 0), cp.sum(labels == 1), sum, num)
    print(bpg, int(math.log2(bpg>>6)))
    print(t)