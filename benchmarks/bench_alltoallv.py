"""
alltoall
src size,dst size,p,q,time,setup
2,2,2,2,0.06488621440560867,3.0094677549786866
2,2,2,4,0.100246454627874,2.31635069695767
2,2,4,2,0.09332001368359973,2.1165932500734925
2,2,4,4,0.07191374967806041,3.2439857539720833
1,1,1,1,0.022727476665750146,1.255340619944036
1,1,1,2,0.05778458198377242,0.91210419498384
1,1,1,4,0.07019886133881907,1.0714828080963343

alltoallv
src size,dst size,p,q,time
1,1,1,1,0.19659508764743805
1,1,1,2,0.2958860929744939
1,1,1,4,0.3810058004843692
1,1,2,1,0.32860612869262695
1,1,2,2,0.3669296423904598
1,1,2,4,0.3560942439362407
1,1,4,1,CUDARuntimeError: cudaErrorInvalidAddressSpace: operation not supported on global/shared address space
1,1,4,2,0.4569302049155037
1,1,4,4,0.39634154302378494
2,2,1,1,0.4222708029362063
2,2,1,2,0.6683115952958664
2,2,1,4,0.6904587103053927
2,2,2,1,0.6615510421494643
2,2,2,2,0.7624605963937938
2,2,2,4,0.8272729429105917
2,2,4,1,CUDADriverError: CUDA_ERROR_INVALID_ADDRESS_SPACE: operation not supported on global/shared address space
2,2,4,2,0.8641329489958783
2,2,4,4,0.8553683892823756
4,4,1,1,OOM
4,4,1,2,1.542395552309851
4,4,1,4,1.5956004702796538
4,4,2,1,OOM
4,4,2,2,1.643076502873252
4,4,2,4,1.5891047273762524
4,4,4,1,OOM
4,4,4,2,1.9290276933461428
4,4,4,4,1.7938273640659947
"""
import numpy as np
import cupy as cp
import crosspy as xp

import time

GB64 = 2**(30 - 3)

def _pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def simple_uniform_cut(arr, num_gpus) -> list:
    arr = _pin_memory(arr)
    src_parts = [arr[i * (len(arr) // num_gpus):(
        (i + 1) * (len(arr) // num_gpus) if i + 1 < num_gpus else None
    )] for i in range(num_gpus)]

    dst_parts = []


    for i in range(num_gpus):
        with cp.cuda.Device(i):
            with cp.cuda.Stream(non_blocking=True):
                x = cp.empty_like(src_parts[i])
                x.set(src_parts[i])
                dst_parts.append(x)
    return dst_parts


def equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b):
    distribution = np.tile(range(ngpu_b), size_b // ngpu_b + 1)[:size_b]
    indices = np.arange(size_a)
    return np.concatenate([indices[distribution == i] for i in range(ngpu_b)])


def bench_alltoallv(size_a=12, size_b=12, ngpu_a=2, ngpu_b=2):
    a = np.random.rand(size_a)
    # indices = np.random.choice(size_a, size_b, replace=False)
    indices = equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b)
    b = a[indices]

    a_parts = simple_uniform_cut(a, ngpu_a)
    xa = xp.array(a_parts, axis=0)
    b.fill(0)  # clear answers
    b_parts = simple_uniform_cut(b, ngpu_b)
    xb = xp.array(b_parts, axis=0)

    times = []
    for i in range(5):
        start = time.perf_counter()
        xp.alltoallv(xa, indices, xb)
        end = time.perf_counter()
        # print(','.join(str(x) for x in [a.nbytes/2**30, b.nbytes/2**30, ngpu_a, ngpu_b, end - start]))
        times.append(end - start)
    trimmean_time = sum(sorted(times)[1:-1]) / 3
    return trimmean_time


def bench_alltoall(size_a=12, size_b=12, ngpu_a=2, ngpu_b=2):
    a = np.random.rand(size_a)
    # indices = np.random.choice(size_a, size_b, replace=False)
    indices = equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b)
    permute = np.arange(size_b)
    b = a[indices]
    times = []
    for i in range(5):
        time.sleep(1)
        start = time.perf_counter()
        b[permute] = a[indices]
        end = time.perf_counter()
        # print(','.join(str(x) for x in [a.nbytes/2**30, b.nbytes/2**30, ngpu_a, ngpu_b, end - start]))
        times.append(end - start)
    trimmean_time = sum(sorted(times)[1:-1]) / 3
    print(trimmean_time)

    from crosspy.core.x1darray import X1D
    a_parts = simple_uniform_cut(a, ngpu_a)
    xa = X1D(a_parts, axis=0)
    b.fill(0)  # clear answers
    b_parts = simple_uniform_cut(b, ngpu_b)
    xb = X1D(b_parts, axis=0)

    setup_start = time.perf_counter()
    assignment = xp.alltoall(xb, permute, xa, indices)
    setup_end = time.perf_counter()
    times = []
    for i in range(5):
        time.sleep(1)
        start = time.perf_counter()
        assignment()
        end = time.perf_counter()
        # print(','.join(str(x) for x in [a.nbytes/2**30, b.nbytes/2**30, ngpu_a, ngpu_b, end - start]))
        times.append(end - start)
    trimmean_time = sum(sorted(times)[1:-1]) / 3
    return trimmean_time, setup_end - setup_start


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=2**(30 - 3), help='Size of source matrix')
    parser.add_argument('-m', type=int, default=2**(30 - 3), help='Size of indices and target matrix')
    parser.add_argument('-p', type=int, default=4, help='Number of GPUs for source matrix')
    parser.add_argument('-q', type=int, default=4, help='Number of GPUs for target matrix')
    args = parser.parse_args()
    # bench_alltoallv(args.n, args.m, args.p, args.q)

    # print(','.join(['src size', 'dst size', 'p', 'q', 'time']))
    # # for x in (4, 2, 1):
    # #     n = m = x * GB64
    # #     for p in (1, 2, 4,):
    # #         for q in (1, 2, 4,):
    # #             print(','.join(str(x) for x in (x, x, p, q, bench_alltoallv(n, m, p, q))))

    # single test
    x, p, q = 1, 4, 2
    n = m = x * GB64
    print(','.join(str(_) for _ in (x, x, p, q, bench_alltoall(n, m, p, q))))

    # print(','.join(['src size', 'dst size', 'p', 'q', 'time', 'setup']))
    # for x in (4, 2, 1):
    #     if x >= 2:
    #         continue
    #     n = m = x * GB64
    #     for p in (1, 2, 4,):
    #         if x >= 2 and p <= 2:
    #             continue
    #         for q in (1, 2, 4,):
    #             if x >= 2 and q <= 2:
    #                 continue
    #             print(x, p, q)
    #             print(','.join(str(x) for x in (x, x, p, q, *bench_alltoall(n, m, p, q))))
