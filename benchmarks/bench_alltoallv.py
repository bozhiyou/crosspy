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
    arr_parts = []
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            part = arr[i * (len(arr) // num_gpus):(
                (i + 1) * (len(arr) // num_gpus) if i + 1 < num_gpus else None
            )]
            dev_part = cp.empty_like(part)
            dev_part.set(part)
            arr_parts.append(dev_part)
    return arr_parts


def equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b):
    distribution = np.tile(range(ngpu_b), size_b // ngpu_b + 1)[:size_b]
    indices = np.arange(size_a)
    return np.concatenate([indices[distribution == i] for i in range(ngpu_b)])


def bench_alltoallv(size_a=12, size_b=12, ngpu_a=2, ngpu_b=2):
    a = np.random.rand(size_a)
    indices = np.random.choice(size_a, size_b, replace=False)
    indices = equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b)
    b = a[indices]

    a_parts = simple_uniform_cut(a, ngpu_a)
    xa = xp.array(a_parts, axis=0)
    b.fill(0)  # clear answers
    b_parts = simple_uniform_cut(b, ngpu_b)
    xb = xp.array(b_parts, axis=0)

    start = time.perf_counter()
    xp.alltoallv(xa, indices, xb)
    end = time.perf_counter()
    print(','.join(str(x) for x in [a.nbytes/2**30, b.nbytes/2**30, ngpu_a, ngpu_b, end - start]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=2**(30 - 3), help='Size of source matrix')
    parser.add_argument('-m', type=int, default=2**(30 - 3), help='Size of indices and target matrix')
    parser.add_argument('-p', type=int, default=4, help='Number of GPUs for source matrix')
    parser.add_argument('-q', type=int, default=4, help='Number of GPUs for target matrix')
    args = parser.parse_args()
    # bench_alltoallv(args.n, args.m, args.p, args.q)

    n = m = 1 * GB64
    p = 2
    for q in range(1, 3):
        bench_alltoallv(n, m, p, q)
