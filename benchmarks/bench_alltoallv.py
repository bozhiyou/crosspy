import numpy as np
import cupy as cp
import crosspy as xp

import time


def simple_uniform_cut(arr, num_gpus) -> list:
    arr_parts = []
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            arr_parts.append(
                cp.asarray(
                    arr[i * (len(arr) // num_gpus):(
                        (i + 1) * (len(arr) // num_gpus) if i + 1 < num_gpus else None
                    )]
                )
            )
    return arr_parts


def bench_alltoallv(size_a=12, size_b=12, ngpu_a=2, ngpu_b=2):
    a = np.random.rand(size_a)
    print(a.nbytes/2**30)
    indices = np.random.choice(size_a, size_b, replace=False)
    b = a[indices]

    a_parts = simple_uniform_cut(a, ngpu_a)
    xa = xp.array(a_parts, axis=0)
    b.fill(0)  # clear answers
    b_parts = simple_uniform_cut(b, ngpu_b)
    xb = xp.array(b_parts, axis=0)

    start = time.time()
    xp.alltoallv(xa, indices, xb)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=2**(30-3), help='Size of source matrix')
    parser.add_argument('-m', type=int, default=2**(30-3), help='Size of indices and target matrix')
    parser.add_argument('-p', type=int, default=4, help='Number of GPUs for source matrix')
    parser.add_argument('-q', type=int, default=4, help='Number of GPUs for target matrix')
    args = parser.parse_args()
    bench_alltoallv(args.n, args.m, args.p, args.q)