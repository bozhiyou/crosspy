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


def equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b):
    """Round Robin"""
    distribution = np.tile(range(ngpu_b), size_b // ngpu_b + 1)[:size_b]
    indices = np.arange(size_a)
    return np.concatenate([indices[distribution == i] for i in range(ngpu_b)])


def test_alltoallv(size_a=12, size_b=12, ngpu_a=2, ngpu_b=2):
    a = np.random.rand(size_a)
    indices = np.random.choice(size_a, size_b)
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


def test_alltoints():
    from crosspy import gpu
    n = m = 20

    a = np.random.rand(n)
    indices = np.random.choice(n, m)
    b = a[indices]

    xa = xp.array(a, distribution=[gpu(0)], axis=0)
    xb = xp.array(b, distribution=[gpu(0)], axis=0)
    xp.all2ints(xb, xa, indices, debug=True)


def test_instoall():
    n = m = 20

    a = cp.random.rand(n)
    indices = cp.random.choice(n, m)
    b = a[indices]
    xa = xp.array([a], axis=0)
    xb = xp.array([b], axis=0)
    xp.alltoallv(xa, np.arange(len(xa)), xb[indices])

def test_alltoall(size_a=12, size_b=12, ngpu_a=2, ngpu_b=2):
    a = np.random.rand(size_a)
    indices = np.random.choice(size_a, size_b, replace=False)
    indices = equally_distributed_indices(size_a, size_b, ngpu_a, ngpu_b)
    permute = np.arange(size_b)
    np.random.shuffle(permute)
    a_indices = indices[permute]
    b = a[a_indices]
    b[permute] = a[a_indices]

    a_parts = simple_uniform_cut(a, ngpu_a)
    xa = xp.array(a_parts, axis=0)
    # b.fill(0)  # clear answers
    b_parts = simple_uniform_cut(b, ngpu_b)
    xb = xp.array(b_parts, axis=0)

    assignment = xp.alltoall(xb, permute, xa, a_indices, debug=True)
    assignment()
    print(','.join(str(x) for x in [xa, xb, ngpu_a, ngpu_b]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=12, help='Size of source matrix')
    parser.add_argument('-m', type=int, default=12, help='Size of indices and target matrix')
    parser.add_argument('-p', type=int, default=4, help='Number of GPUs for source matrix')
    parser.add_argument('-q', type=int, default=4, help='Number of GPUs for target matrix')
    args = parser.parse_args()
    # test_alltoallv(args.n, args.m, args.p, args.q)
    # test_alltoints()
    # test_instoall()
    test_alltoall()
    # import asyncio
    # async def async_test():
    #     test_alltoall()
    # asyncio.run(async_test())
