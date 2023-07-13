"""
Invariance
    Output is adding one dimension to the original data object, which is the outter list.
"""

import time
import numpy as np
import crosspy as xp
from crosspy import cupy as cp
from crosspy import split, PartitionScheme, cpu, gpu

npa = np.arange(3)
cpa = cp.arange(3)

def test_by_device():
    npa = np.arange(10)
    cpa = np.arange(10)
    devices = [gpu(1), cpu(0), gpu(0)]
    for obj in (npa, cpa):
        parts = split(obj, distribution=devices)
        assert isinstance(parts, list)
        assert len(parts) == len(devices)
        assert parts[0].device.id == 1
        assert len(parts[0]) == 4
        assert isinstance(parts[1], np.ndarray)
        assert len(parts[1]) == 3
        assert parts[2].device.id == 0
        assert len(parts[2]) == 3


def test_by_size():
    npa = np.arange(10)
    cpa = np.arange(10)
    good_size = range(5)
    under_size = [1]
    over_size = [1, 5, 6]
    for obj in (npa, cpa):
        for sz in (good_size, under_size, over_size):
            parts = split(obj, distribution=sz)
            assert isinstance(parts, list)
            assert len(parts) == len(list(sz))
            assert all(isinstance(p, np.ndarray) for p in parts) or all(
                p.device.id == obj.device.id for p in parts
            )
            assert all(
                len(parts[i]) == list(sz)[i] for i in range(len(parts))
                ) or (sz == over_size and all(
                len(parts[i]) <= list(sz)[i] for i in range(len(parts))
                ))
    for obj in (npa, cpa):
        for raw_sz in (good_size, under_size, over_size):
            for sz in (np.array(raw_sz), cp.array(raw_sz)):
                parts = split(obj, distribution=sz)
                assert isinstance(parts, list)
                assert len(parts) == len(sz)
                assert all(isinstance(p, np.ndarray) for p in parts) or all(
                    p.device.id == sz.device.id for p in parts
                )
                assert all(
                    len(parts[i]) == sz[i] for i in range(len(parts))
                    ) or (raw_sz == over_size and all(
                    len(parts[i]) <= sz[i] for i in range(len(parts))
                    ))


def test_by_pair():
    npa = np.arange(10)
    cpa = np.arange(10)

    for obj in (npa, cpa):
        from_zip = zip([gpu(1)] * 5, range(5))
        parts = split(obj, distribution=from_zip)
        assert isinstance(parts, list)
        assert len(parts) == 5
        assert all(p.device.id == 1 for p in parts)
        assert all(len(parts[i]) == i for i in range(5))

    from_dict = {
        gpu(1): 1,
        cpu(0): 2,
        gpu(0): 3,
    }.items()
    for obj in (npa, cpa):
        parts = split(obj, distribution=from_dict)
        assert isinstance(parts, list)
        assert len(parts) == 3
        assert parts[0].device.id == 1
        assert len(parts[0]) == 1
        assert isinstance(parts[1], np.ndarray)
        assert len(parts[1]) == 2
        assert parts[2].device.id == 0
        assert len(parts[2]) == 3


def test_singleton_list():
    ap = split([npa], distribution=[gpu(0)])
    assert isinstance(ap, list)
    assert len(ap) == 1
    assert isinstance(ap[0], cp.ndarray)
    assert ap[0].shape == (1, len(npa))

def test_partition_scheme(n=4, ngpus=2):
    matrix_size = n * n

    a0 = np.random.rand(matrix_size).astype('f') 
    partition = PartitionScheme(matrix_size)
    for i in range(0, n):
        partition[i*n:(i+1)*n] = gpu(i % ngpus)
    a0_crosspy = xp.array(a0, distribution=partition, axis=0)
    in_blocks = a0_crosspy

    assert n >= 4
    up = [i for i in range(n//2)]
    down = [i for i in range(n, n + (n//2))]
    interior = [i for i in range(2 * n, 2 * n + (n//2) * (n//2))]

    stream = cp.cuda.get_current_stream()
    s1 = time.perf_counter()
    in_block = in_blocks[interior + up + down]
    in_block = in_block.to(gpu(1))
    print("In block len: {}".format(len(in_block)), flush=True)
    stream.synchronize()
    e1 = time.perf_counter()
    t1 = e1 - s1
    print("Done copying: {}".format(t1), flush=True)



if __name__ == '__main__':
    test_by_device()
    test_by_size()
    test_by_pair()
    test_singleton_list()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=4, help='Size of matrix')
    parser.add_argument('-ngpus', type=int, default=2)
    args = parser.parse_args()
    
    # test_partition_scheme(args.n, args.ngpus)