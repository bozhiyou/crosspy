"""
Invariance
    Output is adding one dimension to the original data object, which is the outter list.
"""

import time
import numpy as np
import crosspy as xp
from crosspy import cupy as cp
from crosspy import partition, PartitionScheme, cpu, gpu

npa = np.arange(3)
cpa = cp.arange(3)

def test_singleton():
    ap = partition(npa, distribution=[gpu(0)])
    assert isinstance(ap, list)
    assert len(ap) == 1
    assert all(p.device == d for p, d in zip(ap, [gpu(0).cupy_device]))
    ap = partition(cpa, distribution=[cpu(0)])
    assert isinstance(ap, list)
    assert len(ap) == 1
    assert isinstance(ap[0], type(npa))
    import crosspy as xp
    a1_crosspy = xp.array(np.asarray([np.arange(16)]), distribution=[gpu(0)])
    a1_crosspy.debug_print()

def test_singleton_list():
    ap = partition([npa], distribution=[gpu(0)])
    assert isinstance(ap, list)
    assert len(ap) == 1
    assert isinstance(ap[0], list)

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
    test_singleton()
    test_singleton_list()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=4, help='Size of matrix')
    parser.add_argument('-ngpus', type=int, default=2)
    args = parser.parse_args()
    
    test_partition_scheme(args.n, args.ngpus)