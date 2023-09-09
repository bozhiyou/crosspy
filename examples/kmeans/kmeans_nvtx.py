import nvtx
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1000000, help="Number of points")
parser.add_argument("-nc", type=int, default=10, help="number of clusters")
parser.add_argument("-d", type=int, default=10, help="number of dimensions")
parser.add_argument("-num_gpus", "-g", type=int, default=2, help="number of GPUs")
parser.add_argument("-it", type=int, default=20, help="number of iterations")
parser.add_argument("--warmup", type=int, default=2, help="number of warmup iterations")
parser.add_argument("--verbose", "-v", default=False, action='store_true', help="verbose output")
parser.add_argument("--test", default=False, action='store_true', help="use static test input")
args = parser.parse_args()

import numpy as np
import cupy as cp

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace, GPUEnvironment
from parla.common.globals import get_current_context
from crosspy import gpu

import crosspy as xp
from crosspy.utils import Timer
from crosspy.device import GPUDevice
import crosspy.dispatcher

GPUDevice.register(GPUEnvironment)
crosspy.dispatcher.set(spawn, AtomicTaskSpace("CrossPy"))

tdistance = Timer()
targmin = Timer()
tnewcent = Timer()

from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score

def main():
    if args.verbose:
        print(f'number of points = {args.n}')
        print(f'number of clusters = {args.nc}')

    placement=tuple(gpu(i) for i in range(args.num_gpus))
    if args.verbose: print('Generating data')
    if args.test:
        with open('tests/data/kmeans_points_1Mx10_10c.npy', 'rb') as f:
            Xh = np.load(f)
        with open('tests/data/kmeans_labels_1Mx10_10c.npy', 'rb') as f:
            labels_ex = np.load(f)
        with open('tests/data/kmeans_centers_1Mx10_10c.npy', 'rb') as f:
            cpos = np.load(f)
    else:
        Xh, labels_ex = make_blobs(
            n_samples=args.n, centers=args.nc, n_features=args.d
        )

        # generate initial guess for clusters
        cpos = np.random.randint(0, args.n, args.nc)
    Ch = Xh[cpos, :]
    X = xp.array(Xh, distribution=placement, axis=0)
    C = xp.array(Ch, distribution=placement, axis=0)

    if args.verbose: print('allocating memory')
    D = xp.zeros((args.n, args.nc), device=placement, axis=0)
    W = xp.zeros((args.n, args.d), device=placement, axis=0)
    L = xp.zeros(args.n, device=placement, axis=0)
    L0 = xp.zeros(args.n, device=placement, axis=0)

    for i in range(args.num_gpus):
        cp.cuda.Device(i).synchronize()

    for j in range(args.it):
        if j == args.warmup: tic = time.perf_counter()
        if args.verbose: print('kmeans iteration', j)
        if j >= args.warmup: tdistance.start()
        for k in range(args.nc):
            Ck = C[k, :]
            nvtx.push_range(message=f"subtract {j} {k}", color="orange", domain="nvtx")
            X_Ck = X - Ck
            nvtx.pop_range(domain="nvtx")
            W = X_Ck
            nvtx.push_range(message=f"square {j} {k}", color="yellow", domain="nvtx")
            WW = W * W
            nvtx.pop_range(domain="nvtx")
            nvtx.push_range(message=f"sum {j} {k}", color="rapids", domain="nvtx")
            Ws = cp.sum(WW, axis=1)
            nvtx.pop_range(domain="nvtx")
            D[:, k] = Ws
        if j >= args.warmup: tdistance.stop()

        if j >= args.warmup: targmin.start()
        L = cp.argmin(D, axis=1)
        if j >= args.warmup: targmin.stop()
        if j == 0: L0 = L

        if j >= args.warmup: tnewcent.start()
        for k in range(args.nc):
            m = cp.sum(L == k)
            # with m.device:
            #     assert m.item() != 0, "centers should be unique; repetition: %s in %s" % (cpos[k], cpos)
            ck = 1 / m * cp.sum(X[L == k, :], axis=0)
            C[k, :] = ck
        if j >= args.warmup: tnewcent.stop()
    toc = time.perf_counter()

    L0h = xp.asnumpy(L0)
    Lh = xp.asnumpy(L)
    if args.test:
        with open('tests/data/kmeans_L0_1Mx10_10c.npy', 'rb') as f:
            L0h_g = np.load(f)
        assert np.array_equal(L0h, L0h_g)
        with open('tests/data/kmeans_L_1Mx10_10c.npy', 'rb') as f:
            Lh_g = np.load(f)
        assert np.array_equal(Lh, Lh_g)
    sc0 = homogeneity_score(L0h, labels_ex)
    sc = homogeneity_score(Lh, labels_ex)
    print(f'Initial homogeneity score = {sc0:.2}')
    print(f'homogeneity score = {sc:.2} (1 is best)')

    print(
        f'Kmeans took {toc-tic:.3} seconds, {(toc-tic) / (args.it - args.warmup):.3} per iteration',
        f"({tdistance.get() / (args.it - args.warmup):.3f} {targmin.get() / (args.it - args.warmup):.3f} {tnewcent.get() / (args.it - args.warmup):.3f})"
    )


if __name__ == '__main__':
    with Parla():
        main()