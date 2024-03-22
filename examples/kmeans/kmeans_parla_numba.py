import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1000000, help="Number of points")
parser.add_argument("-nc", type=int, default=10, help="number of clusters")
parser.add_argument("-d", type=int, default=10, help="number of dimensions")
parser.add_argument("-num_gpus", "-g", type=int, default=2, help="number of GPUs")
parser.add_argument("-it", type=int, default=20, help="number of iterations (including warmup)")
parser.add_argument("--warmup", type=int, default=2, help="number of warmup iterations")
parser.add_argument("--verbose", "-v", default=False, action='store_true', help="verbose output")
parser.add_argument("--test", default=False, action='store_true', help="use static test input")
parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="compute homogeneity score")
args = parser.parse_args()

import numpy as np
import cupy as cp

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace, GPUEnvironment
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu

import crosspy as xp
from crosspy.utils import Timer
from crosspy.device import GPUDevice
import crosspy.dispatcher

GPUDevice.register(GPUEnvironment)
T = AtomicTaskSpace("CrossPy")
crosspy.dispatcher.set(spawn, AtomicTaskSpace("CrossPy"))

from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score

import numba
from numba import cuda

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# tpb should be multiples of warp size to be efficient
dp = cp.cuda.runtime.getDeviceProperties(0)
WARP_SIZE = dp['warpSize']
MAX_THREADS_PER_SM = dp['maxThreadsPerMultiProcessor']
MAX_BLOCKS_PER_SM = dp['maxBlocksPerMultiProcessor']
NUM_SM = dp['multiProcessorCount']
MAX_THREADS_PER_BLOCK = dp['maxThreadsPerBlock']


@cuda.jit
def k_clustering(xr, cr, lr):
    for t in range(cuda.grid(1), xr.shape[0], cuda.gridsize(1)):
        min_k = 0
        min_d = 0
        for j in range(cr.shape[1]):
            w = xr[t, j] - cr[0, j]
            min_d += w * w
        for k in range(1, cr.shape[0]):
            d = 0
            for j in range(cr.shape[1]):
                w = xr[t, j] - cr[k, j]
                d += w * w
                if d >= min_d:
                    break
            if d < min_d:
                min_k = k
                min_d = d
        lr[t] = min_k

@cuda.jit
def k_clear_buffer(sr, nr):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y
    if x < sr.shape[0]:
        nr[x] = 0
        if y < sr.shape[1]:
            sr[x, y] = 0


local_sum_size = args.nc * args.d
local_count_size = args.nc
@cuda.jit
def k_group_by_label(xr, lr, sr, nr):
    if cuda.grid(1) >= xr.shape[0]:
        return
    
    sum = cuda.local.array(local_sum_size, dtype=numba.float64)
    num = cuda.local.array(local_count_size, dtype=numba.int64)
    for i in range(sum.shape[0]):
        sum[i] = 0
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


def make_input():
    if args.test:
        with open('tests/data/kmeans_points_1Mx10_10c.npy', 'rb') as f:
            Xh = np.load(f)
        with open('tests/data/kmeans_labels_1Mx10_10c.npy', 'rb') as f:
            labels_ex = np.load(f)
        with open('tests/data/kmeans_centers_1Mx10_10c.npy', 'rb') as f:
            cpos = np.load(f)
    else:
        if args.verbose:
            print('Generating data', end='...', flush=True)
            t = time.perf_counter()
        Xh, labels_ex = make_blobs(
            n_samples=args.n, centers=args.nc, n_features=args.d
        )
        # generate initial guess for clusters
        cpos = np.random.randint(0, args.n, args.nc)
        if args.verbose: print(f'{time.perf_counter() - t:.3f}s')
    return Xh, labels_ex, cpos

def main():
    if args.verbose:
        print(f'number of points = {args.n}')
        print(f'number of clusters = {args.nc}')

    @spawn(placement=[tuple(gpu(i) for i in range(args.num_gpus))])
    async def _():
        context = get_current_context()
        Xh, labels_ex, cpos = make_input()
        Ch = Xh[cpos, :]
        X = xp.array(Xh, distribution=context.devices, axis=0)

        if args.verbose:
            print('allocating memory buffers', end='...', flush=True)
            t = time.perf_counter()
        L = xp.zeros(args.n, device=context.devices, axis=0, dtype=cp.int64)
        L0 = xp.zeros(args.n, device=context.devices, axis=0)
        C = xp.zeros((args.num_gpus, args.nc, args.d), device=context.devices, axis=0)
        N = xp.zeros((args.num_gpus, args.nc), device=context.devices, axis=0, dtype=cp.int64)
        if args.verbose: print(f'{time.perf_counter() - t:.3f}s')

        for d in context.devices:
            d.synchronize()

        t_relabel = Timer()
        t_reduce = Timer()
        t_cpu = Timer()

        for j in range(args.it):
            if j == args.warmup: tic = time.perf_counter()
            if args.verbose: print('kmeans iteration', j)

            if j >= args.warmup: t_relabel.start()
            kT = AtomicTaskSpace(f"clustering_{j}")
            for x, l, cb in zip(X.block_view(), L.block_view(), C.block_view()):
                tpb = min(x.shape[0], MAX_THREADS_PER_BLOCK)
                bpg = (x.shape[0] + tpb - 1) // tpb
                @spawn(kT[len(kT)], placement=[gpu(x.device.id)])
                def _():
                    with x.device:
                        with cp.cuda.Stream(non_blocking=True) as s:
                            cb[0].set(Ch, stream=s)
                            ks = cuda.external_stream(s.ptr)
                            k_clustering[bpg, tpb, ks](x, cb[0], l)
                        s.synchronize()
            await kT
            if j >= args.warmup: t_relabel.stop()
            if j == 0: L0[:] = L

            if j >= args.warmup: t_reduce.start()
            for x, l, cb, dn in zip(X.block_view(), L.block_view(), C.block_view(), N.block_view()):
                @spawn(kT[len(kT)], placement=[gpu(x.device.id)])
                def _():
                    with x.device:
                        with cp.cuda.Stream(non_blocking=True) as s:
                            ks = cuda.external_stream(s.ptr)
                            k_clear_buffer[((cb.shape[0] + MAX_THREADS_PER_BLOCK - 1) // MAX_THREADS_PER_BLOCK, cb.shape[1]), MAX_THREADS_PER_BLOCK, ks](cb[0], dn[0])
                            k_group_by_label[3 * NUM_SM, WARP_SIZE, ks](x, l, cb[0], dn[0])
                        s.synchronize()
            await kT
            if j >= args.warmup: t_cpu.start()
            Ch[:] = np.asarray(C).sum(axis=0) / np.asarray(N).sum(axis=0).reshape(args.nc, 1)
            if j >= args.warmup: t_cpu.stop()
            if j >= args.warmup: t_reduce.stop()
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
        if args.verify:
            print("Verifying homogeneity score", end='...', flush=True)
            t = time.perf_counter()
            sc0 = homogeneity_score(L0h, labels_ex)
            sc = homogeneity_score(Lh, labels_ex)
            print(f'{time.perf_counter() - t:.3f}s')
            print(f'Initial homogeneity score = {sc0:.2}')
            print(f'homogeneity score = {sc:.2} (1 is best)')

        print(
            f'Kmeans took {toc-tic:.3} seconds, {(toc-tic) / (args.it - args.warmup):.3} per iteration',
            f"({t_relabel.get() / (args.it - args.warmup):.3f} {t_reduce.get() / (args.it - args.warmup):.3f} {t_cpu.get() / (args.it - args.warmup):.3f})"
        )


if __name__ == '__main__':
    with Parla():
        main()