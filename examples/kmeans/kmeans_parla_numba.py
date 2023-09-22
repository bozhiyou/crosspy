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
from parla.cython.device_manager import gpu

import crosspy as xp
from crosspy.utils import Timer
from crosspy.device import GPUDevice
import crosspy.dispatcher

GPUDevice.register(GPUEnvironment)
T = AtomicTaskSpace("CrossPy")
crosspy.dispatcher.set(spawn, AtomicTaskSpace("CrossPy"))

tdistance = Timer()
targmin = Timer()
tnewcent = Timer()

from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score

###### kernels
from numba import cuda
import math

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
def k_distance(xr, cr, wr, dr):
    t = cuda.grid(1)
    if t < xr.shape[0]:
        dr[t] = 0
        for i in range(xr.shape[1]):
            wr[t, i] = xr[t, i] - cr[i]
            dr[t] += wr[t, i] * wr[t, i]

@cuda.jit
def k_argmin_axis1(dr, lr):
    t = cuda.grid(1)
    if t < dr.shape[0]:
        lr[t] = 0
        v = dr[t, 0]
        for i in range(1, dr.shape[1]):
            if dr[t, i] < v:
                lr[t] = i
                v = dr[t, i]

@cuda.jit
def k_sum_1d(ar, br):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    # partition into blocks
    bpg = cuda.gridDim.x
    sz = (ar.shape[0] + bpg - 1) // bpg
    start = bx * sz
    stop = min(start + sz, ar.shape[0])

    # reduce to #tpb
    idx = start + tx
    tpb = cuda.blockDim.x
    if idx < stop:
        for i in range(idx + tpb, stop, tpb):
            ar[idx] += ar[i]
    cuda.syncthreads()

    # block reduction
    stride = (tpb >> 1)
    while stride:
        if tx < stride and (idx + stride) < stop:
            ar[idx] += ar[idx + stride]
        cuda.syncthreads()
        stride >>= 1

    if tx == 0:
        br[bx] = ar[idx]

@cuda.jit
def k_sum_2d_axis0(ar, br):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    # partition into blocks
    bpg = cuda.gridDim.x
    sz = (ar.shape[0] + bpg - 1) // bpg
    start = bx * sz
    stop = min(start + sz, ar.shape[0])

    # reduce to #tpb
    idx = start + tx
    tpb = cuda.blockDim.x
    if idx < stop:
        for i in range(idx + tpb, stop, tpb):
            for j in range(ar.shape[1]):
                ar[idx, j] += ar[i, j]
    cuda.syncthreads()

    # block reduction
    stride = (tpb >> 1)
    while stride:
        if tx < stride and (idx + stride) < stop:
            for j in range(ar.shape[1]):
                ar[idx, j] += ar[idx + stride, j]
        cuda.syncthreads()
        stride >>= 1

    if tx == 0:
        for j in range(ar.shape[1]):
            br[bx, j] = ar[idx, j]

def sum_1d(a):
    if a.dtype.kind == 'b':
        a = a.astype(cp.int64)
    with cp.cuda.Stream(non_blocking=True) as s:
        ks = cuda.external_stream(s.ptr)
        block_reduction = cuda.device_array((min(MAX_THREADS_PER_BLOCK * 2, math.ceil(a.shape[0] / MAX_THREADS_PER_BLOCK)),), dtype=a.dtype, stream=ks)
        k_sum_1d[block_reduction.shape[0], MAX_THREADS_PER_BLOCK, ks](a, block_reduction)
        result = cuda.device_array((1,), dtype=a.dtype, stream=ks)
        k_sum_1d[1, MAX_THREADS_PER_BLOCK, ks](block_reduction, result)
        result = result.copy_to_host(stream=ks)
    s.synchronize()
    return result.item()

def sum_2d_axis0(a):
    with cp.cuda.Stream(non_blocking=True) as s:
        ks = cuda.external_stream(s.ptr)
        block_reduction = cuda.device_array((min(MAX_THREADS_PER_BLOCK * 2, math.ceil(a.shape[0] / MAX_THREADS_PER_BLOCK)), a.shape[1]), dtype=a.dtype, stream=ks)
        k_sum_2d_axis0[block_reduction.shape[0], MAX_THREADS_PER_BLOCK, ks](a, block_reduction)
        result = cuda.device_array((1, a.shape[1]), dtype=a.dtype, stream=ks)
        k_sum_2d_axis0[1, MAX_THREADS_PER_BLOCK, ks](block_reduction, result)
        result = result.copy_to_host(stream=ks)
    s.synchronize()
    return result[0]

###### end kernels

def main():
    if args.verbose:
        print(f'number of points = {args.n}')
        print(f'number of clusters = {args.nc}')

    @spawn(placement=[tuple(gpu(i) for i in range(args.num_gpus))])
    async def _():
        context = get_current_context()
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
        X = xp.array(Xh, distribution=context.devices, axis=0)
        C = xp.array(Ch, distribution=context.devices, axis=0)

        if args.verbose: print('allocating memory')
        D = xp.zeros((args.n, args.nc), device=context.devices, axis=0)
        W = xp.zeros((args.n, args.d), device=context.devices, axis=0)
        L = xp.zeros(args.n, device=context.devices, axis=0)
        L0 = xp.zeros(args.n, device=context.devices, axis=0)

        for i in range(args.num_gpus):
            cp.cuda.Device(i).synchronize()

        for j in range(args.it):
            if j == args.warmup: tic = time.perf_counter()
            if args.verbose: print('kmeans iteration', j)
            if j >= args.warmup: tdistance.start()
            dT = AtomicTaskSpace(f"distance_{j}")
            for k in range(args.nc):
                c = C[k]
                for x, w, d in zip(X.block_view(), W.block_view(), D.block_view()):
                    @spawn(dT[len(dT)], placement=[gpu(x.device.id)])
                    def _():
                        with x.device:
                            with cp.cuda.Stream(non_blocking=True) as s:
                                c_ = cp.empty_like(c)
                                c_.data.copy_from_device_async(c.data, c.nbytes, stream=s)
                                ks = cuda.external_stream(s.ptr)
                                k_distance[math.ceil(x.shape[0] / 32), 32, ks](x, c_, w, d[:, k])
                            s.synchronize()
                # W[:] = X - C[k, :]
                # D[:, k] = cp.sum(W * W, axis=1)
            await dT
            if j >= args.warmup: tdistance.stop()

            if j >= args.warmup: targmin.start()
            mT = AtomicTaskSpace(f"cluster_{j}")
            for l, d in zip(L.block_view(), D.block_view()):
                @spawn(mT[len(mT)], placement=[gpu(d.device.id)])
                def _():
                    with d.device:
                        with cp.cuda.Stream(non_blocking=True) as s:
                            ks = cuda.external_stream(s.ptr)
                            k_argmin_axis1[math.ceil(x.shape[0] / 32), 32, ks](d, l)
                        s.synchronize()
            await mT
            # L[:] = cp.argmin(D, axis=1)
            if j >= args.warmup: targmin.stop()
            if j == 0: L0[:] = L

            if j >= args.warmup: tnewcent.start()
            cT = AtomicTaskSpace(f"center_{j}")
            for k in range(args.nc):
                M = (L == k)
                XM = X[M, :]
                m = [0 for _ in range(M.nparts)]
                for i, m_ in enumerate(M.block_view()):
                    @spawn(cT[len(cT)], placement=[gpu(m_.device.id)])
                    def _():
                        with m_.device:
                            m[i] = sum_1d(m_)
                await cT
                m = sum(m)
                # m = cp.sum(M)
                # with m.device:
                #     assert m.item() != 0, "centers should be unique; repetition: %s in %s" % (cpos[k], cpos)
                ck = cp.zeros_like(C[k, :])
                res = [0 for _ in range(XM.nparts)]
                for i, xm in enumerate(XM.block_view()):
                    @spawn(cT[len(cT)], placement=[gpu(xm.device.id)])
                    def _():
                        with xm.device:
                            ck_ = sum_2d_axis0(xm)
                        with ck.device:
                            res[i] = cp.asarray(ck_)
                await cT
                for x in res:
                    ck += x
                ck = 1 / m * ck
                # ck = 1 / m * cp.sum(XM, axis=0)
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