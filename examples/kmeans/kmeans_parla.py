import nvtx

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace, GPUEnvironment
from parla.common.globals import get_current_context, get_current_stream
from parla.cython.device_manager import gpu

import crosspy as xp
from crosspy.device import GPUDevice
GPUDevice.register(GPUEnvironment)
import crosspy.dispatcher
crosspy.dispatcher.set(spawn, AtomicTaskSpace("CrossPy"))

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score
from time import time
import cupy as cp

TEST = False

VRB=True
test_w_sklearn=False
kmeans_pp = False
kmeans_maxit = 20
dim = 10
NUM_GPUS = 2

# generate data
n = 1000000;
if VRB:print(f'number of points = {n}')

nc = 10;
if VRB:print(f'number of clusters = {nc}')

def main():
    @spawn(placement=[tuple(gpu(i) for i in range(NUM_GPUS))])
    def _():
        context = get_current_context()
        print('Generating data')
        if TEST:
            with open('tests/data/kmeans_points_1Mx10_10c.npy', 'rb') as f:
                Xh = np.load(f)
            with open('tests/data/kmeans_labels_1Mx10_10c.npy', 'rb') as f:
                labels_ex = np.load(f)
            with open('tests/data/kmeans_centers_1Mx10_10c.npy', 'rb') as f:
                cpos = np.load(f)
        else:
            Xh, labels_ex = make_blobs(n_samples=n, centers=nc,
                                n_features=dim)

            # generate initial guess for clusters
            cpos  = np.random.randint(0,n,nc)
        Ch = Xh[cpos,:]
        nvtx.push_range(f"xp array creation", color="yellow", domain="kmeans")
        X = xp.array(Xh, distribution=context.devices, axis=0)
        C = xp.array(Ch, distribution=context.devices, axis=0)

        print('allocating memory')
        D = xp.zeros((n,nc), device=context.devices, axis=0)
        W = xp.zeros((n,dim), device=context.devices, axis=0)
        L = xp.zeros(n, device=context.devices, axis=0)
        L0= xp.zeros(n, device=context.devices, axis=0)
        nvtx.pop_range(domain="kmeans")

        for i in range(NUM_GPUS):
            cp.cuda.Device(i).synchronize()

        print('kmeans iteration')
        tic = time()
        for j in range(kmeans_maxit):
            if j < 2: nvtx.push_range(f"Iteration {j}", color="yellow", domain="iteration")
            if j < 2: nvtx.push_range(f"distance {j}", color="orange", domain="kmeans")
            for k in range(nc):
                if j < 2: nvtx.push_range(f"subtract Center {k} Iteration {j}", color="rapids", domain="kmeans")
                W = X-C[k,:]
                if j < 2: nvtx.pop_range(domain="kmeans")
                if j < 2: nvtx.push_range(f"multiply Center {k} Iteration {j}", color="rapids", domain="kmeans")
                WW = W*W
                if j < 2: nvtx.pop_range(domain="kmeans")
                # nvtx.push_range(f"sum Center {k} Iteration {j}", color="rapids", domain="kmeans")
                S = cp.sum(WW,axis=1)
                # nvtx.pop_range(domain="kmeans")
                if j < 2: nvtx.push_range(f"writeback Center {k} Iteration {j}", color="rapids", domain="kmeans")
                D[:,k] = S
                if j < 2: nvtx.pop_range(domain="kmeans")
            if j < 2: nvtx.pop_range(domain="kmeans")

            if j < 2: nvtx.push_range(f"argmin {j}", color="blue", domain="kmeans")
            L = cp.argmin(D,axis=1)
            if j < 2: nvtx.pop_range(domain="kmeans")
            if j==0:L0=L

            for k in range(nc):
                if j < 2: nvtx.push_range(f"update {j}", color="cyan", domain="kmeans")
                m = cp.sum(L==k)
                # with m.device:
                #     assert m.item() != 0, "centers should be unique; repetition: %s in %s" % (cpos[k], cpos)
                ck = 1/m *cp.sum(X[L==k,:],axis=0)
                C[k,:]=ck
                if j < 2: nvtx.pop_range(domain="kmeans")
            if j < 2: nvtx.pop_range(domain="iteration")

        toc = time()       
        L0h= xp.asnumpy(L0)
        Lh = xp.asnumpy(L)
        if TEST:
            with open('tests/data/kmeans_L0_1Mx10_10c.npy', 'rb') as f:
                L0h_g = np.load(f)
            assert np.array_equal(L0h, L0h_g)
            with open('tests/data/kmeans_L_1Mx10_10c.npy', 'rb') as f:
                Lh_g = np.load(f)
            assert np.array_equal(Lh, Lh_g)
        sc0 =homogeneity_score(L0h, labels_ex)
        sc =homogeneity_score(Lh, labels_ex)
        print(f'Initial homogeneity score = {sc0:.2}')
        print(f'homogeneity score = {sc:.2} (1 is best)')
        print(f'Kmeans took {toc-tic:.3} seconds')

if __name__ == '__main__':
    with Parla():
        # T = AtomicTaskSpace("quicksort")
        main()