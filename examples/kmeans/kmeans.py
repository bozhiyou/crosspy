import crosspy as xp
from crosspy import gpu
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score
from time import time
import cupy as cp

TEST = True

VRB=True
test_w_sklearn=False
kmeans_pp = False
kmeans_maxit = 20
dim = 10
NUM_GPUS = 2
gpus = [gpu(i) for i in range(NUM_GPUS)]

# generate data
n = 1000000;
if VRB:print(f'number of points = {n}')

nc = 10;
if VRB:print(f'number of clusters = {nc}')

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
X = xp.array(Xh, distribution=gpus, axis=0)
C = xp.array(Ch, distribution=gpus, axis=0)

print('allocating memory')
D = xp.zeros((n,nc), distribution=gpus)
W = xp.zeros((n,dim), distribution=gpus)
L = xp.zeros(n, distribution=gpus)
L0= xp.zeros(n, distribution=gpus)

print('kmeans iteration')
tic = time()
for j in range(kmeans_maxit):
    for k in range(nc):
        W = X-C[k,:]
        D[:,k] = cp.sum(W*W,axis=1)

    L = cp.argmin(D,axis=1)
    if j==0:L0=L

    for k in range(nc):
        m = cp.sum(L==k)
        with m.device:
            assert m.item() != 0, "centers should be unique; repetition: %s in %s" % (cpos[k], cpos)
        ck = 1/m *cp.sum(X[L==k,:],axis=0)
        C[k,:]=ck

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

    