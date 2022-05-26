# import logging
# logging.basicConfig(level=logging.DEBUG,
# format=
#     '%(asctime)s [\033[1;4m%(levelname)s\033[0m %(processName)s:%(threadName)s] %(filename)s:%(lineno)s %(message)s'
# )

import crosspy as xp
from crosspy import gpu
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score
from time import time
import cupy as cp

VRB=True
test_w_sklearn=False
kmeans_pp = False
kmeans_maxit = 20
dim = 2
num_gpus = 4

# generate data
n = 1000000
if VRB:print(f'number of points = {n}')

nc = 2
if VRB:print(f'number of clusters = {nc}')

print('Generating data')
Xh, labels_ex = make_blobs(n_samples=n, centers=nc,
                     n_features=dim)

# generate initial guess for clusters
cpos  = np.random.randint(0,n,nc)
Ch = Xh[cpos,:]
X = xp.array(Xh, placement=[gpu(i) for i in range(num_gpus)]) 
C = xp.array(Ch, placement=[gpu(i) for i in range(num_gpus)])

print('allocating memory')
D = xp.zeros((n,nc), placement=[gpu(i) for i in range(num_gpus)])
W = xp.zeros((n,dim), placement=[gpu(i) for i in range(num_gpus)])
L = xp.zeros(n, placement=[gpu(i) for i in range(num_gpus)])
L0= xp.zeros(n, placement=[gpu(i) for i in range(num_gpus)])

print('kmeans iteration')
tic = time()
for j in range(kmeans_maxit):
    for k in range(nc):
        Z = C[k,:] # (dim,)
        # toc = time()       
        # print(f'get Kmeans took {toc-tic:.3} seconds')
        W = X-Z # (n, dim)
        # toc = time()       
        # print(f'- Kmeans took {toc-tic:.3} seconds')
        s = cp.sum(W*W,axis=1) # (n,)
        # toc = time()       
        # print(f'*sum Kmeans took {toc-tic:.3} seconds')
        D[:,k] = s
        # toc = time()       
        # print(f'set Kmeans took {toc-tic:.3} seconds')
    toc = time()       
    print(f'A Kmeans took {toc-tic:.3} seconds')

    L = cp.argmin(D,axis=1) # (n,)
    if j==0:L0=L

    for k in range(nc):
        m = cp.sum(L==k)
        with m.item().device:
            assert m.item() != 0, "cannot handle corner case when cpos has repetitive points %s" % cpos
        ck = 1/m
        II = L==k
        XX = X[II,:]
        ck = ck *cp.sum(XX,axis=0)
        C[k,:]=ck

toc = time()       
L0h= xp.asnumpy(L0)
Lh = xp.asnumpy(L)
sc0 =homogeneity_score(L0h, labels_ex)
sc =homogeneity_score(Lh, labels_ex)
print(f'Initial homogeneity score = {sc0:.2}')
print(f'homogeneity score = {sc:.2} (1 is best)')
print(f'Kmeans took {toc-tic:.3} seconds')

    