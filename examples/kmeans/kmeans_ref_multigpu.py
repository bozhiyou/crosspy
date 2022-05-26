import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import homogeneity_score
from time import time
import cupy as cp

VRB=True
test_w_sklearn=False
kmeans_pp = False
kmeans_maxit = 20
dim = 10
NUM_GPUS = 4

# generate data
n = 10000000;
if VRB:print(f'number of points = {n}')
from math import ceil
block_size = ceil(n / NUM_GPUS)

nc = 10;
if VRB:print(f'number of clusters = {nc}')

print('Generating data')
Xh, labels_ex = make_blobs(n_samples=n, centers=nc,
                     n_features=dim)

# generate initial guess for clusters
cpos  = np.random.randint(0,n,nc)
Ch = Xh[cpos,:]
X = []
C = []

print('allocating memory')
D = []
W = []
L = []
L0 = []
for i in range(NUM_GPUS):
    with cp.cuda.Device(i):
        X.append(cp.asarray(Xh[i*block_size:(i+1)*block_size]))
        C.append(cp.asarray(Ch))
        D.append(cp.zeros((block_size,nc))) # distance (num_points, num_centers)
        W.append(cp.zeros((block_size,dim))) # weight (num_points, dim)
        L.append(cp.zeros(block_size)) # label (num_points)
        L0.append(cp.zeros(block_size)) # label first iteration

print('kmeans iteration')
tic = time()
for j in range(kmeans_maxit):
    for k in range(nc):
        for i in range(NUM_GPUS):
            with cp.cuda.Device(i):
                W[i] = X[i]-C[i][k,:]
                D[i][:,k] = cp.sum(W[i]*W[i],axis=1)
    # toc = time()
    # print(f'A Kmeans took {toc-tic:.3} seconds')

    for i in range(NUM_GPUS):
        with cp.cuda.Device(i):
            L[i] = cp.argmin(D[i],axis=1)
            if j==0:L0[i]=L[i]

    for k in range(nc):
        m = None
        for i in range(NUM_GPUS):
            with cp.cuda.Device(i):
                if m is None:
                    m = cp.sum(L[i]==k) # int, cluster_size
                else:
                    m = cp.asarray(m) + cp.sum(L[i]==k) # int, cluster_size
        reduced_sum = None
        for i in range(NUM_GPUS):
            with cp.cuda.Device(i):
                if reduced_sum is None:
                    reduced_sum = cp.sum(X[i][L[i]==k,:],axis=0) # new_center (dim, )
                else:
                    reduced_sum = cp.asarray(reduced_sum) + cp.sum(X[i][L[i]==k,:],axis=0) # new_center (dim, )
        with m.device:
            ck = 1/m * reduced_sum
        for i in range(NUM_GPUS):
            with cp.cuda.Device(i):
                C[i][k,:]=cp.asarray(ck)

toc = time()       
L0h= cp.asnumpy(cp.concatenate([cp.asarray(x) for x in L0]))
Lh = cp.asnumpy(cp.concatenate([cp.asarray(x) for x in L]))
sc0 =homogeneity_score(L0h, labels_ex)
sc =homogeneity_score(Lh, labels_ex)
print(f'Initial homogeneity score = {sc0:.2}')
print(f'homogeneity score = {sc:.2} (1 is best)')
print(f'Kmeans took {toc-tic:.3} seconds')