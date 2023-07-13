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

# generate data
n = 10000000;
if VRB:print(f'number of points = {n}')

nc = 10;
if VRB:print(f'number of clusters = {nc}')

print('Generating data')
Xh, labels_ex = make_blobs(n_samples=n, centers=nc,
                     n_features=dim)

# generate initial guess for clusters
cpos  = np.random.randint(0,n,nc)
Ch = Xh[cpos,:]
X = cp.asarray(Xh) 
C = cp.asarray(Ch)

print('allocating memory')
D = cp.zeros((n,nc))
W = cp.zeros((n,dim))
L = cp.zeros(n)
L0= cp.zeros(n)

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
        ck = 1/m *cp.sum(X[L==k,:],axis=0)
        C[k,:]=ck

toc = time()       
L0h= cp.asnumpy(L0)
Lh = cp.asnumpy(L)
sc0 =homogeneity_score(L0h, labels_ex)
sc =homogeneity_score(Lh, labels_ex)
print(f'Initial homogeneity score = {sc0:.2}')
print(f'homogeneity score = {sc:.2} (1 is best)')
print(f'Kmeans took {toc-tic:.3} seconds')

    