"""
rng = da.random.default_rng()
x = rng.random((1000, 1000), chunks=(10, 10))
x[x < 0.95] = 0

s = x.map_blocks(sparse.COO)

_ = s.sum(axis=0)[:100].compute()
print(_.todense())
"""

import numpy as np
from numpy import arange, logical_and, where, zeros, unique, searchsorted
from numpy.random import randn
from numpy.linalg import norm
import scipy as sp
from scipy.io import mmread
from scipy.sparse import coo_matrix
import os

print('1. Creating a sparse symmetric SPD matrix')
def init_matrix():
    url = 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos5.mtx.gz'
    clean = False
    file ='m.mtx.gz'
    if os.path.isfile(file) is not True:
        import requests
        rsp = requests.get(url)
        with open(file,"wb") as f: f.write(rsp.content)

    A=mmread(file); A=A.T
    if clean: os.remove(file)
    return A
A = init_matrix()

# number of GPUs
p = 2

m = A.shape[0]
bs = A.shape[0]//p   # block size: number of rows per GPU
print(f'\tMatrix size {m}-by-{m} with nnz={A.data.shape[0]} non zero values')


print(f'2. Partition matrix into p={p} row blocks')
import cupy as cp
import cupyx.scipy.sparse
import dask.array as da
e = 0
Ap = []  # A distributed in 4 GPUs
for k in range(p):   # this would be p independent parallel tasks in Parla
    s = e
    e = s+bs
    r = where(logical_and( A.row>=s, A.row<e))
    # the -s in the row argument below is need needed because
    # each block matrix should start from zero
    bA = coo_matrix((A.data[r], (A.row[r]-s,A.col[r])), shape=(bs,m))
    with cp.cuda.Device(k):
        # Dask requires the blocks to be subscriptable which sparse matrix does not support
        gA = cupyx.scipy.sparse.coo_matrix(bA).toarray()
    Ap.append(gA)

dA = da.concatenate(Ap, axis=0)
# dA = da.block([[x] for x in Ap]).compute()

print(f'3. Distributed matvec y=Ax where x is replicated -- this doesn\'t scale for large p')    
x = randn(m)
yex = A@x

# super simple matvec in wich x is replicated in each GPU
yp = []
for k in range(p):
    with cp.cuda.Device(k):
        yp.append(cp.zeros(bs))
y = da.concatenate(yp, axis=0)
e = 0
for k in range(p):  # p parallel tasks in parla, replicate x,  exclusive block write for y
    s = e
    e = s+bs
    with cp.cuda.Device(k):
        x_ = cp.asarray(x)
        y[s:e] = dA[s:e]@x_    # this would be cupy sparse matvec per GPU
y = y.compute()  # becomes an cp.ndarray
err = norm(yex-y.get())/norm(yex)
print(f'\tDistributed y=Ax error: {err:.3e}')

print(f'4. Distributed matvec y=Ax where x is _not_ replicated -- this _does_ scale for large p')
yp = []
for k in range(p):
    with cp.cuda.Device(k):
        yp.append(cp.zeros(bs))
y = da.concatenate(yp, axis=0)
e = 0
for k in range(p): # p parallel tasks in parla,  partial non-blocked reads from x (allgather), blocked exclusive reads for y
    s = e
    e = s+bs
    with cp.cuda.Device(k):
        ci = cp.unique(Ap[k].col)             # unique column indices required in GPU k
        g2l = cp.searchsorted(ci, Ap[k].col)  # coompute global index to local index mapping
        z = cp.asarray(x[ci.get()])                          # read part of x required in the local matvec;
                                        # nonsliced operation. can be done using the alltoall crosspy support on top ofparray
                                        # read only operation
        # perform matvec
        nnz = Ap[k].data.shape[0]     # number of nonzeros
        # loop over nonzero values
        for l in range(nnz):    # this loop  does teh matvec; use pykokkos for performance
            i = Ap[k].row[l]+s   # output row index
            j = g2l[l]           # find where x[col[l]] is stored in z using the g2l mapping
            Aij = Ap[k].data[l]  # get array value
            y[i] += Aij * z[j]   # perform matvec

# check error
y = np.concatenate([b.get() for b in yp], axis=0)
err = norm(yex-y)/norm(yex)
print(f'\t Distributed y=Ax error: {err:.3e}')


# comments:
# A is row-block partitioned, only reads, embarrasingly paraelle
# both y and x need to be partitioned;
# y is partitioned in contiguous blocks using either parray sliced reades or crosspy dirrectly
#   embarrasingly parallel  
# x is the challenge: it has nonsliced overlapping reads and needs to be optimized.