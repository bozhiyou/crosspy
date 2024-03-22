from lap3d import lap3d
import numpy as np
import cupy as cp
import crosspy as xp
from numpy import arange, logical_and, where, zeros, unique, searchsorted
from numpy.random import randn
from numpy.linalg import norm
import scipy as sp
from scipy.io import mmread
from scipy.sparse import coo_matrix
import os

m = 33
A = lap3d(m)
A=A.tocoo(copy=True)

A = sp.sparse.random(40000, 40000, density=0.01)

# number of GPUs
p = 4

m = A.shape[0]
bs = A.shape[0]//p   # block size: number of rows per GPU
print(f'\tMatrix size {m}-by-{m} with nnz={A.data.shape[0]} non zero values')


print(f'2. Partition matrix into p={p} row blocks')
import cupy as cp
import cupyx.scipy.sparse
import crosspy as xp
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
        gA = cupyx.scipy.sparse.coo_matrix(bA)
    Ap.append(gA)
xA = xp.array(Ap, axis=0)

print(f'3. Distributed matvec y=Ax where x is replicated -- this doesn\'t scale for large p')    
x = randn(m)
yex = A@x
"""
# super simple matvec in wich x is replicated in each GPU
yp = []
for k in range(p):
    with cp.cuda.Device(k):
        yp.append(cp.zeros(bs))
y = xp.array(yp, axis=0)
# TODO p parallel tasks in parla, replicate x,  exclusive block write for y
y = xA@x    # this would be cupy sparse matvec per GPU
y = np.asarray(xA)
err = norm(yex-y)/norm(yex)
print(f'\tDistributed y=Ax error: {err:.3e}')    
"""
print(f'4. Distributed matvec y=Ax where x is _not_ replicated -- this _does_ scale for large p')
x_ = xp.array(x, distribution=[cp.cuda.Device(k) for k in range(p)], axis=0)

from time import perf_counter
t = perf_counter()
for k in range(p):
    with cp.cuda.Device(k):
        ci = cp.unique(Ap[k].col)             # unique column indices required in GPU k
        g2l = cp.searchsorted(ci, Ap[k].col)  # coompute global index to local index mapping
        z = x_[ci]                          # read part of x required in the local matvec;
t = perf_counter() - t
print(t)