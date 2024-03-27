import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.sparse import diags

def lap3d(n):
    d0 = 2*np.ones(n);
    d1 = -1*np.ones(n-1);
    D = diags([d1,d0,d1], [-1,0,1])
    I = sp.identity(n)

    A1 = sp.kron(D,I)
    A2 = sp.kron(I,D)
    A = sp.kron(A1,I)+sp.kron(I,A1)+sp.kron(I,A2)
    A.eliminate_zeros()
    return A