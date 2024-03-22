import numpy as np
import cupy as cp
import crosspy as xp

import time

import ctypes


GB64 = 2**(30-3)

a = np.random.rand(1*GB64)
b = a[:len(a)//2]
c = b[:len(b)//2]
print(a.nbytes)
print(b.nbytes)
print(c.nbytes)

# with cp.cuda.Device(0):
#     aa = cp.asarray(a)
#     ae = cp.empty_like(a)
#     ae.data.copy_from_host(a.ctypes.data, a.nbytes)

# with cp.cuda.Device(0):
#     bb = cp.asarray(b)

# with cp.cuda.Device(0):
#     cc = cp.asarray(c)