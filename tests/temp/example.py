import numpy as np, cupy as cp
dummy_large_number = 10

# conceptually
a_origin = np.random.rand(dummy_large_number)
b_origin = np.random.rand(dummy_large_number)

# a_first_half = a_origin[:dummy_large_number // 2]
# b_first_half = b_origin[:dummy_large_number // 2]

# with cp.cuda.Device(0):
#     a_second_half = cp.asarray(a_origin[dummy_large_number // 2:])
#     b_second_half = cp.asarray(b_origin[dummy_large_number // 2:])

import crosspy as xp
from crosspy import cpu, gpu

ax = xp.array(a_origin, distribution=[cpu(0), gpu(0)], axis=0)
bx = xp.array(b_origin, distribution=[cpu(0), gpu(0)], axis=0)
cx = ax + bx
print(cx)