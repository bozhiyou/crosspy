import cupy as cp
import crosspy as xp
import numpy as np

with cp.cuda.Device(0):
    a1 = cp.arange(5)
    a11 = cp.zeros(5, dtype=cp.int64)

with cp.cuda.Device(1):
    a2 = cp.arange(3)
    a22 = cp.arange(3, dtype=cp.int64)

left_array = xp.array([a1, a2], dim=0)
right_array = xp.array([a11, a22], dim=0)

index_set = np.arange(len(left_array))[0:4]
np.random.shuffle(index_set)


print("array: ", left_array)
print("array2: ", right_array)

right_array[0:4] = left_array[index_set]