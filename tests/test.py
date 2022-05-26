import logging


def _get_logger(name=None, *, level=logging.WARNING, fmt=logging.BASIC_FORMAT):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


logger = _get_logger(
    __name__,
    # level=logging.DEBUG,
    fmt=
    '%(asctime)s [\033[1;4m%(levelname)s\033[0m %(processName)s:%(threadName)s] %(filename)s:%(lineno)s %(message)s'
)


import crosspy as xp
from crosspy import PartitionScheme
from crosspy import numpy as np
from crosspy import cupy as cp


import cupy as cp
import crosspy as xp


with cp.cuda.Device(0):
    a1 = cp.arange(5)
with cp.cuda.Device(1):
    a2 = cp.arange(3)

left_array = xp.array([a1, a2], dim=0)
print("array: ", left_array)
print("array 0:2 : ", left_array[4])




x_cpu = np.array([1, 2, 3])
x_gpu = cp.array([4, 5])
x_cross = xp.array([x_cpu, x_gpu], dim=0)
print(x_cross)

for i, parray_list in enumerate(x_cross.device_view()):
    print(i, ", ", type(parray_list), ", ", parray_list)


def main(T):

    # Per device size
    m = 5

    global_size = m * 2
    global_array = np.arange(global_size, dtype=np.int32)
    np.random.shuffle(global_array)

    print(global_array)

    partition = xp.PartitionScheme(global_size, default_device=gpu(0))

    #This also gives errors without or without 
    #for i in range(global_size):
    #    partition[i] = gpu(0)

    A = xp.array(global_array, partition=partition)

    print(A)



if __name__ == "__main__":
    T = None
    main(T)



if __name__ == '__main__':
    partition_2 = PartitionScheme(3, default_device=xp.gpu(0))
    partition_2[0:3] = xp.gpu(0)
    a = xp.array(np.array([1, 2, 3]), partition=partition_2)

    # a = xp.array([[np.full((2,2), i*2+j) for j in range(2)] for i in range(2)])
    A = xp.array(range(10), placement=[xp.cpu(0), xp.gpu(0)])
    B = xp.array(range(10), placement=[xp.cpu(0), xp.gpu(0), xp.gpu(1)])
    A[1:5] = B[4:9]

    A = np.arange(64).reshape(8, 8)
    A_cross = xp.array(A, placement=[xp.gpu(0), xp.gpu(1), xp.gpu(2), xp.gpu(3)])
    print(A_cross)
    blocks = 4
    block_size = 2
    for d in range(blocks):
        for j in range(blocks):
            a_block = A_cross[d*block_size:(d+1)*block_size, j*block_size:(j+1)*block_size]
            print(a_block)

    from crosspy import cpu, gpu, PartitionScheme
    # a = xp.array(range(6), placement=[cpu(0), gpu(0), gpu(1)])
    partition = PartitionScheme(6, default_device=cpu(0))
    # partition[(0, 4, 5)] = cpu(0)
    partition[1:3] = gpu(0)
    partition[3] = gpu(1)
    print(partition)
    a = xp.array(range(6), partition=partition)
    print(a)
    print(a.devices)
    print(a.type_map)
    print()

    a[0] = a[2] + a[4]
    print(a)
    print(a.devices)
    print(a.type_map)
    print()

    b = a.to([gpu(0), cpu(0)])
    print(b)
    print(b.types)

    # array API
    x_cpu = np.array([[1, 2, 3]])
    print("numpy array", x_cpu, "on cpu")

    x_gpu = cp.array([[4, 5]])
    print("cupy array", x_gpu, "on device", x_gpu.device)

    x_cross = xp.array([x_cpu, x_gpu])
    print("crosspy array", x_cross)
    print("shape:", x_cross.shape)

    x = xp.array([x_gpu, x_gpu], dim=1)
    print(x)
    print(x.shape)
    # x = xp.array([x_cpu, x_cpu.T])

    # arbitrary slicing
    x = x_cross[0]
    print("x_cross[0]", x)
    print("shape:", x.shape)

    x = x_cross[0, (0, 3)]
    print("x_cross[0, indices]", x)
    print("shape:", x.shape)

    x = x_cross[:, 0:2]
    print("x_cross[0, 0:2]", x)
    print("shape:", x.shape)

    x_cross[0] = np.array([6, 7, 8, 9, 0])
    x_cross[0] = cp.array([6, 7, 8, 9, 0])
    print("assigned new value from numpy/cupy", x_cross)
    print("shape:", x_cross.shape)

    # interoperability with numpy/cupy
    y_cpu = x_cross.to(cpu(0))
    print("all to cpu", y_cpu, type(y_cpu))
    y_gpu0 = x_cross[:1, (0, 2, 4)].to(gpu(0))
    print("all to", y_gpu0.device, " ", y_gpu0, type(y_gpu0))
    y_gpu1 = x_cross.to(gpu(1))
    print("all to", y_gpu1.device, " ", y_gpu1, type(y_gpu1))