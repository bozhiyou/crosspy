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
from crosspy import numpy as np
from crosspy import cupy as cp

if __name__ == '__main__':
    x = xp.array([np.array([2,4,6,8]), np.array([1,3,5,7])], axis=0)
    # x[0:2]
    print(x)

    print(x[-1])
    print(x.shape)

    # boundaries = (4, 8)
    # def part(i):
    #     return slice(0 if i == 0 else boundaries[i-1], boundaries[i])
    # num_gpus = 2
    # def quick_sort(a):
    #     pivot = a[-1]
    #     for di in range(num_gpus):
    #         with cp.cuda.Device(di):
                

    # raise








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