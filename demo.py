import crosspy as xp
from crosspy import numpy as np
from crosspy import cupy as cp

if __name__ == '__main__':
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
    y_cpu = x_cross.to(-1)
    print("all to cpu", y_cpu, type(y_cpu))
    y_gpu0 = x_cross[:1, (0, 2, 4)].to(0)
    print("all to", y_gpu0.device, " ", y_gpu0, type(y_gpu0))
    y_gpu1 = x_cross.to(1)
    print("all to", y_gpu1.device, " ", y_gpu1, type(y_gpu1))
