import numpy as np
import cupy as cp
import crosspy as xp
from crosspy import cpu, gpu

from crosspy.utils import recipe


def array_list_equal(la, lb):
    for a, b in zip(la, lb):
        if not (a == b).all():
            return False
    return True


def test_random():
    a = xp.random.rand()
    assert isinstance(a, float)

    a = xp.random.rand(device=gpu(1))
    assert isinstance(a, cp.ndarray)
    assert a.device.id == 1
    assert a.shape == ()

    a = xp.random.rand(1)
    assert isinstance(a, np.ndarray)
    assert a.shape == (1,)

    a = xp.random.rand(1, device=gpu(1))
    assert isinstance(a, cp.ndarray)
    assert a.shape == (1,)
    assert a.device.id == 1

    a = xp.random.rand([2, 1], device=gpu(1))
    assert a.shape == (3,)

    a = xp.random.rand([2, 1], device=[gpu(1), gpu(0)])
    assert a.shape == (3,)

    # corner cases
    a = xp.random.rand(0)
    assert isinstance(a, np.ndarray)
    assert a.shape == (0,)

    a = xp.random.rand(0, device=gpu(1))
    assert isinstance(a, cp.ndarray)
    assert a.device.id == 1
    assert a.shape == (0,)

    # bad cases
    try: a = xp.random.rand([], [])
    except TypeError: pass

    try: a = xp.random.rand([2, 1], device=[gpu(1)])
    except ValueError: pass


def test_like():
    proto = xp.random.rand([2, 3, 4], device=[gpu(1), cpu(0), gpu(0)])

    x = xp.empty_like(proto)
    x = xp.zeros_like(proto)
    x = xp.ones_like(proto)

def test_create():
    x_cpu = np.array([1, 2, 3, 4])
    x_gpu = cp.array([5, 6, 7, 8])

    ########
    # regular
    ########
    x1 = xp.array(x_cpu)
    assert x1.shape == x_cpu.shape
    assert array_list_equal(list(x1.device_view()), [x_cpu])
    x1 = xp.array([x_cpu])
    assert(x1.shape == (1,) + x_cpu.shape)
    assert array_list_equal(list(x1.device_view()), [x_cpu])
    x1 = xp.array([x_cpu, x_cpu])
    assert(x1.shape == (2,) + x_cpu.shape)
    assert array_list_equal(list(x1.device_view()), [x_cpu, x_cpu])

    x1 = xp.array(x_gpu)
    assert(x1.shape == x_gpu.shape)
    # assert array_list_equal(list(x1.device_view()), [x_gpu])
    x1 = xp.array([x_gpu])
    assert(x1.shape == (1,) + x_gpu.shape)
    # assert array_list_equal(list(x1.device_view()), [x_gpu])
    x1 = xp.array([x_gpu, x_gpu])
    assert(x1.shape == (2,) + x_gpu.shape)
    # assert array_list_equal(list(x1.device_view()), [x_gpu, x_gpu])

    x1 = xp.array([x_cpu, x_gpu])
    assert(x1.shape == (2,) + x_cpu.shape)
    # assert array_list_equal(list(x1.device_view()), [x_gpu, x_gpu])

    ########
    # dim/axis
    ########
    # x1 = xp.array(x_cpu, axis=0)
    # assert x1.shape == ???
    # assert array_list_equal(list(x1.device_view()), [x_cpu])
    x1 = xp.array([x_cpu], axis=0)
    assert x1.shape == x_cpu.shape
    # assert array_list_equal(list(x1.device_view()), [x_cpu])
    x1 = xp.array([x_cpu, x_cpu], axis=0)
    assert x1.shape == (x_cpu.shape[0] * 2,) + x_cpu.shape[1:]
    assert array_list_equal(list(x1.device_view()), [x_cpu, x_cpu])

    # x1 = xp.array(x_gpu)
    # assert(x1.shape == x_gpu.shape)
    # assert array_list_equal(list(x1.device_view()), [x_gpu])
    x1 = xp.array([x_gpu], axis=0)
    assert x1.shape == x_cpu.shape
    # assert array_list_equal(list(x1.device_view()), [x_gpu])
    x1 = xp.array([x_gpu, x_gpu], axis=0)
    assert x1.shape == (x_gpu.shape[0] * 2,) + x_gpu.shape[1:]
    # assert array_list_equal(list(x1.device_view()), [x_gpu, x_gpu])

    x1 = xp.array([x_cpu, x_gpu], axis=0)
    assert x1.shape == (x_cpu.shape[0] * 2,) + x_cpu.shape[1:]
    # assert array_list_equal(list(x1.device_view()), [x_gpu, x_gpu])

    ########
    # partition
    ########
    x1 = xp.array(x_cpu, distribution=[gpu(0), gpu(1)])
    print(x1)

    return
    x_cpu = np.array([[1, 2, 3]])
    x_gpu = cp.array([[4, 5]])
    print("cupy array", x_gpu, "on device", x_gpu.device)

    x_cross = xp.array([x_cpu, x_gpu])
    print("crosspy array", x_cross)
    print("shape:", x_cross.shape)

    x = xp.array([x_gpu, x_gpu], axis=1)
    print(x)
    print(x.shape)
    # x = xp.array([x_cpu, x_cpu.T])

    # x = xp.array(np.array)
    # x = xp.array(cp.array)
    # x = xp.array([...])
    # partition obj
    # generate list

def test_recipe(n=6, m=4, ngpus=2):
    a = recipe(lambda i: cp.random.rand(n//ngpus),
               lambda i: cp.cuda.Device(i),
               i=range(ngpus))

    indices = np.random.choice(n, m)

    # b = a[indices]
    b = recipe(lambda i: cp.asarray(indices[i*(m//ngpus):(i+1)*(m//ngpus)]),
               lambda i: cp.cuda.Device(i),
               i=range(ngpus))
    
    
if __name__ == '__main__':
    test_random()
    test_like()
    print(xp.empty([2,4], device=[gpu(0), gpu(1)]))
    # test_create()