import numpy as np
import cupy as cp
import crosspy as xp

def test_ints_get():
    with cp.cuda.Device(1):
        a = xp.array([np.array([0, 2, 5]), cp.array([4, 1])], axis=0)
    int_slice = a[[1, 4, 3]]
    int_slice = a[np.arange(1, 5)]
    print(int_slice)

def test_slice_get():
    subarrays = []
    with cp.cuda.Device(0):
        subarrays.append(cp.arange(4))
    with cp.cuda.Device(1):
        subarrays.append(cp.arange(3)*10)
    with cp.cuda.Device(1):
        subarrays.append(cp.arange(2)*100)
    with cp.cuda.Device(0):
        subarrays.append(cp.arange(2)*1000)

    a = xp.array(subarrays, axis=0)
    print(a[0:0])
    print(a[:])
    for i in range(len(a)):
        print(i, a[i:])
    print(a[len(a):])
    for i in range(len(a)):
        print(i, a[:i + 1])

def test_slice_getset():
    with cp.cuda.Device(0):
        a1 = cp.arange(5)
        a11 = cp.zeros(5, dtype=cp.int64)

    with cp.cuda.Device(1):
        a2 = cp.arange(3)
        a22 = cp.arange(3, dtype=cp.int64)

    left_array = xp.array([a1, a2], axis=0)
    right_array = xp.array([a11, a22], axis=0)

    num_ind = 4
    index_set = np.arange(num_ind)
    np.random.shuffle(index_set)

    print("array: ", left_array)
    print("array2: ", right_array)
    buf = left_array[index_set]
    right_array[:num_ind] = buf
    print("array: ", left_array)
    print("array2: ", right_array)

def test_slice_get_int_get():
    # Initilize a CrossPy Array
    cupy_list_A = []
    for _ in range(2):
        with cp.cuda.Device(0):
            random_array = cp.random.randint(0, 100, size=3).astype(cp.int32)
            cupy_list_A.append(random_array)

    xA = xp.array(cupy_list_A, axis=0)
    print(xA[0:len(xA)])
    slicedA = xA[slice(0, 1)]
    assert slicedA[0] == cupy_list_A[0][0]
    print(type(slicedA[0]))
    print(type(slicedA[0].to(-1)))
    int_ = xA[0]
    assert int_ == cupy_list_A[0][0]


if __name__ == "__main__":
    test_ints_get()
    # test_slice_get()
    # test_slice_getset()
    # test_slice_get_int_get()