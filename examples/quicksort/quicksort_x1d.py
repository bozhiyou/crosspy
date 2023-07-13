import crosspy as xp
from crosspy import cupy as cp  # CrossPy handles import errors

from timeit import timeit
import time

def quicksort(array):
    if len(array) <= 1:
        return
    
    # start = time.perf_counter()
    pivot = int(array[len(array) - 1])  # without type conversion it's a view, not copy
    left_mask = (array < pivot)
    left = array[left_mask]
    right_mask = ~left_mask
    right_mask[len(array) - 1] = False
    right = array[right_mask]
    # end = time.perf_counter()
    # print("body", end-start)
    # assert len(array) == len(left) + 1 + len(right)

    array[len(left)] = pivot
    if len(left):
        quicksort(left)
        xp.alltoall(array, slice(0, len(left)), left, None)
        # out[cp.arange(len(left))] = left
    if len(right):
        quicksort(right)
        xp.alltoall(array, slice(len(left) + 1, len(array)), right, None)
        # out[cp.arange(len(left) + 1, len(array))] = right


def main(args):
    # np.random.seed(10)
    # cp.random.seed(10)

    cupy_list_in = []
    cupy_list_out = []
    for i in range(args.n):
        with cp.cuda.Device(i):
            random_array = cp.random.randint(0, 100, size=args.m).astype(cp.int32)
            random_array = cp.asarray([59, 17, 76, 19,  6]).astype(cp.int32) if i == 0 else cp.asarray([69, 31, 89, 63, 89]).astype(cp.int32)
            cupy_list_in.append(random_array)
            cupy_list_out.append(cp.zeros_like(random_array))
    print(timeit(lambda:
        cp.sort(cupy_list_in[0])
    , number=3)/3)

    from crosspy.core.x1darray import X1D as xparray
    x_in = xparray(cupy_list_in, axis=0)
    x_out = xparray(cupy_list_out, axis=0)
    # x_in = cupy_list_in[0]
    # x_out = cupy_list_out[0]

    print(timeit(lambda:
        quicksort(x_in, out=x_out)
    , number=3)/3)

    print("origin:", x_in)
    print("sorted:", x_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=10000, help="Size of array per GPU.")
    parser.add_argument("-n", type=int, default=2, help="Number of GPUs.")
    args = parser.parse_args()
    main(args)