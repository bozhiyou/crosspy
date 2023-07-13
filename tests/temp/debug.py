
import numpy as np
import cupy as cp
import crosspy as xp

np.random.seed(10)
cp.random.seed(10)

import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu


def create_array(per_gpu_size, num_gpus, unique=False, dtype=np.int32):
    if unique:
        global_array = np.arange(per_gpu_size * num_gpus, dtype=dtype)
        np.random.shuffle(global_array)
    else:
        global_array = cp.random.randint(0, 1000000, per_gpu_size * num_gpus).astype(dtype, copy=False)

    # Init a distributed crosspy array
    cupy_list_A = []
    cupy_list_B = []
    for i in range(num_gpus):
        with cp.cuda.Device(i) as dev:
            random_array = cp.asarray(global_array[per_gpu_size * i:per_gpu_size * (i + 1)])
            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.empty(per_gpu_size, dtype=dtype))

    for i in range(num_gpus):
        with cp.cuda.Device(i) as dev:
            dev.synchronize()

    return global_array, cupy_list_A, cupy_list_B

def reorder(array, srcs, T):
    src = next(srcs)
    tid = id(src)
    @spawn(T[tid] if T is not None else None, placement=[gpu(array.device.id)])
    def _():
        src.tobuffer(array, stream=dict(non_blocking=True))

def partition(array, pivot, l, r, T):
    tid = id(array)
    @spawn(T[tid], placement=gpu(array.device.id))
    def _():
        left_mask = (array < pivot)
        right_mask = ~left_mask
        left = array[left_mask]
        right = array[right_mask]
        if len(left):
            l[tid] = left
        if len(right):
            r[tid] = right


def quicksort(array: xp.CrossPyArray, T, Tid=1):
    if len(array) <= 10000:
        return
    placement = tuple(array.block_view(lambda arr: gpu(arr.device.id)))
    # print(Tid, array, placement)

    @spawn(T[Tid], placement=[placement])
    async def _():
        pivot = int(array[len(array) - 1])  # without type conversion it's a view, not copy
        
        lefts, rights = {}, {}
        pT = AtomicTaskSpace("pT")
        array.block_view(partition, pivot=pivot, l=lefts, r=rights, T=pT)
        await pT
        left = xp.array(list(a for a in lefts.values() if len(a)), axis=0)
        right = xp.array(list(a for a in rights.values() if len(a)), axis=0)

        array[len(left)] = pivot
        if len(left):
            aleft = array[:len(left)]
            liter = iter(xp.split(left, aleft.boundaries))
            lT = AtomicTaskSpace("lT")
            aleft.block_view(reorder, liter, T=lT)
            await lT
            quicksort(aleft, T, 2 * Tid)
        if len(right) > 1:
            # array[len(left) + 1:] = right[:-1]
            aright = array[len(left) + 1:]
            riter = iter(xp.split(right[:-1], aright.boundaries))
            rT = AtomicTaskSpace("rT")
            aright.block_view(reorder, riter, T=rT)
            await rT
            quicksort(aright, T, 2 * Tid + 1)
    return

    @spawn(T[Tid], placement=[placement])
    async def quicksort_task():
        if len(array) < 2:
            return

        pivot = int(array[len(array) - 1])  # without type conversion it's a view, not copy
        left_mask = (array < pivot)
        left = array[left_mask]
        right_mask = ~left_mask
        right_mask[len(array) - 1] = False
        right = array[right_mask]
        assert len(array) == len(left) + 1 + len(right)

        array[len(left)] = pivot
        if len(left):
            array[:len(left)] = left
            await quicksort(array[:len(left)], T, 2 * Tid)
        if len(right):
            array[len(left) + 1:] = right
            await quicksort(array[len(left) + 1:], T, 2 * Tid + 1)

    return T[Tid]

def main(args):
    global_array, cupy_list, _ = create_array(args.m, args.num_gpus)

    # Initilize a CrossPy Array
    xA = xp.array(cupy_list, axis=0)

    # print("Original Array: ", xA, flush=True)

    with Parla():
        # t_start = time.perf_counter()
        # with cp.cuda.Device(0):
        #     cp.sort(cupy_list[0])
        # with cp.cuda.Device(1):
        #     cp.sort(cupy_list[1])
        # t_end = time.perf_counter()
        # print("Time: ", t_end - t_start)

        time.sleep(1)

        # T = AtomicTaskSpace("T")
        # t_start = time.perf_counter()
        # @spawn(T[0], placement=gpu(0))
        # def _0():
        #     cp.sort(cupy_list[0])
        # @spawn(T[1], placement=gpu(1))
        # def _1():
        #     cp.sort(cupy_list[1])
        # T.wait()
        # t_end = time.perf_counter()
        # print("Time: ", t_end - t_start)
        # return

        T = AtomicTaskSpace("T")
        t_start = time.perf_counter()
        quicksort(xA, T)
        T.wait()
        t_end = time.perf_counter()

    # print("Sorted:")
    # print(xA)
    print("Time: ", t_end - t_start)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
    parser.add_argument("-num_gpus", type=int, default=1)
    parser.add_argument("-m", type=int, default=1000000)
    args = parser.parse_args()
    main(args)
