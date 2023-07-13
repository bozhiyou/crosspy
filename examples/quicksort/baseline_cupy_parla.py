# from parla import Parla, spawn, TaskSpace
import cupy as cp
import numpy as np
import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.cython.device_manager import gpu
from parla.common.globals import get_current_context
from parla.common.array import copy


#Note: this experimental allocator breaks memcpy async
#cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

np.random.seed(10)
cp.random.seed(10)

# COMMENT(wlr): Our distributed array here will be a list of cupy arrays. Everything will be managed manually


def partition_kernel(A, B, comp, pivot):
    # TODO(wlr): Fuse this kernel
    bufferA = A
    bufferB = B
    comp[:] = bufferA[:] < pivot
    mid = int(comp.sum())
    bufferB[:mid] = bufferA[comp]
    bufferB[mid:] = bufferA[~comp]
    return mid


def partition(A, B, pivot):
    """Partition A against pivot and output to B, return mid index."""
    context = get_current_context()
    n_partitions = len(A)
    mid = np.zeros(n_partitions, dtype=np.uint32)

    for i, (array_in, array_out) in enumerate(zip(A, B)):
        with context.devices[i]:
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[i] = partition_kernel(array_in, array_out, comp, pivot)
    return mid


def balance_partition(A, left_counts):
    sizes = np.zeros(len(A), dtype=np.uint32)
    for i, array in enumerate(A):
        sizes[i] = len(array)

    remaining_left = np.copy(left_counts)
    remaining_right = np.copy(sizes) - left_counts
    free = np.copy(sizes)

    source_start_left = np.zeros((len(A), len(A)), dtype=np.uint32)
    target_start_left = np.zeros((len(A), len(A)), dtype=np.uint32)
    sz_left = np.zeros((len(A), len(A)), dtype=np.uint32)

    source_start_right = np.zeros((len(A), len(A)), dtype=np.uint32)
    target_start_right = np.zeros((len(A), len(A)), dtype=np.uint32)
    sz_right = np.zeros((len(A), len(A)), dtype=np.uint32)

    # Pack all left data to the left first
    target_idx = 0
    local_target_start = 0

    for source_idx in range(len(A)):
        local_source_start = 0
        message_size = remaining_left[source_idx]
        while message_size > 0:
            if free[target_idx] == 0:
                target_idx += 1
                local_target_start = 0
                continue
            used = min(free[target_idx], message_size)
            free[target_idx] -= used
            remaining_left[source_idx] -= used

            sz_left[source_idx, target_idx] = used
            source_start_left[source_idx, target_idx] = local_source_start
            target_start_left[source_idx, target_idx] = local_target_start
            local_source_start += used
            local_target_start += used

            message_size = remaining_left[source_idx]

    # Pack all right data to the right
    for source_idx in range(len(A)):
        local_source_start = left_counts[source_idx]
        message_size = remaining_right[source_idx]
        while message_size > 0:
            if free[target_idx] == 0:
                target_idx += 1
                local_target_start = 0
                continue
            used = min(free[target_idx], message_size)
            free[target_idx] -= used
            remaining_right[source_idx] -= used

            sz_right[source_idx, target_idx] = used
            source_start_right[source_idx, target_idx] = local_source_start
            target_start_right[source_idx, target_idx] = local_target_start
            local_source_start += used
            local_target_start += used

            message_size = remaining_right[source_idx]

    return (
        source_start_left,
        target_start_left,
        sz_left
    ), (
        source_start_right,
        target_start_right,
        sz_right
    )


def scatter(A, B, left_info, right_info):
    context = get_current_context()

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.devices[target_idx]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                copy(
                    target[target_start:target_end],
                    source[source_start:source_end]
                    )


    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.devices[target_idx]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                copy(
                    target[target_start:target_end],
                    source[source_start:source_end]
                    )




def quicksort(A, workspace, tid, T):
    device_list = tuple([gpu(arr.device.id) for arr in A])

    @spawn(T[tid], placement=[device_list], vcus=1)
    def quicksort_task():
        nonlocal A
        nonlocal workspace
        nonlocal T
        # print("TASK", T[tid], get_current_context(), flush=True)

        n_partitions = len(A)

        if n_partitions <= 1:
            # Base case.
            if n_partitions == 1:
                # The data is on a single gpu.
                A[0].sort()
            return

        # Form local prefix sum
        sizes = np.zeros(len(A) + 1, dtype=np.uint32)
        for i in range(len(A)):
            sizes[i + 1] = len(A[i])
        size_prefix = np.cumsum(sizes)
        print("Array", size_prefix[-1], n_partitions, sizes)

        # Choose random pivot
        pivot_block = np.random.randint(0, n_partitions)
        pivot_idx = np.random.randint(0, len(A[pivot_block]))
        pivot = (int)(A[pivot_block][pivot_idx])
        # print("Pivot: ", pivot)

        # Local partition and repacking (no communication)
        local_mids = partition(A, workspace, pivot)
        global_left_count = np.sum(local_mids)

        # compute communication pattern
        left_info, right_info = balance_partition(A, local_mids)

        # Send local left/right to global left/right (workspace->A communication)
        scatter(A, workspace, left_info, right_info)

        global_mid_block_idx = np.searchsorted(size_prefix, global_left_count, side="right") - 1
        global_mid_local_offset = global_left_count - size_prefix[global_mid_block_idx]

        left_partition = []
        left_partition += [A[i] for i in range(global_mid_block_idx)]
        if global_mid_block_idx < len(A) and global_mid_local_offset > 0:
            left_partition += [A[global_mid_block_idx][0:global_mid_local_offset]]

        workspace_left = []
        workspace_left += [workspace[i] for i in range(global_mid_block_idx)]
        if global_mid_block_idx < len(A) and global_mid_local_offset > 0:
            workspace_left += [workspace[global_mid_block_idx][0:global_mid_local_offset]]

        right_partition = []
        if global_mid_block_idx < len(A) and global_mid_local_offset < sizes[global_mid_block_idx+1]:
            right_partition += [A[global_mid_block_idx][global_mid_local_offset:sizes[global_mid_block_idx+1]]]
        right_partition += [A[i] for i in range(global_mid_block_idx+1, len(A))]

        workspace_right = []
        if global_mid_block_idx < len(A) and global_mid_local_offset < sizes[global_mid_block_idx+1]:
            workspace_right += [workspace[global_mid_block_idx]
                                [global_mid_local_offset:sizes[global_mid_block_idx+1]]]
        workspace_right += [workspace[i] for i in range(global_mid_block_idx+1, len(A))]

        quicksort(left_partition, workspace, 2 * tid, T)
        quicksort(right_partition, workspace, 2 * tid + 1, T)


def main(args):
    # This is also treated as the maximum number of points that can be on each device (very strict constraint).
    # This has a large performance impact due to recursion on the boundaries between partitions.
    # TODO Separate this into two variables?

    global_array = cp.random.randint(0, 100000000000, size=args.m * args.num_gpus).astype(cp.int32)
    # global_array = np.arange(args.m * args.num_gpus, dtype=np.int32)
    # np.random.shuffle(global_array)

    cupy_list_A = []
    cupy_list_B = []
    for i in range(args.num_gpus):
        with cp.cuda.Device(i) as dev:
            random_array = cp.asarray(global_array[args.m * i:args.m * (i + 1)])
            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.empty(args.m, dtype=cp.int32))

    for i in range(args.num_gpus):
        with cp.cuda.Device(i) as dev:
            dev.synchronize()

    A = cupy_list_A
    workspace = cupy_list_B


    with Parla():
        T = TaskSpace("T")
        t_start = time.perf_counter()
        quicksort(A, workspace, 1, T)
        T.wait()
        t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    # print("Sorted")
    # for array in A:
    #     print(array)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=10000, help="Size of array per GPU.")
    args = parser.parse_args()
    main(args)

"""
    start = int(start)
    end = int(end)

    # # How to select the active block from the global array
    start_block_idx = np.searchsorted(global_prefix, start, side="right") - 1
    end_block_idx = np.searchsorted(global_prefix, end, side="right") - 1

    # Split within a block at the endpoints (to form slices)
    start_local_offset = start - global_prefix[start_block_idx]
    end_local_offset = end - global_prefix[end_block_idx]

    start_local_offset = (int)(start_local_offset)
    end_local_offset = (int)(end_local_offset)

    A = []
    workspace = []

    #Reform a global array out of sliced components (NOTE: THESE SEMANTICS BREAK PARRAY. Concurrent slices cannot be written)
    if start_block_idx == end_block_idx and start_local_offset < end_local_offset:
        A.append(global_A[start_block_idx][start_local_offset:end_local_offset])
        workspace.append(global_workspace[start_block_idx])
    else:
        if (start_block_idx < end_block_idx) and (
            start_local_offset < len(global_A[start_block_idx])
        ):
            A.append(global_A[start_block_idx][start_local_offset:])
            workspace.append(global_workspace[start_block_idx][start_local_offset:])

        for i in range(start_block_idx + 1, end_block_idx):
            A.append(global_A[i])
            workspace.append(global_workspace[i])

        if (end_block_idx < len(global_A)) and end_local_offset > 0:
            A.append(global_A[end_block_idx][:end_local_offset])
            workspace.append(global_workspace[end_block_idx][:end_local_offset])
"""