# from parla import Parla, spawn, TaskSpace
import cupy as cp
import numpy as np
import time

np.random.seed(10)
cp.random.seed(10)

def partition_kernel(A, B, comp, pivot):
    # TODO(wlr): Fuse this kernel
    bufferA = A
    bufferB = B
    comp[:] = bufferA[:] < pivot
    mid = int(comp.sum())
    bufferB[:mid] = bufferA[comp]
    bufferB[mid:] = bufferA[~comp]
    # print("Reordered Buffer:", bufferA, comp, bufferB, (bufferB[:] < pivot))
    return mid


def partition(A, B, pivot):
    """Partition A against pivot and output to B, return mid index."""
    n_partitions = len(A)
    mid = np.zeros(n_partitions, dtype=np.uint32)

    for i, (array_in, array_out) in enumerate(zip(A, B)):
        with cp.cuda.Device(i):
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[i] = partition_kernel(array_in, array_out, comp, pivot)
    return mid


def balance_partition(A, left_counts):
    """Redistribution scheme.
    Returns three |A| x |A| matrix for local source start, local target start, and msg size, for left and right respectively.
    """
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

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with cp.cuda.Device(0):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]
                # print("TARGET: ", target, type(target))
                # print("SOURCE: ", source, type(source))
                target[target_start:target_end] = source[source_start:source_end]

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with cp.cuda.Device(0):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                # A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]
                target = A[target_idx]
                source = B[source_idx]
                target[target_start:target_end] = source[source_start:source_end]


def quicksort(A, workspace):

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

    quicksort(left_partition, workspace_left)
    quicksort(right_partition, workspace_right)


def main(args):
    global_array = cp.random.randint(0, 100000000000, size=args.m * args.num_gpus).astype(cp.int32)

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

    t_start = time.perf_counter()
    with cp.cuda.Device(0) as d:
        quicksort(A, workspace)
        d.synchronize()
    t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    with cp.cuda.Device(0) as d:
        d.synchronize()
        t_start = time.perf_counter()
        global_array.sort()
        d.synchronize()
        t_end = time.perf_counter()

    print("Single-GPU reference time: ", t_end - t_start)

    # print("Sorted")
    # for array in A:
    #     print(array)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=1000, help="Size of array per GPU.")
    args = parser.parse_args()
    main(args)