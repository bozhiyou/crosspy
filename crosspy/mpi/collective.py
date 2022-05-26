from contextlib import nullcontext
from functools import lru_cache

from crosspy import cupy

def pull(array, context):
    # Assume device >= 0 means the device is a GPU, device < 0 means the device is a CPU.
    source_device = getattr(array, 'device', None)
    if source_device == context:
        return array

    if isinstance(context, nullcontext):
        if source_device:  # GPU to CPU
            with source_device:
                with cupy.cuda.Stream(non_blocking=True) as stream:
                    membuffer = cupy.asnumpy(array, stream=stream)
                    stream.synchronize()
            return membuffer
        return array  # CPU to CPU

    with context:
        with cupy.cuda.Stream(non_blocking=True) as stream:
            membuffer = cupy.empty(array.shape, dtype=array.dtype)
            if source_device:  # Copy GPU to GPU
                membuffer.data.copy_from_device_async(array.data, array.nbytes, stream=stream)
            else:  # Copy CPU to GPU
                membuffer.set(array, stream=stream)
            stream.synchronize()
            return membuffer

pull_to = lambda context: (lambda array: pull(array, context))

def alltoallv(sendbuf, sdispls, recvbuf, debug=False):
    """
    sendbuf [[] [] []]
    sdispls [. . .]
    """
    source_bounds = sendbuf._bounds
    target_bounds = recvbuf.boundaries

    @lru_cache(maxsize=len(recvbuf.block_view()))
    def index_for_j(j):
        with getattr(sdispls, 'device', nullcontext()):
            sdispls_j = sdispls[(target_bounds[j-1] if j else 0):target_bounds[j]] if recvbuf.heteroaxis is not None else sdispls
            source_block_ids = sendbuf._global_index_to_block_id(sdispls_j)
        return sdispls_j, source_block_ids

    for i, send_block in enumerate(sendbuf.block_view()):
        for j, recv_block in enumerate(recvbuf.block_view()):
            # gather
            with getattr(send_block, 'device', nullcontext()) as context:
                pull_here = pull_to(context)
                sdispls_j, source_block_ids = index_for_j(j)
                gather_mask = (pull_here(source_block_ids) == i)
                gather_indices_local = pull_here(sdispls_j)[gather_mask] - (source_bounds[i-1] if i else 0)
                # assert sum(scatter_mask) == len(gather_indices_local)
                buf = send_block[gather_indices_local]
            # scatter
            with getattr(recv_block, 'device', nullcontext()) as context:
                pull_here = pull_to(context)
                scatter_mask = pull_here(gather_mask)
                assert not debug or cupy.allclose(recv_block[scatter_mask], pull_here(buf))
                recv_block[scatter_mask] = pull_here(buf)

def all2ints(src, dest, dest_index, debug=False):
    """
    sendbuf [[] [] []]
    sdispls [. . .]
    """
    target_bounds = dest._bounds
    source_bounds = src._bounds

    @lru_cache(maxsize=len(src._original_data))
    def _cached(i):
        block_local_indices = cupy.asarray(dest_index[(source_bounds[i-1] if i else 0):source_bounds[i]])
        i_target_block_ids = cupy.sum(cupy.expand_dims(block_local_indices, axis=-1) >= cupy.asarray(target_bounds), axis=-1, keepdims=False)
        return block_local_indices, i_target_block_ids

    for i in range(len(src._original_data)):
        for j in range(len(dest._original_data)):
            # gather
            with getattr(src._original_data[i], 'device', nullcontext()):
                i_target_indices, i_target_block_ids = _cached(i)
                gather_mask = (cupy.asarray(i_target_block_ids) == j)
                buf = src._original_data[i][gather_mask]
            with getattr(dest._original_data[j], 'device', nullcontext()):
                scatter_mask = cupy.asarray(gather_mask)
                scatter_indices_local = cupy.asarray(i_target_indices)[scatter_mask] - (target_bounds[j-1] if j else 0)
                # assert sum(scatter_mask) == len(scatter_indices_local)
                assert not debug or cupy.allclose(dest._original_data[j][scatter_indices_local], cupy.asarray(buf))
                dest._original_data[j][scatter_indices_local] = cupy.asarray(buf)
            # scatter

def alltoall(target, target_indices, source, source_indices):
    with getattr(target_indices, 'device', nullcontext()):
        target_block_ids = target_indices
    with getattr(source_indices, 'device', nullcontext()):
        source_block_ids = source_indices

    for i in range(len(sendbuf._original_data)):
        for j in range(len(recvbuf._original_data)):
            # gather
            with getattr(sendbuf._original_data[i], 'device', nullcontext()):
                if target_indices is None:
                    source_indices[(target_bounds[j-1] if j else 0):target_bounds[j]]
                g_source_indices = source_indices[target_block_id == j] 
                j_source_block_id = g_source_indices

                ij_source_mask = (j_source_block_id == i)
                ij_gather_indices_global = g_source_indices[source_mask]
                ij_gather_indices_local = ij_gather_indices_global
                buf = sendbuf._original_data[i][gather_indices_local]
            # scatter
            with getattr(recvbuf._original_data[j], 'device', nullcontext()):
                j_global_indices = target_indices[target_block_id == j][ij_source_mask]
                j_local_indices = j_global_indices
                recvbuf._original_data[j][j_local_indices] = buf


def assignment(left, left_indices, right, right_indices):
    if left_indices is None:
        alltoallv(right, right_indices, left)
    elif right_indices is None:
        all2ints(right, left, left_indices)
    else:
        alltoall(left, left_indices, right, right_indices)