from contextlib import nullcontext
from functools import lru_cache

import numpy
from crosspy import cupy

def _pinned_memory_empty_like(array):
    if cupy and isinstance(array, numpy.ndarray):
        mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
        ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        return ret
    return array

def _pin_memory(array):
    ret = _pinned_memory_empty_like(array)
    ret[...] = array
    return ret

def same_place(x, y):
    return type(x) == type(y) and (
        isinstance(x, numpy.ndarray) or (
        cupy and isinstance(x, cupy.ndarray) and x.device == y.device
    ))

    
def any_to_cuda(array, stream, out=None):
    if out is None:
        out = cupy.empty(array.shape, dtype=array.dtype)
    if isinstance(array, numpy.ndarray):  # Copy CPU to GPU
        out.set(array, stream=stream)
    elif isinstance(array, cupy.ndarray):  # Copy GPU to GPU
        out.data.copy_from_device_async(array.data, array.nbytes, stream=stream)
    else:
        raise NotImplementedError("Transferring %s object to gpu is not supported yet" % type(array))
    return out


def pull(array, context, stream_src=None):
    # Assume device >= 0 means the device is a GPU, device < 0 means the device is a CPU.
    source_device = getattr(array, 'device', None)
    if source_device == context or (not hasattr(array, 'device') and isinstance(context, nullcontext)):
        return array

    # to CPU
    if isinstance(context, nullcontext):
        if source_device:  # GPU to CPU
            with source_device:
                with cupy.cuda.Stream(non_blocking=True) as stream:
                    membuffer = _pinned_memory_empty_like(array)
                    array.get(stream=stream, out=membuffer)
                    stream.synchronize()
            return membuffer
        return array  # CPU to CPU

    # to GPU
    with context:
        membuffer = cupy.empty(array.shape, dtype=array.dtype)
        with cupy.cuda.Stream(non_blocking=True) as stream_dst:
            any_to_cuda(array, stream=stream_src or stream_dst, out=membuffer)
            if stream_src: stream_src.synchronize()
            stream_dst.synchronize()
        return membuffer

pull_to = lambda context: (lambda array: pull(array, context))

def alltoallv(sendbuf, sdispls, recvbuf, debug=False):
    """
    sendbuf [[] [] []]
    sdispls [. . .]

    indices size = recv size

    --- old implementation ---
    for i, send_block in enumerate(sendbuf.block_view()):
        for j, recv_block in enumerate(recvbuf.block_view()):
            sdispls_j, source_block_ids_j = index_for_j(j)
            # gather
            with getattr(send_block, 'device', nullcontext()) as context:
                pull_here = pull_to(context)
                gather_mask = (pull_here(source_block_ids_j) == i)
                gather_indices_local = pull_here(sdispls_j)[gather_mask] - (source_bounds[i-1] if i else 0)
                # assert sum(gather_mask) == len(gather_indices_local)
                send_buf = send_block[gather_indices_local]
            # scatter
            with getattr(recv_block, 'device', nullcontext()) as context:
                pull_here = pull_to(context)
                scatter_mask = pull_here(gather_mask)
                assert not debug or cupy.allclose(recv_block[scatter_mask], pull_here(send_buf))
                recv_block[scatter_mask] = pull_here(send_buf)
    """
    source_bounds = sendbuf.boundaries
    target_bounds = recvbuf.boundaries

    if isinstance(sdispls, numpy.ndarray):
        sdispls = _pin_memory(sdispls)

    recv_cache = []
    for j, recv_block in enumerate(recvbuf.block_view()):
        with getattr(recv_block, 'device', nullcontext()) as recv_device:
            with cupy.cuda.Stream(non_blocking=True) as s_recv:
                j_range = slice((target_bounds[j-1] if j else 0), target_bounds[j])
                sdispls_j = any_to_cuda(sdispls[j_range], stream=s_recv)
                source_block_ids_j = sendbuf._global_index_to_block_id(sdispls_j)
            recv_cache.append((s_recv, sdispls_j, source_block_ids_j))

    streams = []
    for i, send_block in enumerate(sendbuf.block_view()):
        for j, recv_block in enumerate(recvbuf.block_view()):
            with getattr(recv_block, 'device', nullcontext()) as recv_device:
                s_recv, sdispls_j, source_block_ids_j = recv_cache[j]
                with s_recv:
                    mask = (source_block_ids_j == i)
                    gather_indices_local = sdispls_j[mask] - (source_bounds[i-1] if i else 0)

                    if len(gather_indices_local) >= (len(send_block) // 2):
                        # one hop communication: transfer whole block
                        if not same_place(send_block, recv_block):
                            recv_buf = cupy.empty_like(send_block)
                            with getattr(send_block, 'device', nullcontext()) as send_device:
                                with cupy.cuda.Stream(non_blocking=True) as s_src:
                                    any_to_cuda(send_block, stream=s_src, out=recv_buf)
                            s_src.synchronize()
                        else:
                            recv_buf = send_block
                        assert not debug or cupy.allclose(recv_block[mask], recv_buf[gather_indices_local])
                        recv_block[mask] = recv_buf[gather_indices_local]
                    else:
                        # two hop communication: transfer indices then transfer slices
                        with getattr(send_block, 'device', nullcontext()) as send_device:
                            with cupy.cuda.Stream(non_blocking=True) as s_src:  # TODO no stream for host
                                if not same_place(gather_indices_local, send_block):
                                    gather_indices_local_send = cupy.empty_like(gather_indices_local)
                                    with getattr(recv_block, 'device', nullcontext()) as recv_device:
                                        with s_recv:
                                            any_to_cuda(gather_indices_local, stream=s_recv, out=gather_indices_local_send)
                                        s_recv.synchronize()
                                else:
                                    gather_indices_local_send = gather_indices_local
                                # assert sum(gather_mask) == len(gather_indices_local)
                                send_buf = send_block[gather_indices_local_send]

                        with getattr(recv_block, 'device', nullcontext()) as recv_device:
                            with s_recv:
                                # gather_mask, send_buf = msgs[i][j]
                                if not same_place(send_buf, recv_block):
                                    recv_buf = cupy.empty_like(send_buf)
                                    with getattr(send_block, 'device', nullcontext()) as send_device:
                                        with s_src:  # cupy.cuda.Stream(non_blocking=True) as s0:
                                            any_to_cuda(send_buf, stream=s_src, out=recv_buf)
                                    s_src.synchronize()
                                else:
                                    recv_buf = send_buf
                                assert not debug or cupy.allclose(recv_block[mask], recv_buf), recv_buf
                                recv_block[mask] = recv_buf
                        
                streams.append(s_recv)
    
    for s in streams:
        s.synchronize()


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