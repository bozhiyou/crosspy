import itertools
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import numpy
from crosspy import cupy, _pin_memory, _pinned_memory_empty_like

from crosspy.device import get_device
from crosspy.utils.array import get_array_module

from .cache import same_place, any_to_cuda, pull, pull_to


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

    # Stage 1: indices distribution
    if isinstance(sdispls, numpy.ndarray):
        sdispls = _pin_memory(sdispls)

    recv_cache = []
    for j, recv_block in enumerate(recvbuf.block_view()):
        with getattr(recv_block, 'device', nullcontext()) as recv_device:
            with cupy.cuda.Stream(non_blocking=True) as s_recv:
                j_range = slice((target_bounds[j-1] if j else 0), target_bounds[j])
                sdispls_j = any_to_cuda(sdispls[j_range], stream=s_recv)
            recv_cache.append([s_recv, sdispls_j])

    for j, recv_block in enumerate(recvbuf.block_view()):
        with getattr(recv_block, 'device', nullcontext()) as recv_device:
            s_recv, sdispls_j = recv_cache[j]
            with s_recv:
                source_block_ids_j = sendbuf._global_index_to_block_id(sdispls_j)
                recv_cache[j].append(source_block_ids_j)

    # copy
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


class alltoall:
    def __init__(self, target, target_indices, source, source_indices, debug=False):
        # Stage 1: indices distribution
        if isinstance(target_indices, numpy.ndarray):
            target_indices = _pin_memory(target_indices)
        if isinstance(source_indices, numpy.ndarray):
            source_indices = _pin_memory(source_indices)

        # make replicas
        # TODO: indices may not cover all devices hence no need to replicate for all
        src_stream_pool = defaultdict(list)
        dst_stream_pool = defaultdict(list)

        def _make_handles(xpa, indices, stream_pool):
            """create handles for data on devices"""
            handles = {}
            for did, darray in xpa.device_array.items():
                replicas = []
                with get_device(darray) as device:
                    for array in (indices, *xpa.get_metadata()):
                        with cupy.cuda.Stream(non_blocking=True) as s:
                            stream_pool[did].append(s)
                            replicas.append(cupy.empty(array.shape, dtype=array.dtype))
                            replicas[-1].set(array, stream=s)
                handles[did] = replicas
            return handles
        src_handles = _make_handles(source, source_indices, src_stream_pool)
        dst_handles = _make_handles(target, target_indices, dst_stream_pool)

        # compute masks and local indices
        def get_blkid(i, local_block_offset):
            return cupy.searchsorted(local_block_offset, i, side='right')
        
        def _compute_blkid(handles, stream_pool):
            for did, dhandles in handles.items():
                assert len(dhandles) == 4
                indices_, block_offset_, *_ = dhandles
                with get_device(indices_) as device:
                    with stream_pool[did][1] as s:  # block_offset
                        stream_pool[did][0].synchronize()  # indices
                        block_idx = get_blkid(indices_, block_offset_)
                        dhandles[1] = block_idx
        _compute_blkid(src_handles, src_stream_pool)
        _compute_blkid(dst_handles, dst_stream_pool)

        def _compute_mask(handles, stream_pool):
            for did, dhandles in handles.items():
                assert len(dhandles) == 4
                _, block_idx_, device_idx_, _ = dhandles
                with get_device(device_idx_) as device:
                    with stream_pool[did][2] as s:   # device_idx
                        stream_pool[did][1].synchronize()  # block_offset -> blkid
                        device_idx_ = device_idx_[block_idx_]
                        dhandles[2] = (device_idx_ == did)
        _compute_mask(src_handles, src_stream_pool)
        _compute_mask(dst_handles, dst_stream_pool)

        def _mask_blkid(handles, stream_pool):
            for did, dhandles in handles.items():
                assert len(dhandles) == 4
                _, block_idx_, mask_, _ = dhandles
                with get_device(block_idx_) as device:
                    with stream_pool[did][1] as s:   # blkid
                        stream_pool[did][2].synchronize()  # device_idx -> mask
                        dhandles[1] = block_idx_[mask_]

        _mask_blkid(src_handles, src_stream_pool)
        _mask_blkid(dst_handles, dst_stream_pool)

        def _mask_indices(handles, stream_pool):
            for did, dhandles in handles.items():
                assert len(dhandles) == 4
                indices_, _, mask_, _ = dhandles
                with get_device(indices_) as device:
                    with stream_pool[did][0] as s:   # indices
                        stream_pool[did][2].synchronize()  # device_idx -> mask
                        dhandles[0] = indices_[mask_]

        _mask_indices(src_handles, src_stream_pool)
        _mask_indices(dst_handles, dst_stream_pool)

        def _compute_offset(handles, stream_pool):
            for did, dhandles in handles.items():
                assert len(dhandles) == 4
                _, block_idx_, _, device_offset_ = dhandles
                with get_device(device_offset_) as device:
                    with stream_pool[did][3] as s:   # device_offset
                        stream_pool[did][1].synchronize()  # blkid
                        dhandles[3] = device_offset_[block_idx_]
                        del dhandles[1]  # blkid

        _compute_offset(src_handles, src_stream_pool)
        _compute_offset(dst_handles, dst_stream_pool)

        def _convert_indices(handles, stream_pool):
            for did, dhandles in handles.items():
                assert len(dhandles) == 3
                indices_, _, device_offset_ = dhandles
                with get_device(indices_) as device:
                    with stream_pool[did][0] as s:   # indices
                        stream_pool[did][3].synchronize()  # dev_offset
                        # convert indices to local
                        dtype_ = dhandles[0].dtype
                        dhandles[0] = (dhandles[0] - device_offset_).astype(dtype_)
                        del dhandles[2]  # device_offset
        
        _convert_indices(src_handles, src_stream_pool)
        _convert_indices(dst_handles, dst_stream_pool)

        # build channels
        self.channels = {}
        for sdid, ddid in itertools.product(src_handles, dst_handles):
            with get_device(src_handles[sdid][0]) as device:
                with cupy.cuda.Stream(non_blocking=True) as sstream:
                    dmask_s = cupy.empty(target_indices.shape, dtype=cupy.bool_)
            with get_device(dst_handles[ddid][0]) as device:
                with cupy.cuda.Stream(non_blocking=True) as dstream:
                    smask_d = cupy.empty(source_indices.shape, dtype=cupy.bool_)
            self.channels[sdid, ddid] = [sstream, dstream, dmask_s, smask_d]
        self.emit_order = list(self.channels.keys())
        self.emit_order = [self.emit_order[i % len(self.channels)] for i in range(0, len(self.channels) * (len(dst_handles) + 1), len(dst_handles) + 1) ]

        # exchange mask
        for sdid, ddid in self.emit_order:
            channel = self.channels[sdid, ddid]
            sstrm, dstrm, dmask_s, smask_d = channel
            smask = src_handles[sdid][1]
            dmask = dst_handles[ddid][1]
            with get_device(smask) as device:
                with sstrm as s:
                    dst_stream_pool[ddid][2].synchronize()
                    dmask_s.data.copy_from_device_async(dmask.data, dmask.nbytes, stream=dstrm)
            with get_device(dmask) as device:
                with dstrm as s:
                    src_stream_pool[sdid][2].synchronize()
                    smask_d.data.copy_from_device_async(smask.data, smask.nbytes, stream=sstrm)

        # reduce mask
        for sdid, ddid in self.emit_order:
            channel = self.channels[sdid, ddid]
            sstrm, dstrm, dmask_s, smask_d = channel
            _, smask = src_handles[sdid]
            _, dmask = dst_handles[ddid]
            with get_device(smask) as device:
                with sstrm as s:
                    channel[2] = dmask_s[smask]
            with get_device(dmask) as device:
                with dstrm as s:
                    channel[3] = smask_d[dmask]

        # reduce indices
        for sdid, ddid in self.emit_order:
            channel = self.channels[sdid, ddid]
            sstrm, dstrm, smask, dmask = channel
            sindices, _ = src_handles[sdid]
            dindices, _ = dst_handles[ddid]
            with get_device(smask) as device:
                with sstrm as s:
                    src_stream_pool[sdid][0].synchronize()
                    channel[2] = sindices[smask]
            with get_device(dmask) as device:
                with dstrm as s:
                    dst_stream_pool[ddid][0].synchronize()
                    channel[3] = dindices[dmask]

        self.source = source
        self.target = target
        return

    def __call__(self):
        for sdid, ddid in self.emit_order:
            sstrm, dstrm, sindices, dindices = self.channels[sdid, ddid]
            with get_device(sindices) as device:
                with sstrm:
                    buf = self.source.device_array[sdid][sindices]
            with get_device(dindices) as device:
                with dstrm:
                    sstrm.synchronize()
                    self.target.device_array[ddid][dindices].data.copy_from_device_async(buf.data, buf.nbytes, stream=sstrm)
                    dstrm.synchronize()
                    

def assignment(left, left_indices, right, right_indices):
    if left_indices is None:
        alltoallv(right, right_indices, left)
    elif right_indices is None:
        all2ints(right, left, left_indices)
    else:
        alltoall(left, left_indices, right, right_indices)