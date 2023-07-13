import asyncio
import functools
import itertools
from contextlib import nullcontext
from functools import lru_cache

import numpy
from crosspy import cupy
from crosspy.utils.cupy import _pin_memory, _pinned_memory_empty_like

from crosspy.device import get_device

from .cache import same_place, any_to_cuda


# compute masks and local indices
def get_blkid(i, local_block_offset, out=None):
    if out is not None:
        cupy._sorting.search._searchsorted_kernel(i, local_block_offset, local_block_offset.size, True, True, out)
        return
    return cupy.searchsorted(local_block_offset, i, side='right')

class DeviceProxy:
    def __init__(self, id, device, *, stream_pool=None):
        self.__id = id
        self.__device = device
        self._stream_pool = stream_pool or []

    @property
    def id(self): return self.__id
    @property
    def device(self): return self.__device

    def here(func: ...):  # type: ignore
        """(decorator) execute under current device"""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.device:
                if "stream" in kwargs and kwargs['stream'] is not None:
                    with kwargs['stream']:
                        return func(self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)
        return wrapper

    async def setup(self, metadata, indices, debug=None):
        device, stream_pool = self.device, self._stream_pool
        # make replicas
        # TODO: indices may not cover all devices hence no need to replicate for all
        # replicate indices
        with device:
            with cupy.cuda.Stream(non_blocking=True) as s:
                stream_pool.append(s)
                indices_ = cupy.empty((2, *indices.shape), dtype=indices.dtype)  # 0 for indices, 1 for blkid
                indices_[0].set(indices, stream=s)
        await asyncio.sleep(0)
        # replicate metadata
        with device:
            replicas = []
            for array in metadata:
                with cupy.cuda.Stream(non_blocking=True) as s:
                    stream_pool.append(s)
                    replicas.append(cupy.empty(array.shape, dtype=array.dtype))
                    replicas[-1].set(array, stream=s)
        block_offset_, device_idx_, device_offset_ = replicas
        del replicas
        await asyncio.sleep(0)
        ########
        # compute blk id
        ########
        with device:
            with stream_pool[0] as s:  # indices
                stream_pool[1].synchronize()  # block_offset
                # stream sync state [x s x x]
                get_blkid(indices_[0], block_offset_, out=indices_[1])
                del block_offset_
        await asyncio.sleep(0)
        ########
        # get device id and compute mask
        ########
        with device:
            with stream_pool[0] as s:  # indices -> [indices, blkid]
                stream_pool[2].synchronize()  # device_idx
                # stream sync state [x s s x]
                device_idx_ = device_idx_[indices_[1]]
        await asyncio.sleep(0)
        with device:
            with stream_pool[0] as s:  # indices -> [indices, blkid] -> device_idx
                mask_ = (device_idx_ == self.id)
                del device_idx_
        await asyncio.sleep(0)
        self._handles = mask_, indices_, device_offset_

    def free(self):
        del self._handles

    async def pair(self, other_mask, debug=None):
        my_mask, indices, device_offset = self._handles
        device = self.device
        with device:
            with cupy.cuda.Stream(non_blocking=True) as stream:
                mask = cupy.empty(other_mask.shape, dtype=cupy.bool_)
                mask.data.copy_from_device_async(other_mask.data, other_mask.nbytes, stream=stream)
        await asyncio.sleep(0)
        with device:
            with stream:
                self._stream_pool[0].synchronize()
                mask = mask & my_mask
        await asyncio.sleep(0)
        ########
        # mask indices and block id
        ########
        with device:
            with stream:
                # NOTE several equivalences:
                #   indices_ = indices_[:, mask]  # slow, 1 copy bool + 2 scan
                #   indices_ = cupy.compress(mask, indices_, axis=1)  # compress by bool = nonzero + take
                #   maskidx = mask.nonzero().squeeze(axis=-1)  # argwhere implements nonzero/where TODO: inplace hack cupy/_core/_routines_indexing.pyx:_ndarray_argwhere
                idx = cupy.argwhere(mask).squeeze(axis=-1)
        await asyncio.sleep(0)
        with device:
            with stream:
                indices = indices[:, idx]
        ########
        # gather offset and convert indices
        ########
        with device:
            with stream:
                self._stream_pool[3].synchronize()  # device_offset
                # stream sync state [x s s s]
                # NOTE use take to avoid memcpy in indices_[1] = device_offset_[indices_[1]]
                device_offset.take(indices[1], out=indices[1])
                del device_offset  # device_offset
        await asyncio.sleep(0)
        with device:
            with stream:
                # convert indices to local
                cupy.subtract(indices[0], indices[1], out=idx)  # mem reuse
        return stream, idx

class Channel:
    def __init__(self, src: DeviceProxy, dst: DeviceProxy):
        self.src = src
        self.dst = dst

    async def setup(self):
        (self.src_stream, self.srcidx), (self.dst_stream, self.dstidx) = await asyncio.gather(
            self.src.pair(self.dst._handles[0]),
            self.dst.pair(self.src._handles[0]),
        )

    async def launch(self, source, target):
        sdid, ddid = self.src.id, self.dst.id
        src = source.device_array[sdid]
        dst = target.device_array[ddid]
        with self.src.device as sdevice:
            with self.src_stream as sstrm:
                sendbuf = src[self.srcidx]
        await asyncio.sleep(0)
        with self.dst.device as ddevice:
            with self.dst_stream as dstrm:
                sstrm.synchronize()
                recvbuf = cupy.empty(sendbuf.shape, dtype=sendbuf.dtype)
                recvbuf.data.copy_from_device_async(sendbuf.data, sendbuf.nbytes, stream=dstrm)
        await asyncio.sleep(0)
        with self.dst.device as ddevice:
            with self.dst_stream as dstrm:
                dst[self.dstidx] = recvbuf
        await asyncio.sleep(0)
        with self.dst.device as ddevice:
            with self.dst_stream as dstrm:
                dstrm.synchronize()

class alltoall:
    def __init__(self, target, dst_indices, source, src_indices, debug=None):
        assert dst_indices.dtype.char in ['l', 'L'] and src_indices.dtype.char in ['l', 'L'], \
            "Only support indices of 64-bit dtype (got %s/%s); use astype for conversion" % (dst_indices.dtype, src_indices.dtype)
        self.source = source
        self.target = target
        init_coro = self.init_async(target, dst_indices, source, src_indices, debug)
        try:
            asyncio.run(init_coro)
        except RuntimeError:
            self.init_coro = init_coro
        else:
            self.init_coro = None

    def __await__(self):
        if self.init_coro:
            return self.init_coro.__await__()

    async def init_async(self, target, target_indices, source, source_indices, debug=None):
        if isinstance(target_indices, numpy.ndarray):
            target_indices = _pin_memory(target_indices)
        if isinstance(source_indices, numpy.ndarray):
            source_indices = _pin_memory(source_indices)

        # set up proxies
        src_proxies = [DeviceProxy(did, get_device(darray)) for did, darray in source.device_array.items()]
        dst_proxies = [DeviceProxy(did, get_device(darray)) for did, darray in target.device_array.items()]
        # set up channels
        self.emit_order = list(itertools.product(src_proxies, dst_proxies))
        self.emit_order = [self.emit_order[i % len(self.emit_order)] for i in range(0, len(self.emit_order) * (len(dst_proxies) + 1), len(dst_proxies) + 1)]
        self.channels = [Channel(src, dst) for src, dst in self.emit_order]

        async def _endpoint_setup(proxies, metadata, indices, debug=None):
            await asyncio.gather(*(
                dp.setup(metadata, indices, debug)
                for dp in proxies
            ))

        await asyncio.gather(
                _endpoint_setup(src_proxies, source.metadata, source_indices, 'src'),
                _endpoint_setup(dst_proxies, target.metadata, target_indices, 'dst')
            )

        await asyncio.gather(*(
            edge.setup() for edge in self.channels
        ))

        for n in (*src_proxies, *dst_proxies):
            n.free()

        return self
    
    def __call__(self):
        coro = self.run_async()
        try:
            asyncio.run(coro)
        except RuntimeError:
            return coro

    async def run_async(self):
        await asyncio.gather(*(
            channel.launch(self.source, self.target) for channel in self.channels
        ))

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
