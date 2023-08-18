import cupy
import cupy.cuda

from contextlib import contextmanager
from functools import wraps, lru_cache
import os

import numpy

import logging

logger = logging.getLogger(__name__)

import crosspy
from crosspy import context
from crosspy import device
from crosspy.device import get_device, MemoryKind, Memory, register_memory
from crosspy.device.meta import Architecture, Device, DeviceMeta
from crosspy.utils.array import ArrayType, register_array_type


__all__ = ["gpu", "cupy"]


def _pinned_memory_empty_like(array):
    if isinstance(array, numpy.ndarray):
        mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
        ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        return ret
    return array

def _pin_memory(array):
    ret = _pinned_memory_empty_like(array)
    ret[...] = array
    return ret


@device.of(cupy.ndarray)
def default_cupy_device(np_arr):
    assert(isinstance(np_arr, (cupy.ndarray)))
    return gpu(np_arr.device.id)


class CuPyCarrier:
    def __init__(self, stream=None):
        self.stream = stream

    def __del__(self):
        if self.stream:
            self.stream.synchronize()

    @property
    def module(self):
        return cupy

    def pull(self, src, out=None, copy=False):
        if isinstance(src, crosspy.ndarray):
            if src.shape != ():
                raise ValueError("use tobuffer() to pull non-scalars")
            src = src.item()
        if isinstance(src, numpy.ndarray):
            buf = cupy.empty(src.shape, dtype=src.dtype) if out is None else out
            buf.set(src, stream=self.stream)
            return buf
        if isinstance(src, cupy.ndarray):
            dev_id = cupy.cuda.runtime.getDevice()
            if out is not None and out.device.id != dev_id:
                raise ValueError("`out` on device %d is different from current device %d" % (out.device.id, dev_id))
            if src.device.id == dev_id and out is None:
                return src if not copy else src.astype(src.dtype, copy=True)
            buf = cupy.empty(src.shape, dtype=src.dtype) if out is None else out
            buf.data.copy_from_device_async(src.data, src.nbytes, stream=self.stream)
            return buf
        try:
            return cupy.array(src)
        except BaseException:
            raise TypeError(f"{type(src)} is not supported yet")

    def copy(self, dst, dst_idx, src, src_idx, stream=None):
        src_buf = src[src_idx]
        dst_buf = dst[dst_idx]
        if isinstance(src, numpy.ndarray):
            dst_buf.set(src_buf, stream=stream or self.stream)
            return
        if isinstance(src, cupy.ndarray):
            dst_buf.data.copy_from_device_async(src_buf.data, src_buf.nbytes, stream=stream or self.stream)
            return
        raise TypeError(f"{type(src)} is not supported yet")



@context.register(cupy)
@contextmanager
def cupy_context(obj, stream=None):
    with get_device(obj) as ctx:
        if stream is None:
            yield ctx
            return
        if isinstance(stream, dict):
            stream = cupy.cuda.Stream(**stream)
        yield CuPyCarrier(stream)


class _DeviceCUPy:
    def __init__(self, ctx: "GPUDevice"):
        self._ctx: "GPUDevice" = ctx

    def __getattr__(self, item: str):
        v = getattr(cupy, item)
        if callable(v):

            def _wrap_for_device(ctx: "GPUDevice", f):
                @wraps(f)
                def ff(*args, **kwds):
                    with ctx.cupy_device_context():
                        return f(*args, **kwds)

                return ff

            return _wrap_for_device(self._ctx, v)
        return v


class _GPUMemory(Memory):
    @property
    @lru_cache(None)
    def np(self):
        return _DeviceCUPy(self.device)

    def asarray_async(self, src):
        if isinstance(src, cupy.ndarray) and src.device.id == self.device.index:
            return src
        if not (src.flags['C_CONTIGUOUS'] or src.flags['F_CONTIGUOUS']):
            raise NotImplementedError(
                'Only contiguous arrays are currently supported for gpu-gpu transfers.'
            )
        dst = cupy.empty_like(src)
        dst.data.copy_from_device_async(src.data, src.nbytes)
        return dst

    def __call__(self, target):
        # TODO Several threads could share the same device object.
        #      It causes data race and CUDA context is incorrectly set.
        #      For now, this remove assumes that one device is always
        #      assigned to one task.
        # FIXME This code breaks the semantics since a different device
        #       could copy data on the current device to a remote device.
        #with self.device._device_context():
        with cupy.cuda.Device(self.device.index):
            if isinstance(target, numpy.ndarray):
                logger.debug("Moving data: CPU => %r", cupy.cuda.Device())
                return cupy.asarray(target)
            elif isinstance(target, cupy.ndarray) and \
                 cupy.cuda.Device() != getattr(target, "device", None):
                logger.debug(
                    "Moving data: %r => %r", getattr(target, "device", None),
                    cupy.cuda.Device()
                )
                return self.asarray_async(target)
            elif hasattr(target, 'all_to'):  # FIXME for crosspy array
                return target.all_to(self.device.index)
            else:
                logger.debug(
                    "NOT moving data of type %r to %r", target,
                    cupy.cuda.Device()
                )
                return target


class GPUDevice(Device, metaclass=DeviceMeta):
    def __init__(self, architecture: "_GPUArchitecture", index, *args, **kwds):
        try:
            with cupy.cuda.Device(index) as self._d:
                with cupy.cuda.Stream(non_blocking=True) as self._s:
                    if os.getenv('CUPY_INIT_MEMPOOL', '1').lower() not in ('0', 'false'):
                        # TODO config prealloc
                        cupy.cuda.alloc(self._d.mem_info[0] & ~((1 << 30) - 1))  # maximize prealloc to GB
        except cupy.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(e.args[0] + " %d" % index)
        super().__init__(architecture, index, *args, **kwds)

    def __del__(self):
        self._s.synchronize()
        self._d.synchronize()

    def __getattr__(self, name):
        return getattr(self._d, name)

    @property
    def cupy_device(self):
        return self._d

    @property
    @lru_cache(None)
    def resources(self):
        dev = self._d
        free, total = dev.mem_info
        attrs = dev.attributes
        threads = attrs["MultiProcessorCount"] * attrs["MaxThreadsPerMultiProcessor"]
        return free, total, threads

    @contextmanager
    def cupy_device_context(self):
        with self.cupy_device:
            yield

    @lru_cache(None)
    def memory(self, kind: MemoryKind = None):
        return _GPUMemory(self, kind)

    def get_array_module(self):
        return cupy

    def __repr__(self):
        return "<CUDA {}>".format(self.index)

    def __enter__(self):
        self._d.__enter__()
        self._s.__enter__()
        return CuPyCarrier(self._s)

    def __exit__(self, *exctype):
        self._s.__exit__(*exctype)
        self._d.__exit__(*exctype)


class _GPUArchitecture(Architecture):
    _devices: list[GPUDevice]

    def __init__(self, name, id):
        super().__init__(name, id)
        self._devices = [
            GPUDevice(self, device_id) for device_id in range(cupy.cuda.runtime.getDeviceCount())]

    @property
    def devices(self):
        return self._devices
    
    def __iter__(self):
        return iter(self._devices)

    def __getitem__(self, index):
        return self._devices[index]

    def __call__(self, index, *args, **kwds):
        return self._devices[index]

class _CuPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        # TODO: We should be able to do direct copies from numpy to cupy arrays, but it doesn't seem to be working.
        # return isinstance(b, (cupy.ndarray, numpy.ndarray))
        return isinstance(b, cupy.ndarray)

    def get_memory(self, a):
        return gpu(a.device.id).memory()

    def get_array_module(self, a):
        return cupy.get_array_module(a)

register_memory(GPUDevice)(CuPyCarrier)
register_array_type(cupy.ndarray)(_CuPyArrayType())

GPUDevice.register(cupy.cuda.Device)

gpu = _GPUArchitecture("GPU", "gpu")
gpu.__doc__ = """Architecture for CUDA GPUs.

>>> gpu(0)
"""
