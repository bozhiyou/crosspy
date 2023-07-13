from collections import defaultdict
from contextlib import nullcontext

import numpy
from crosspy import cupy
from crosspy.utils.cupy import _pin_memory, _pinned_memory_empty_like

from crosspy.device import get_device, _CPUDevice, _GPUDevice

# device_cache = defaultdict(dict)
# Two objects with non-overlapping lifetimes may have the same id() value.

def fetch(obj, where=None, stream=None):
    if where is None:
        return obj
    obj_dev = get_device(obj)
    if obj_dev == where:
        return obj
    
    # cache = device_cache[repr(where)]
    # key = id(obj)
    # if key in cache:
    #     assert len(cache[key]) == len(obj)  # may fire
    #     return cache[key]
    
    local_obj = pull(obj, where, stream)
    # cache[key] = local_obj
    return local_obj


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
    # to CPU
    if isinstance(context, _CPUDevice):
        src_dev = get_device(array)
        if isinstance(src_dev, _GPUDevice):  # GPU to CPU
            with src_dev:
                with cupy.cuda.Stream(non_blocking=True) as stream:
                    membuffer = _pinned_memory_empty_like(array)
                    array.get(stream=stream, out=membuffer)
                    stream.synchronize()
            return membuffer
        return array  # CPU to CPU

    # to GPU
    assert isinstance(context, _GPUDevice)
    with context:
        membuffer = cupy.empty(array.shape, dtype=array.dtype)
        with cupy.cuda.Stream(non_blocking=True) as stream_dst:
            any_to_cuda(array, stream=stream_src or stream_dst, out=membuffer)
            if stream_src: stream_src.synchronize()
            stream_dst.synchronize()
        return membuffer

pull_to = lambda context: (lambda array: pull(array, context))
