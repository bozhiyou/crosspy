import numpy
import cupy

def _pinned_memory_empty(shape, dtype):
    mem = cupy.cuda.alloc_pinned_memory(numpy.prod(shape) * (dtype().itemsize if isinstance(dtype, type) else dtype.itemsize))
    ret = numpy.frombuffer(mem, dtype, numpy.prod(shape)).reshape(shape)
    return ret

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