"""
CrossPy
=======

Provides
  1. Arbitrary slicing

"""
import warnings
warnings.simplefilter("default")

import numpy
try:
    import cupy
    import cupy.cuda
except (ImportError, AttributeError) as e:
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all(
        "sphinx" not in f.filename
        for f in inspect.getouterframes(inspect.currentframe())
    ):
        import traceback
        warnings.warn(ImportWarning(traceback.format_exc() + "\nFailed to import CuPy due to the error above. GPU functionalities not available."))
    cupy = None

from typing import Iterable, Optional
from types import ModuleType

from .device.cpu import cpu
if cupy:
    from .device.gpu import gpu

from .core import CrossPyArray

from .ldevice import PartitionScheme, partition

from .mpi import alltoallv, all2ints, assignment

from . import utils

# from .device import get_all_devices
# print(get_all_devices())

__all__ = ['numpy', 'cupy', 'array', 'cpu', 'gpu', 'PartitionScheme', 'partition']

class PerObjWrapper:
    initial_devices = {}
    initial_shapes = {}

    def __init__(self, wrapper, perserve_device=False, perserve_shape=False) -> None:
        def attr_wrapper(obj, *args, **kwds):
            device_ = getattr(obj, "device", "cpu") if perserve_device else None
            shape_ = getattr(obj, "shape", None) if perserve_shape else None

            wrapped_obj = wrapper(obj, *args, **kwds)

            if not hasattr(wrapped_obj, "device") and device_ is not None:
                self.initial_devices[id(wrapped_obj)] = device_
            if not hasattr(wrapped_obj, "shape") and shape_ is not None:
                self.initial_shapes[id(wrapped_obj)] = shape_

            return wrapped_obj

        self._wrapper = attr_wrapper

    def __call__(self, obj, *args, **kwds):
        return self._wrapper(obj, *args, **kwds)


def array(
    obj: Iterable,
    dtype=None,
    shape=None,
    # offset=0,
    # strides=None,
    # formats=None,
    # names=None,
    # titles=None,
    # aligned=False,
    # byteorder=None,
    # copy=True,
    axis: Optional[int] = None,
    dim: Optional[int] = None,
    *,
    distribution=None,
    wrapper=None,
):# -> CrossPyArray:
    """
    Create a CrossPy array.

    :param obj: Same to ``numpy.array``.
    :param dtype: Same to ``numpy.array``.
    :param shape: Same to ``numpy.array``.
    :param axis: Concatenate ``obj`` along ``axis``.
    :param distribution: Partition ``obj`` according to ``distribution`` scheme. Same as ``partition(obj, distribution)``.
    :param wrapper: Applied to each subarray.
    :return: A CrossPy array.
    :rtype: class:`CrossPyArray`
    """
    assert obj is not None, NotImplementedError("array with no content not supported")

    if dim is not None:
        warnings.warn(DeprecationWarning("`dim` is deprecated; use `axis` instead"))
    axis = dim if dim is not None else axis

    if wrapper is not None:
        wrapper = PerObjWrapper(wrapper, perserve_device=True, perserve_shape=True)

    if distribution is not None:
        obj = partition(obj, distribution=distribution, wrapper=wrapper)

    from .array import is_array
    def inner(obj, axis):
        if is_array(obj):  # numpy, cupy, crosspy
            if wrapper:
                arr = CrossPyArray(wrapper(obj), axis=axis, initial_devices=wrapper.initial_devices, initial_shapes=wrapper.initial_shapes)
            else:
                arr = CrossPyArray(obj, axis=axis)
        elif isinstance(obj, (list, tuple)):
            obj = type(obj)(x if is_array(x) else inner(x, None if axis in (None, 0) else axis - 1) for x in obj)
            assert all(is_array(a) for a in obj)
            if wrapper:
                arr = CrossPyArray(type(obj)(wrapper(a)for a in obj), axis, initial_devices=wrapper.initial_devices, initial_shapes=wrapper.initial_shapes)
            else:
                arr = CrossPyArray(obj, axis=axis)
        else:
            raise NotImplementedError("cannot convert %s to CrossPy array" % type(obj))

        assert isinstance(arr, CrossPyArray), type(arr)
        return arr
    
    return inner(obj, axis=axis)

def asnumpy(input: CrossPyArray):
    return numpy.asarray(input)

def zeros(shape, placement):
    """
    Only support 1-D placement.
    """
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    n_parts = len(placement)
    sub_shapes = [(shape[0] // n_parts, *shape[1:]) for i in range(n_parts)]
    if shape[0] != shape[0] // n_parts * n_parts:
        sub_shapes[-1] = (n_parts - shape[0] // n_parts * (n_parts - 1), *shape[1:])
    def array_gen(i: int):
        if isinstance(placement[i], gpu(0).__class__): # TODO this is an ugly check
            with placement[i].cupy_device:
                return placement[i].get_array_module().zeros(sub_shapes[i])
        return placement[i].get_array_module().zeros(sub_shapes[i])
    return CrossPyArray.from_shapes(sub_shapes, array_gen, dim=0)

def to(input, device: int):
    """
    Move CrossPy arrays to the device identified by device.

    :param input: The input array
    :type input: class:`CrossPyArray`
    :param device: If ``device`` is a negative integer, the target device is CPU; otherwise GPU with the corresponding ID.
    :type device: int | class:`cupy.cuda.Device`
    :return: NumPy array if ``device`` refers to CPU, otherwise CuPy array.
    """
    return input.to(device)


def config_backend(backend):
    if isinstance(backend, ModuleType):
        backend = backend.__name__
    import sys
    submodules = {}
    for k, v in sys.modules.items():
        if k.startswith(f"{backend}."):
            setattr(sys.modules[__name__], k[len(backend) + 1:], v)
            submodules[k.replace(backend, __name__)] = v
    sys.modules.update(submodules)


config_backend(numpy)
