"""
CrossPy
=======

Provides
  1. Arbitrary slicing

"""
from ._backend import set_backend
import numpy

# set_backend(numpy)

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
# else:
#     if cupy.cuda.runtime.driverGetVersion() >= 11020:
#         try:  # use asynchronous stream ordered memory
#             cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)
#         except RuntimeError:
#             pass

from typing import Iterable, Optional

import warnings
warnings.simplefilter("default")

from . import random, utils

from crosspy.device import get_device
from crosspy.device.cpu import cpu
if cupy:
    from .device.gpu import gpu
    from .utils.cupy import _pin_memory, _pinned_memory_empty, _pinned_memory_empty_like

from .core import CrossPyArray
ndarray = CrossPyArray
from .core import empty, zeros, ones
from .core import empty_like, zeros_like, ones_like

from .partition import PartitionScheme, split

from .transfer import alltoallv, alltoall

from crosspy.utils.wrapper import DynamicObjectManager

# from .device import get_all_devices
# print(get_all_devices())

__all__ = [
    'random', 'utils',
    'numpy', 'cupy', 'array', 'cpu', 'gpu', 'PartitionScheme', 'split'
    ]


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
    dim: Optional[int] = None,  # DEPRECATED
    *,
    distribution=None,
    data_manager=None,
):# -> CrossPyArray:
    """
    Create a CrossPy array.

    :param obj: Same to ``numpy.array``.
    :param dtype: Same to ``numpy.array``.
    :param shape: Same to ``numpy.array``.
    :param axis: Concatenate ``obj`` along ``axis``.
    :param distribution: Partition ``obj`` according to ``distribution`` scheme. Same as ``partition(obj, distribution)``.
    :param data_manager: Applied to each subarray.
    :return: A CrossPy array.
    :rtype: class:`CrossPyArray`
    """
    assert obj is not None, NotImplementedError("array with no content not supported")

    if dim is not None:
        warnings.warn(DeprecationWarning("`dim` is deprecated; use `axis` instead"))
    axis = dim if dim is not None else axis

    if distribution is not None:
        obj = split(obj, distribution=distribution, axis=axis)

    # Parla-specific
    if data_manager is not None:
        assert isinstance(data_manager, DynamicObjectManager), "`data_manager` should be derived from `utils.DynamicObjectManager`"

    from .utils.array import is_array
    def inner(obj, axis):
        if isinstance(obj, (list, tuple)):
            obj = type(obj)(x if is_array(x) else inner(x, None if axis in (None, 0) else axis - 1) for x in obj)
            assert all(is_array(a) for a in obj)
        else:
            assert is_array(obj), NotImplementedError("cannot convert %s to CrossPy array" % type(obj))

        return CrossPyArray.fromobject(obj, axis=axis, data_manager=data_manager).finish()

    return inner(obj, axis=axis)

def asnumpy(input: CrossPyArray):
    return numpy.asarray(input)

def zeros(shape, distribution):
    """
    Only support 1-D distribution.
    """
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    n_parts = len(distribution)
    axis = 0
    sub_shapes = [(*shape[:axis], shape[axis] // n_parts, *shape[axis + 1:]) for _ in range(n_parts)]
    if shape[0] != shape[0] // n_parts * n_parts:
        sub_shapes[-1] = (*shape[:axis], n_parts - shape[0] // n_parts * (n_parts - 1), *shape[axis + 1:])
    def array_gen(i: int):
        if isinstance(distribution[i], gpu(0).__class__): # TODO this is an ugly check
            with distribution[i].cupy_device:
                return distribution[i].get_array_module().zeros(sub_shapes[i])
        return distribution[i].get_array_module().zeros(sub_shapes[i])
    return CrossPyArray.fromobject([array_gen(i) for i in range(n_parts)], axis=0).finish()

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