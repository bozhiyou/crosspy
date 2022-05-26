"""
CrossPy
=======

Provides
  1. Arbitrary slicing

"""

from typing import Iterable, Optional
import numpy
import cupy

from types import ModuleType

from .core.ndarray import IndexType

from .device.cpu import cpu
from .device.gpu import gpu

from .core import CrossPyArray

from .ldevice import PartitionScheme

# from .device import get_all_devices
# print(get_all_devices())

__all__ = ['numpy', 'cupy', 'array', 'cpu', 'gpu', 'PartitionScheme']


def fromarrays(
    arrayList,
    dtype=None,
    shape=None,
    formats=None,
    names=None,
    titles=None,
    aligned=False,
    byteorder=None,
    dim=None
):
    return CrossPyArray.from_array_list(arrayList, dim)


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
    dim: Optional[int] = None,
    *,
    partition=None,
    placement=None
):
    """
    Create a CrossPy array.

    :param obj: Same to ``numpy.array``.
    :param dtype: Same to ``numpy.array``.
    :param shape: Same to ``numpy.array``.
    :param dim: If ``obj`` has multiple arrays, merge them along dimension ``dim``.
    :param partition: A tuple of partitioning scheme.
    :return: A CrossPy array.
    """
    if obj is None:
        raise NotImplementedError("array with no content not supported")

    from .array import is_array
    if not is_array(obj):
        try:
            arr = numpy.asarray(obj) # TODO: hinted by placement
        except:
            if isinstance(obj, (list, tuple)):
                def _recursive_parse(seq, d):
                    if all(is_array(a) for a in seq):
                        return fromarrays(seq, dtype=dtype, shape=shape, dim=d)
                    if all(isinstance(o, (list, tuple)) for o in seq):
                        d = d or 0
                        return fromarrays([_recursive_parse(o, d+1) for o in seq], dtype=dtype, shape=shape, dim=d)
                    raise NotImplementedError
                arr = _recursive_parse(obj, dim)
    else:
        arr = obj

    # TODO: necessary at this point?
    arr = fromarrays(
        (arr,),
        dtype=dtype,
        shape=shape,
        dim=dim
    )

    if partition is None and placement is None:
        return arr

    if placement is not None:
        from .ldevice import LDeviceSequenceBlocked
        Partitioner = LDeviceSequenceBlocked
        mapper = Partitioner(len(placement), placement=placement)
        arr_p = mapper.partition_tensor(arr)
        return CrossPyArray.from_array_list(arr_p, dim)

    if partition is not None:
        from .ldevice import LDeviceSequenceArbitrary
        Partitioner = LDeviceSequenceArbitrary
        mapper = Partitioner(partition)
        arr_p = mapper.partition_tensor(arr)
        return CrossPyArray.from_array_list(arr_p, dim)


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
    :type input: CrossPy array
    :param device: If ``device`` is a negative integer, the target device is CPU; otherwise GPU with the corresponding ID.
    :type device: int | cupy.cuda.Device
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
