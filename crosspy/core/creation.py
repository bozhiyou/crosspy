from collections.abc import Sequence
from numbers import Integral

from warnings import warn
from operator import attrgetter

import numpy
import crosspy as xp

from crosspy.device import Device
from crosspy.context import context
from crosspy.utils.array import is_array

def shape_factory(method: str, *shape, device=None, mode='raise', **kwargs):
    """
    If no shape is specified, default to `call_from(_py)()`.

    :param distribution:
        - An iterable of devices. In this case, the array-like object can will be
        devided (almost) equally. Size of each partition is guaranteed to be
        either ceil(s) or floor(s) where s = total_size / len(devices)
        - An iterable of integers. In this case, the array-like object will be
        devided accordingly without changing its device (i.e. no communication)
        except when the iterable is an array and the partitions will be on the
        device of `distribution`
        - An iterable of (device, integer) pairs, which can typically be a zip
        or dict.items object
    :param mode:
        Specifies how out-of-bounds `distribution` will behave.
        - None - Skip all checks (default)
        - 'raise' - raise an error
        - 'wrap' - wrap around
        - 'clip' - clip to the range
        'clip' mode means that sizes that are larger than the size of `array_like`
        will be ignored.
    """
    call_from = attrgetter(method)
    _py = numpy

    if len(shape) == 0:
        if device is None:
            return call_from(_py)(**kwargs)
        assert isinstance(device, Device)
        with device as ctx:
            _py = ctx.module
            return call_from(_py)(*shape, **kwargs)
    
    axis = None
    for i, size in enumerate(shape):
        if not isinstance(size, Integral):
            if axis is not None:
                raise TypeError(f"more than one non-integral axis f{axis} and f{i}")
            axis = i
    if axis is None:
        if device is None:
            return call_from(_py)(*shape, **kwargs)
        with device as ctx:
            _py = ctx.module
            return call_from(_py)(*shape, **kwargs)

    layout = shape[axis]
    try:
        nparts = len(layout)
    except TypeError:
        layout = (*layout,)
        nparts = len(layout)

    if nparts == 0:
        raise TypeError(f"Failed to parse {layout} as a distributed dimension")
    if mode == 'raise' and isinstance(device, Sequence) and nparts != len(device):
        raise ValueError("Inconsistent device-size matching")

    partitions = []
    # sizes by array
    if is_array(layout):
        assert layout.dtype.kind == 'i', f"array of sizes must have integer dtype, not {layout.dtype}"
        if isinstance(device, Sequence):
            for d, s in zip(device, layout):
                with d as ctx:
                    _py = ctx.module
                    partitions.append(call_from(_py)(*shape[:axis], s, *shape[axis + 1:], **kwargs))
        else:
            with device or context(layout) as ctx:
                _py = ctx.module
                partitions = [call_from(_py)(*shape[:axis], s, *shape[axis + 1:], **kwargs) for s in layout]
        return xp.array(partitions, axis=axis)

    try:
        peeked = layout[0]
    except TypeError:
        layout = (*layout,)
        peeked = layout[0]
    if isinstance(peeked, Integral):
        if isinstance(device, Sequence):
            for d, s in zip(device, layout):
                with d as ctx:
                    _py = ctx.module
                    partitions.append(call_from(_py)(*shape[:axis], s, *shape[axis + 1:], **kwargs))
        elif device is not None:
            with device as ctx:
                _py = ctx.module
                partitions = [call_from(_py)(*shape[:axis], s, *shape[axis + 1:], **kwargs) for s in layout]
        else:
            partitions = [call_from(_py)(*shape[:axis], s, *shape[axis + 1:], **kwargs) for s in layout]
        return xp.array(partitions, axis=axis)
    if isinstance(peeked, tuple):
        if device is not None:
            warn("with device-size pairs, `device` keyword is ignored")
        for d, s in zip(device, layout):
            with d as ctx:
                _py = ctx.module
                partitions.append(call_from(_py)(*shape[:axis], s, *shape[axis + 1:], **kwargs))
        return xp.array(partitions, axis=axis)
    raise TypeError(f"Unrecognizable distribution with type {type(peeked)}")

def empty(*shape, device=None, mode='raise', **kwargs):
    return shape_factory('empty', *shape, device=device, mode=mode, **kwargs)

def zeros(*shape, device=None, mode='raise', **kwargs):
    return shape_factory('zeros', *shape, device=device, mode=mode, **kwargs)

def ones(*shape, device=None, mode='raise', **kwargs):
    return shape_factory('ones', *shape, device=device, mode=mode, **kwargs)

def like_factory(method: str, prototype, **kwargs):
    call_from = attrgetter(method)

    if isinstance(prototype, xp.ndarray):
        device_array = {}
        for did, array in prototype.device_array.items():
            with context(array) as ctx:
                _py = ctx.module
                device_array[did] = call_from(_py)(array, **kwargs)
        return xp.CrossPyArray(prototype.shape, prototype.dtype, prototype.metadata, device_array, axis=prototype.axis)
    if is_array(prototype):
        with context(prototype) as ctx:
            with context(prototype) as ctx:
                _py = ctx.module
                return call_from(_py)(prototype, **kwargs)
    raise TypeError(f"type {type(prototype)} is not suppported yet")

def empty_like(prototype, **kwargs):
    return like_factory('empty_like', prototype, **kwargs)

def zeros_like(prototype, **kwargs):
    return like_factory('zeros_like', prototype, **kwargs)

def ones_like(prototype, **kwargs):
    return like_factory('ones_like', prototype, **kwargs)
