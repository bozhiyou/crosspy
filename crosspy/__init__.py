"""
CrossPy
=======

Provides
  1. Arbitrary slicing

"""

import logging
import numpy
import cupy

from types import ModuleType

__all__ = ['numpy', 'cupy', 'array']


def _get_logger(name=None, *, level=logging.WARNING, fmt=logging.BASIC_FORMAT):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


logger = _get_logger(
    __name__,
    # level=logging.DEBUG,
    fmt=
    '%(asctime)s [\033[1;4m%(levelname)s\033[0m %(processName)s:%(threadName)s] %(filename)s:%(lineno)s %(message)s'
)

from typing import TypeVar, Tuple, Union, Iterable, Callable


class _Array:
    _array_map: dict
    _shape: tuple[tuple]
    _dim: int  # concat dim; TODO: support topological concat

    def __init__(self, array_list: list, dim=None) -> None:
        if len(array_list) == 0:
            raise ValueError("Empty array not supported")

        try:
            shapes: tuple[tuple] = tuple(a.shape for a in array_list)
        except AttributeError:
            raise AttributeError("Arrays are required to have 'shape'")

        logger.debug(shapes)
        if not all([len(s) == len(shapes[0]) for s in shapes]):
            raise ValueError("Array dimensions mismatch")

        mask = [len(set(d)) == 1 for d in zip(*shapes)]
        logger.debug(mask)
        if all(mask):
            if dim is None:
                logger.info("Concat dim not specified; use 0 by default")
                dim = 0
        elif sum(mask) == len(mask) - 1:
            _dim = mask.index(False)
            if dim is None:
                dim = _dim
            elif dim != _dim:
                raise ValueError(
                    "Cannot concat on dim %s, but %s is feasible" % (dim, _dim)
                )
        else:
            raise ValueError(
                "Incompatible shapes with %s different dims" %
                (len(mask) - sum(mask))
            )
        logger.debug(dim)
        self._dim = dim
        # merge shapes
        shape = list(shapes[0])
        shape[dim] = sum([s[dim] for s in shapes])
        shape = tuple(shape)
        logger.debug(shape)
        self._shape = shape

        self._array_map = {}
        offsets = [0 for _ in range(len(shape))]  # TODO topological concat
        for array in array_list:
            logger.debug(type(array))
            key = list(array.shape)
            for i in range(len(shape)):
                key[i] = (offsets[i], offsets[i] + key[i])
                if i == dim:
                    offsets[i] += key[i][1]
            key = tuple(key)
            self._array_map[key] = array

    @property
    def shape(self):
        return tuple(self._shape)

    def __repr__(self) -> str:
        return str("array %s" % self._array_map)

    # @property
    # def devices(self):
    #     return get_placement_for_any(self._latest_view)

    # @property
    # def types(self):
    #     return [type(x) for x in self._latest_view]

    BasicIndexType = Union[int, slice, Iterable[int]]
    IndexType = Union[BasicIndexType, Tuple[BasicIndexType, ...]]

    def _index_intersection(
        self, part_range: tuple[int, int], target: BasicIndexType
    ) -> Union[BasicIndexType, None]:
        '''On one dimension, given the source range and target index, return

        TODO move to utils
        '''
        l, r = part_range
        if isinstance(
            target, int
        ) and l <= target < r:  # TODO negative indexing
            return (target - l)  # global to local
        elif isinstance(target, Iterable):
            in_range = [
                (i - l) for i in target if l <= i < r
            ]  # TODO negative indexing
            return in_range if len(in_range) else None
        elif isinstance(target, slice):
            new_start = None
            new_stop = None
            for i in range(
                target.start or 0, target.stop or r, target.step or 1
            ):
                if new_start is None and l <= i:
                    new_start = i
                if i < r:
                    new_stop = i + 1
            return slice(
                new_start - l, new_stop -
                l if new_stop is not None else None, target.step
            ) if new_start is not None else None
        return None

    def __getitem__(self, index: IndexType):  # -> Union[Array, List[Array]]
        """
        Note: CuPy handles out-of-bounds indices differently from NumPy. 
        NumPy handles them by raising an error, but CuPy wraps around them.
        """
        # unify the form to list of slices
        if not isinstance(index, tuple):
            index = (index, )

        ret = []
        for k, v in self._array_map.items():
            local_indices = [
                self._index_intersection(k[d], i) for d, i in enumerate(index)
            ]
            if all([i is not None for i in local_indices]):
                ret.append(v[tuple(local_indices)])
        # TODO check out of range in advance
        if len(ret) == 0:
            raise IndexError("Index out of range")
        # FIXME: shape may change!!!
        return _Array(ret)

    def _check_index(self, index: Tuple[BasicIndexType]):
        def _meta_check(target, max):
            if isinstance(target,
                          int) and (0 <= target < max or 0 > target >= -max):
                return True
            elif isinstance(target, Iterable):
                return all([(0 <= i < max or 0 > i >= -max) for i in target])
            elif isinstance(target, slice):
                return all(
                    [
                        i < max for i in range(
                            target.start or 0, target.stop or max,
                            target.step or 1
                        )
                    ]
                )
            raise TypeError("index out of range", target, "vs", max)

        if not all(
            [_meta_check(i, self._shape[d]) for d, i in enumerate(index)]
        ):
            raise TypeError("index out of range")

    def __setitem__(self, index: IndexType, value):
        """
        Assign :param:`value` to a partition which may not on the current device.

        :param index: index of the target partition(s)

        .. todo:
            Assignment of different values to multiple partitions (ndarrays) are currently NOT supported. The :param:`value` is assigned as a whole to each of the target partition(s).
        """
        # unify the form to list of slices
        if not isinstance(index, tuple):
            index = (index, )

        self._check_index(index)

        def _target_shape(index, caps):
            def _target_dim_shape(target, max):
                if isinstance(target,
                          int):
                    return 1
                elif isinstance(target, Iterable):
                    return sum([1 for _ in target]) # len() may not be available
                elif isinstance(target, slice):
                    return sum([1 for _ in range(
                            target.start or 0, target.stop or max,
                            target.step or 1
                        )])
                raise TypeError("unknown index type")
            return [_target_dim_shape(i, caps[d]) for d,i in enumerate(index)]

        source_shape_start = None
        for k, v in self._array_map.items():
            local_indices = [
                self._index_intersection(k[d], i) for d, i in enumerate(index)
            ]
            if all([i is not None for i in local_indices]):
                target_shape = _target_shape(local_indices, [r[1] for r in k])
                for i in range(len(target_shape), len(k)):
                    target_shape.append(k[i][1]-k[i][0])
                if source_shape_start is None:
                    source_shape_start = [0 for _ in range(len(value.shape))]
                source_shape_end = [a+b for a,b in zip(source_shape_start, target_shape[-len(value.shape):])]
                source_indices = [slice(start,stop) for start,stop in zip(source_shape_start,source_shape_end)]
                source_shape_start = source_shape_end
                if hasattr(v, 'device'): # target is cupy array
                    with v.device:
                        v[tuple(local_indices)] = cupy.asarray(value[tuple(source_indices)])
                elif hasattr(value, 'device'): # numpy <= cupy
                    v[tuple(local_indices)] = cupy.asnumpy(value[tuple(source_indices)])
                else: # numpy <= numpy
                    v[tuple(local_indices)] = value[tuple(source_indices)]

    def to(self, device):
        def _aggregate(concat, pull_op):
            output = None
            for k, v in sorted(self._array_map.items()):
                pulled = pull_op(v)
                if output is None:
                    output = pulled
                else:
                    diff_dim = -1
                    shape = [(0, s) for s in output.shape]
                    assert len(shape) == len(k)
                    for i, (range1, range2) in enumerate(zip(shape, k)):
                        if range1 != range2:
                            diff_dim = i
                            break
                    output = concat((output, pulled), axis=diff_dim)
            return output

        if isinstance(device, int):
            if device < 0:
                return _aggregate(numpy.concatenate, cupy.asnumpy)
            else:
                device = cupy.cuda.Device(device)
        with device:
            return _aggregate(cupy.concatenate, cupy.asarray)


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
    return _Array(arrayList, dim)


def array(
    obj,
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
    dim: int=None
):
    """
    Create a CrossPy array.

    :param obj: Same to ``numpy.array``.
    :param dtype: Same to ``numpy.array``.
    :param shape: Same to ``numpy.array``.
    :param dim: If ``obj`` has multiple arrays, merge them along dimension ``dim``.
    :return: A CrossPy array.
    """
    if obj is None:
        raise ValueError("None initialization not supported")

    elif isinstance(obj, (list, tuple)):
        # TODO: recursive iterables
        return fromarrays(obj, dtype=dtype, shape=shape, dim=dim)

    else:
        return array((obj, ), dtype=dtype, shape=shape, dim=dim)

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
