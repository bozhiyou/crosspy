import builtins
from abc import ABCMeta, abstractmethod
from typing import Dict

import logging
logger = logging.getLogger(__name__)

import numpy, numpy as np

from crosspy.device import device
from crosspy.utils import get_module

__all__ = [
    "get_array_module", "get_memory", "is_array", "asnumpy", "storage_size"
]

_array_types: Dict[type, 'ArrayType'] = dict()

def get_array_module(a):
    """
    :param a: A numpy-compatible array.
    :return: The numpy-compatible module associated with the array class (e.g., cupy or numpy).
    """
    mod = get_module(a)
    if mod is builtins:
        return numpy
    return mod or _array_types[type(a)].get_array_module(a)


def is_array(a) -> bool:
    """
    :param a: A value.
    :return: True if `a` is an array of some type known to parla.
    """
    return type(a) in _array_types


def asnumpy(a):
    try:
        ar = get_array_module(a)
        if hasattr(ar, "asnumpy"):
            return getattr(ar, "asnumpy")(a)
    except KeyError:
        pass  # numpy.int64
    return np.asarray(a)


class ArrayType(metaclass=ABCMeta):
    @classmethod
    def register(cls, ty):
        def inner(get_memory_impl: ArrayType):
            _array_types[ty] = get_memory_impl

        return inner

    @abstractmethod
    def get_memory(self, a):
        """
        :param a: An array of self's type.
        :return: The memory containing `a`.
        """
        pass

    @abstractmethod
    def can_assign_from(self, a, b):
        """
        :param a: An array of self's type.
        :param b: An array of any type.
        :return: True iff `a` supports assignments from `b`.
        """
        pass

    @abstractmethod
    def get_array_module(self, a):
        """
        :param a: An array of self's type.
        :return: The `numpy` compatible module for the array `a`.
        """
        pass

def register_array_type(ty):
    def register(get_memory_impl: ArrayType):
        _array_types[ty] = get_memory_impl

    return register


def can_assign_from(a, b):
    """
    :param a: An array.
    :param b: An array.
    :return: True iff `a` supports assignments from `b`.
    """
    return _array_types[type(a)].can_assign_from(a, b)


def get_memory(a) -> "device.Memory":
    """
    :param a: An array object.
    :return: A memory in which `a` is stored.
    (Currently multiple memories may be equivalent, because they are associated with CPUs on the same NUMA node,
    for instance, in which case this will return one of the equivalent memories, but not necessarily the one used
    to create the array.)
    """
    if not is_array(a):
        raise TypeError(
            "Array required, given value of type {}".format(type(a))
        )
    return _array_types[type(a)].get_memory(a)


def storage_size(*arrays):
    """
    :return: the total size of the arrays passed as arguments.
    """
    return sum(a.size * a.itemsize for a in arrays)
