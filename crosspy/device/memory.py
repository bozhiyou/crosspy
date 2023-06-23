from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Optional

from .detail import Detail


class MemoryKind(Enum):
    """
    MemoryKinds specify a kind of memory on a device.
    """
    Fast = "local memory or cache prefetched"
    Slow = "DRAM or similar conventional memory"


class Memory(Detail, metaclass=ABCMeta):
    """
    Memory locations are specified as a device and a memory type:
    The `Device` specifies the device which has direct (or primary) access to the location;
    The `Kind` specifies what kind of memory should be used.

    A `Memory` instance can also be used as a detail on data references (such as an `~numpy.ndarray`) to copy the data
    to the location. If the original object is already in the correct location, it is returned unchanged, otherwise
    a copy is made in the correct location.
    There is no relationship between the original object and the new one, and the programmer must copy data back to the
    original if needed.

    .. testsetup::
        import numpy as np
    .. code-block:: python

        gpu.memory(MemoryKind.Fast)(np.array([1, 2, 3])) # In fast memory on a GPU.
        gpu(0).memory(MemoryKind.Slow)(np.array([1, 2, 3])) # In slow memory on GPU #0.

    :allocation: Sometimes (if a the placement must change).
    """
    def __init__(
        self,
        device: Optional['Device'] = None,
        kind: Optional[MemoryKind] = None
    ):
        """
        :param device: The device which owns this memory (or None meaning any device).
        :type device: A `Device`, `Architecture`, or None.
        :param kind: The kind of memory (or None meaning any kind).
        :type kind: A `MemoryKind` or None.
        """
        self.device = device
        self.kind = kind

    @property
    @abstractmethod
    def np(self):
        """
        Return an object with an interface similar to the `numpy` module, but
        which operates on arrays in this memory.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, target):
        """
        Copy target into this memory.

        :param target: A data object (e.g., an array).
        :return: The copied data object in this memory. The returned object should have the same interface as the original.
        """
        raise NotImplementedError()

    def __repr__(self):
        return "<{} {} {}>".format(type(self).__name__, self.device, self.kind)

