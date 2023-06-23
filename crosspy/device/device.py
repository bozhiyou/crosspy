from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Dict, Iterable
from abc import ABCMeta, abstractmethod

from .memory import Memory, MemoryKind

class Device(nullcontext, metaclass=ABCMeta):
    """
    An instance of `Device` represents a compute device and its associated memory.
    Every device can directly access its own memory, but may be able to directly or indirectly access other devices memories.
    Depending on the system configuration, potential devices include one CPU core or a whole GPU.

    As devices are logical, the runtime may choose to implement two devices using the same hardware.
    """
    architecture: "Architecture"
    index: Optional[int]

    @lru_cache(maxsize=None)
    def __new__(cls, *args, **kwargs):
        return super(Device, cls).__new__(cls)

    def __init__(self, architecture: "Architecture", index, *args, **kwds):
        """
        Construct a new Device with a specific architecture.
        """
        super().__init__()
        self.architecture = architecture
        self.index = index  # index of gpu
        self.args = args
        self.kwds = kwds

    @property
    @abstractmethod
    def resources(self) -> Dict[str, float]:
        raise NotImplementedError()

    def memory(self, kind: Optional[MemoryKind] = None):
        return Memory(self, kind)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, type(self)) and \
               self.architecture == o.architecture and \
               self.index == o.index and \
               self.args == o.args and \
               self.kwds == o.kwds

    def __hash__(self):
        return hash(self.architecture) + hash(self.index) * 37

    @abstractmethod
    def get_array_module(self):
        raise NotImplementedError()

class Architecture(metaclass=ABCMeta):
    """
    An Architecture instance represents a range of devices that can be used via the same API and can run the same code.
    For example, an architecture could be "host" (representing the CPUs on the system), or "CUDA" (representing CUDA supporting GPUs).
    """
    def __init__(self, name, id):
        """
        Create a new Architecture with a name and the ID which the runtime will use to identify it.
        """
        self.name = name
        self.id = id

    def __call__(self, *args, **kwds):
        """
        Create a device with this architecture.
        The arguments can specify which physical device you are requesting, but the runtime may override you.

        >>> gpu(0)
        """
        return Device(self, *args, **kwds)

    def __getitem__(self, ind):
        if isinstance(ind, Iterable):
            return [self(i) for i in ind]
        else:
            return self(ind)

    @property
    @abstractmethod
    def devices(self):
        """
        :return: all `devices<Device>` with this architecture in the system.
        """
        pass

    def __parla_placement__(self):
        return self.devices

    # def memory(self, kind: MemoryKind = None):
    #     return Memory(self, kind)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, type(self)) and \
               self.id == o.id and self.name == o.name

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return type(self).__name__
