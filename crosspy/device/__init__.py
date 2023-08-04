"""
Parla provides an abstract model of compute devices and memories.
The model is used to describe the placement restrictions for computations and storage.
"""
from typing import List, Mapping, Optional

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "cpu", "gpu",
    "Device", "GPUDevice",
    "get_all_devices", "get_device",
    "MemoryKind", "Memory",  "register_memory", "get_memory",
    "Architecture", "get_all_architectures", "get_architecture", "register_architecture",
    "kib", "Mib", "Gib"
]

kib = 1024
Mib = kib * 1024
Gib = Mib * 1024

_device_getter = {}  # array type -> getter func
_device_memory = {}

_architectures: Mapping[str, 'Architecture'] = {}
_architectures_list: List['Architecture'] = []


def of(*types):
    """
    (decorator) device getter for array type

    >>> @device.of(numpy.ndarray)
        def get_device(np_arr):
            return crosspy.cpu()
    """
    def register(getter):
        for t in types:
            _device_getter[t] = getter
        return getter

    return register


def get_device(obj, *, reg_by_id: Optional[dict] = None):
    if type(obj) in _device_getter:
        return _device_getter[type(obj)](obj)
    if hasattr(obj, 'device'):
        return getattr(obj, 'device')
    if reg_by_id and id(obj) in reg_by_id:
        return reg_by_id[id(obj)]
    return DEFAULT_DEVICE


def register_memory(dev):
    if dev in _device_memory:
        raise ValueError("Memory handler is already registered for %s" % repr(dev))

    def register(memctx):
        _device_memory[dev] = memctx

    return register


def get_memory(dev):
    try:
        return _device_memory[dev]
    except KeyError:
        raise ValueError("No memory handler for %s" % repr(dev))


def register_architecture(name: str):
    """decorator"""
    if name in _architectures:
        raise ValueError("Architecture %s is already registered" % repr(name))

    def register(impl):
        _architectures[name] = impl
        _architectures_list.append(impl)

    return register


def get_architecture(name):
    try:
        return _architectures[name]
    except KeyError:
        raise ValueError("Non-existent architecture: " + name)


def get_all_devices() -> List['Device']:
    """
    :return: A list of all Devices in all Architectures.
    """
    return [d for arch in _architectures_list for d in arch.devices]


def get_all_architectures() -> List['Architecture']:
    """
    :return: A list of all Architectures.
    """
    return list(_architectures_list)

from .meta import Device, Architecture
from .memory import MemoryKind, Memory

from .cpu import cpu, _CPUDevice
try:
    from .gpu import gpu, GPUDevice
except ImportError:
    pass

DEFAULT_DEVICE = cpu(0)  # all objects default to be on cpu
