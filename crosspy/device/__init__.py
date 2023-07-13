"""
Parla provides an abstract model of compute devices and memories.
The model is used to describe the placement restrictions for computations and storage.
"""
from typing import List, Mapping, Optional

from .device import Device, Architecture
from .memory import MemoryKind, Memory

import logging
logger = logging.getLogger(__name__)

__all__ = [
    "cpu",
    "get_device",
    "MemoryKind", "Memory", "Device", "Architecture", "get_all_devices",
    "get_all_architectures", "get_architecture", "kib", "Mib", "Gib",
    "register_architecture"
]

kib = 1024
Mib = kib * 1024
Gib = Mib * 1024

_device_getter = {}  # array type -> getter func

_architectures: Mapping[str, Architecture] = {}
_architectures_list: List[Architecture] = []


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


def get_architecture(name):
    try:
        return _architectures[name]
    except KeyError:
        raise ValueError("Non-existent architecture: " + name)


def register_architecture(name: str):
    """decorator"""
    if name in _architectures:
        raise ValueError("Architecture {} is already registered".format(name))

    def register(impl):
        _architectures[name] = impl
        _architectures_list.append(impl)

    return register


def get_all_devices() -> List[Device]:
    """
    :return: A list of all Devices in all Architectures.
    """
    return [d for arch in _architectures_list for d in arch.devices]


def get_all_architectures() -> List[Architecture]:
    """
    :return: A list of all Architectures.
    """
    return list(_architectures_list)

from .cpu import cpu, _CPUDevice
DEFAULT_DEVICE = cpu(0)  # all objects default to be on cpu


def get_device(obj, *, reg_by_id: Optional[dict]=None):
    if type(obj) in _device_getter:
        return _device_getter[type(obj)](obj)
    if hasattr(obj, 'device'):
        return getattr(obj, 'device')
    if reg_by_id and id(obj) in reg_by_id:
        return reg_by_id[id(obj)]
    return DEFAULT_DEVICE

try:
    from .gpu import gpu, _GPUDevice
except ImportError:
    pass