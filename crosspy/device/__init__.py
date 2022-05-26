"""
Parla provides an abstract model of compute devices and memories.
The model is used to describe the placement restrictions for computations and storage.
"""

from .device import MemoryKind, Memory, Device, Architecture, get_all_architectures, get_all_devices, get_architecture, register_architecture
# from .cpu import cpu
# from .gpu import gpu

import logging

__all__ = [
    # "cpu", "gpu",
    "MemoryKind", "Memory", "Device", "Architecture", "get_all_devices",
    "get_all_architectures", "get_architecture", "kib", "Mib", "Gib",
    "register_architecture"
]

logger = logging.getLogger(__name__)

kib = 1024
Mib = kib * 1024
Gib = Mib * 1024
