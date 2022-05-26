# FIXME: This load of numpy causes problems if numpy is multiloaded. So this breaks using VECs with parla tasks.
#  Loading numpy locally works for some things, but not for the array._register_array_type call.
import numpy

from ...array import ArrayType, register_array_type
from ..device import register_architecture
from .generic import cpu, _CPUMemory

__all__ = ["cpu"]


class _NumPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        return isinstance(b, numpy.ndarray)

    def get_memory(self, a):
        # TODO: This is an issue since there is no way to attach allocations of CPU arrays to specific CPU devices.
        return _CPUMemory(cpu(0))

    def get_array_module(self, a):
        return numpy


register_architecture("cpu")(cpu)
register_array_type(numpy.ndarray)(_NumPyArrayType())
