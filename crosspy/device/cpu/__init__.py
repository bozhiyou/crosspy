# FIXME: This load of numpy causes problems if numpy is multiloaded. So this breaks using VECs with parla tasks.
#  Loading numpy locally works for some things, but not for the array._register_array_type call.
import numpy

from crosspy import device
from crosspy.utils.array import ArrayType, register_array_type
from .generic import cpu, _CPUDevice, _CPUMemory

__all__ = ["cpu"]


device.register_architecture("cpu")(cpu)

@device.of(numpy.ndarray)
def default_numpy_device(np_arr):
    assert(isinstance(np_arr, (numpy.ndarray)))
    return cpu(0)


class _NumPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        return isinstance(b, numpy.ndarray)

    def get_memory(self, a):
        # TODO: This is an issue since there is no way to attach allocations of CPU arrays to specific CPU devices.
        return _CPUMemory(cpu(0))

    def get_array_module(self, a):
        return numpy

register_array_type(numpy.ndarray)(_NumPyArrayType())
