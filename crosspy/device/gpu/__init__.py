from crosspy import device
from .cuda import gpu, _GPUDevice
from crosspy.utils.array import ArrayType, register_array_type

try:
    import cupy
    import cupy.cuda
except (ImportError, AttributeError) as e:
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all(
        "sphinx" not in f.filename
        for f in inspect.getouterframes(inspect.currentframe())
    ):
        raise e

__all__ = ['gpu', 'cupy']

device.register_architecture("gpu")(gpu)


class _CuPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        # TODO: We should be able to do direct copies from numpy to cupy arrays, but it doesn't seem to be working.
        # return isinstance(b, (cupy.ndarray, numpy.ndarray))
        return isinstance(b, cupy.ndarray)

    def get_memory(self, a):
        return gpu(a.device.id).memory()

    def get_array_module(self, a):
        return cupy.get_array_module(a)

register_array_type(cupy.ndarray)(_CuPyArrayType())