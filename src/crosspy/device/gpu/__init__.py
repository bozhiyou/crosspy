from .cuda import gpu, GPUDevice

__all__ = ['gpu', 'GPUDevice']

from crosspy import device

device.register_architecture("gpu")(gpu)
