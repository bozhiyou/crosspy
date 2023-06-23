try:
    from .cupy_based import _GPUDevice, gpu, _pin_memory, _pinned_memory_empty_like
except ImportError:
    pass
# from .cupy_based import gpu