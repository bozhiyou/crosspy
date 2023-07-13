"""
Integration with parla.environments
"""
import threading

from typing import Collection, List, Optional

from . import cuda

try:
    import cupy
    import cupy.cuda
except (ImportError, AttributeError):
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all(
        "sphinx" not in f.filename
        for f in inspect.getouterframes(inspect.currentframe())
    ):
        raise
    cupy = None

__all__ = ["GPUComponent", "MultiGPUComponent"]


class _GPUStacksLocal(threading.local):
    _stream_stack: List[cupy.cuda.Stream]
    _device_stack: List[cupy.cuda.Device]

    def __init__(self):
        super(_GPUStacksLocal, self).__init__()
        self._stream_stack = []
        self._device_stack = []

    def push_stream(self, stream):
        self._stream_stack.append(stream)

    def pop_stream(self) -> cupy.cuda.Stream:
        return self._stream_stack.pop()

    def push_device(self, dev):
        self._device_stack.append(dev)

    def pop_device(self) -> cupy.cuda.Device:
        return self._device_stack.pop()

    @property
    def stream(self):
        if self._stream_stack:
            return self._stream_stack[-1]
        else:
            return None

    @property
    def device(self):
        if self._device_stack:
            return self._device_stack[-1]
        else:
            return None


class GPUComponentInstance(EnvironmentComponentInstance):
    _stack: _GPUStacksLocal
    gpus: List["cuda._GPUDevice"]

    def __init__(self, descriptor: "GPUComponent", env: TaskEnvironment):
        super().__init__(descriptor)
        self.gpus = [
            d for d in env.placement if isinstance(d, "cuda._GPUDevice")
        ]
        assert len(self.gpus) == 1
        self.gpu = self.gpus[0]
        # Use a stack per thread per GPU component just in case.
        self._stack = _GPUStacksLocal()

    def _make_stream(self):
        with self.gpu.cupy_device:
            return cupy.cuda.Stream(null=False, non_blocking=True)

    def __enter__(self):
        _gpu_locals._gpus = self.gpus
        dev = self.gpu.cupy_device
        dev.__enter__()
        self._stack.push_device(dev)
        stream = self._make_stream()
        stream.__enter__()
        self._stack.push_stream(stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dev = self._stack.device
        stream = self._stack.stream
        try:
            stream.synchronize()
            stream.__exit__(exc_type, exc_val, exc_tb)
            _gpu_locals._gpus = None
            ret = dev.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._stack.pop_stream()
            self._stack.pop_device()
        return ret

    def initialize_thread(self) -> None:
        for gpu in self.gpus:
            # Trigger cuBLAS/etc. initialization for this GPU in this thread.
            with cupy.cuda.Device(gpu.index) as device:
                a = cupy.asarray([2.])
                cupy.cuda.get_current_stream().synchronize()
                with cupy.cuda.Stream(False, True) as stream:
                    cupy.asnumpy(cupy.sqrt(a))
                    device.cublas_handle
                    device.cusolver_handle
                    device.cusolver_sp_handle
                    device.cusparse_handle
                    stream.synchronize()
                    device.synchronize()


class GPUComponent(EnvironmentComponentDescriptor):
    """A single GPU CUDA component which configures the environment to use the specific GPU using a single
    non-blocking stream

    """
    def combine(self, other):
        assert isinstance(other, GPUComponent)
        return self

    def __call__(self, env: TaskEnvironment) -> GPUComponentInstance:
        return GPUComponentInstance(self, env)


class _GPULocals(threading.local):
    _gpus: Optional[Collection["cuda._GPUDevice"]]

    def __init__(self):
        super(_GPULocals, self).__init__()
        self._gpus = None

    @property
    def gpus(self):
        if self._gpus:
            return self._gpus
        else:
            raise RuntimeError("No GPUs configured for this context")


_gpu_locals = _GPULocals()

from ..device import Device


def get_gpus() -> Collection[Device]:
    return _gpu_locals.gpus


class MultiGPUComponentInstance(EnvironmentComponentInstance):
    gpus: List["cuda._GPUDevice"]

    def __init__(self, descriptor: "MultiGPUComponent", env: TaskEnvironment):
        super().__init__(descriptor)
        self.gpus = [
            d for d in env.placement if isinstance(d, "cuda._GPUDevice")
        ]
        assert self.gpus

    def __enter__(self):
        assert _gpu_locals._gpus is None
        _gpu_locals._gpus = self.gpus
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _gpu_locals._gpus == self.gpus
        _gpu_locals._gpus = None
        return False

    def initialize_thread(self) -> None:
        for gpu in self.gpus:
            # Trigger cuBLAS/etc. initialization for this GPU in this thread.
            with cupy.cuda.Device(gpu.index) as device:
                a = cupy.asarray([2.])
                cupy.cuda.get_current_stream().synchronize()
                with cupy.cuda.Stream(False, True) as stream:
                    cupy.asnumpy(cupy.sqrt(a))
                    device.cublas_handle
                    device.cusolver_handle
                    device.cusolver_sp_handle
                    device.cusparse_handle
                    stream.synchronize()
                    device.synchronize()


class MultiGPUComponent(EnvironmentComponentDescriptor):
    """A multi-GPU CUDA component which exposes the GPUs to the task via `get_gpus`.

    The task code is responsible for selecting and using the GPUs and any associated streams.
    """
    def combine(self, other):
        assert isinstance(other, MultiGPUComponent)
        return self

    def __call__(self, env: TaskEnvironment) -> MultiGPUComponentInstance:
        return MultiGPUComponentInstance(self, env)
