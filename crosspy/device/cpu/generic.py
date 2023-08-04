import numpy

from contextlib import contextmanager
from typing import Dict

import os
import psutil

import crosspy
from crosspy import context
from crosspy.device import Architecture, Memory, Device, MemoryKind
from crosspy.utils import array

__all__ = ["cpu"]

import logging
logger = logging.getLogger(__name__)

_MEMORY_FRACTION = 15 / 16  # The fraction of total memory Parla should assume it can use.


def get_n_cores():
    return psutil.cpu_count(logical=False)


def get_total_memory():
    return psutil.virtual_memory().total


class _CPUMemory(Memory):
    @property
    def np(self):
        return numpy

    def __call__(self, target):
        if getattr(target, "device", None) is not None:
            logger.debug(
                "Moving data: %r => CPU", getattr(target, "device", None)
            )
        return array.asnumpy(target)


class NumPyCarrier:
    def pull(self, src, copy=False, out=None, **kwargs):
        if isinstance(src, crosspy.ndarray):
            if src.nparts != 1:
                raise ValueError("use tobuffer() to pull CrossPy array over %d devices" % src.nparts)
            src = src.item()
        if isinstance(src, numpy.ndarray):
            if out is not None:
                if copy or (out is not src):
                    out[...] = src
                return src
            if copy:
                return src.astype(src.dtype, copy=True)
            return src
        # TODO rewrite using register paradigm
        import cupy
        from crosspy.device.gpu.cuda.cupy_based import _pinned_memory_empty_like
        if isinstance(src, cupy.ndarray):
            stream = kwargs.pop('stream', None)
            src_ = _pinned_memory_empty_like(src) if out is None else out
            if stream is not None:
                return src.get(stream=stream, out=src_)
            with src.device:
                with cupy.cuda.Stream(non_blocking=True) as stream:
                    src.get(stream=stream, out=src_)
                stream.synchronize()
            return src_
        try:
            src_ = numpy.array(src) if copy else numpy.asanyarray(src)
            if out is not None:
                out[...] = src_
            return src_
        except BaseException:
            raise TypeError(f"{type(src)} is not supported yet")

    @property
    def module(self):
        return numpy

@context.register(numpy)
@contextmanager
def numpy_context(obj, **kwargs):
    yield NumPyCarrier()


class _CPUDevice(Device):
    def __init__(
        self, architecture: "Architecture", index, *args, n_cores, **kws
    ):
        super().__init__(architecture, index, *args, **kws)
        self.n_cores = n_cores or get_n_cores()
        self.available_memory = get_total_memory(
        ) * _MEMORY_FRACTION / get_n_cores() * self.n_cores

    @property
    def resources(self) -> Dict[str, float]:
        return dict(
            threads=self.n_cores,
            memory=self.available_memory,
            vcus=self.n_cores
        )

    # @property
    # def default_components(self) -> Collection[component.EnvironmentComponentDescriptor]:
    #     return [component.UnboundCPUComponent()]

    def memory(self, kind: MemoryKind = None):
        return _CPUMemory(self, kind)

    def get_array_module(self):
        import numpy
        return numpy

    def __repr__(self):
        return "<CPU {}>".format(self.index)
    
    def __enter__(self):
        return NumPyCarrier()


class _GenericCPUArchitecture(Architecture):
    def __init__(self, name, id):
        super().__init__(name, id)
        self.n_cores = get_n_cores()


class _CPUCoresArchitecture(_GenericCPUArchitecture):
    """
    A CPU architecture that treats each CPU core as a Parla device.
    Each device will have one VCU.

    WARNING: This architecture configures OpenMP and MKL to execute without any parallelism.
    """

    n_cores: int
    """
    The number of cores for which this process has affinity and are exposed as devices.
    """
    def __init__(self, name, id):
        super().__init__(name, id)
        self._devices = [self(i) for i in range(self.n_cores)]
        logger.warning(
            "CPU 'cores mode' enabled. "
            "Do not use parallel kernels in this mode (it will cause massive over subscription of the CPU). "
            "Setting OMP_NUM_THREADS=1 and MKL_THREADING_LAYER=SEQUENTIAL to avoid implicit parallelism."
        )
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

    @property
    def devices(self):
        return self._devices

    def __call__(self, id, *args, **kwds) -> _CPUDevice:
        return _CPUDevice(self, id, *args, **kwds, n_cores=1)


class _CPUWholeArchitecture(_GenericCPUArchitecture):
    """
    A CPU architecture that treats the entire CPU as a single Parla device.
    That device will have one VCU per core.
    """

    n_cores: int
    """
    The number of cores for which this process has affinity and are exposed as VCUs.
    """
    def __init__(self, name, id):
        super().__init__(name, id)
        self._device = self(0)

    @property
    def devices(self):
        return [self._device]

    def __call__(self, id=0, *args, **kwds) -> _CPUDevice:
        assert id == 0, "Whole CPU architecture only supports a single CPU device."
        return _CPUDevice(self, id, *args, **kwds, n_cores=None)


ENV_CPU_ARCH = os.environ.get("CROSSPY_CPU_ARCHITECTURE",
                  "whole").lower()
if ENV_CPU_ARCH not in ("whole", "cores"):
    logger.warning("CROSSPY_CPU_ARCHITECTURE only supports cores or whole.")

cpu = _CPUCoresArchitecture(
    "CPU Cores", "cpu"
) if ENV_CPU_ARCH == "cores" else _CPUWholeArchitecture("Whole CPU", "cpu")
cpu.__doc__ = """The `~parla.device.Architecture` for CPUs.

>>> cpu()
"""
