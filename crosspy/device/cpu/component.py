"""
Integration with parla.environments
"""
from parla.environments import EnvironmentComponentInstance, TaskEnvironment, EnvironmentComponentDescriptor
from . import generic


class UnboundCPUComponentInstance(EnvironmentComponentInstance):
    def __init__(self, descriptor, env):
        super().__init__(descriptor)
        cpus = [d for d in env.placement if isinstance(d, 'device._CPUDevice')]
        assert len(cpus) == 1
        self.cpus = cpus

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def initialize_thread(self) -> None:
        pass


class UnboundCPUComponent(EnvironmentComponentDescriptor):
    """A single CPU component that represents a "core" but isn't automatically bound to the given core.
    """
    def combine(self, other):
        assert isinstance(other, UnboundCPUComponent)
        assert self.cpus == other.cpus
        return self

    def __call__(self, env: TaskEnvironment) -> UnboundCPUComponentInstance:
        return UnboundCPUComponentInstance(self, env)