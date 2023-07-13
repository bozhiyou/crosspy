from types import ModuleType
import sys

def set_backend(backend):
    """
    Link backend modules under CrossPy
    """
    if isinstance(backend, ModuleType):
        backend = backend.__name__
    this_module = sys.modules[__name__]
    submodules = {}
    for imp, mod in sys.modules.items():
        if imp.startswith(f"{backend}."):
            rep = imp.replace(backend, __name__)
            if not rep in sys.modules:
                setattr(this_module, imp[len(backend) + 1:], mod)
                submodules[rep] = mod
    sys.modules.update(submodules)