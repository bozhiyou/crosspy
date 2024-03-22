from contextlib import contextmanager
from crosspy.utils.array import get_array_module

_context_manager = {}

def register(module_):
    def decorator(ctx):
        _context_manager[module_] = ctx
        return ctx
    return decorator

def get(obj):
    """Get environment manager"""
    module_ = get_array_module(obj)
    try:
        return _context_manager[module_]
    except KeyError:
        raise KeyError(f"No binding context manager for {module_}")

@contextmanager
def context(obj, **params):
    with get(obj)(obj, **params) as c:
        yield c