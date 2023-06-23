from .indexing import IndexType

from .construction import recipe

import inspect

def get_module(obj, default=None):
    m = inspect.getmodule(type(obj)) or default
    if m:
        return m
    raise ModuleNotFoundError("No module for object of type %s" % type(obj))