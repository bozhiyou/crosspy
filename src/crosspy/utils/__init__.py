from .profile import Timer

from .indexing import IndexType

from .construction import recipe

import inspect
import sys

def is_morphous(obj):
    if hasattr(obj, 'shape'):
        return True
    if isinstance(obj, (int, float, bool)):
        return True  # scalars
    # TODO other ways to measure shape
    return False

def get_module(obj, root=True, default=None):
    t = type(obj)
    if root and hasattr(t, '__module__'):
        t = sys.modules.get(t.__module__.split('.')[0])
    m = inspect.getmodule(t) or default
    if m:
        return m
    raise ModuleNotFoundError("No module for object of type %s" % type(obj))

def get_module_root(m):
    assert inspect.ismodule(m), ValueError("Try to get root of non-module object", type(m), m)
    return sys.modules.get(m.__name__.split('.')[0])

def get_length(obj):
    try:
        return len(obj)
    except AttributeError as e:
        if obj is None:
            return 0
        if isinstance(obj, (int, float, bool)):
            return 1
        if isinstance(obj, slice):
            # TODO relax this constraint
            assert obj.stop is not None and obj.start is not None, ValueError("Unknown length of slice", obj)
            # TODO negative step
            return obj.stop - obj.start
        raise e

def tuplize(obj):
    if isinstance(obj, tuple):
        return obj
    try:
        return (*obj,)
    except TypeError:
        return (obj,)

def allsame(attr=None):
    if attr is None:
        def allsameobj(objs):
            return all([x == objs[0] for x in objs[1:]])
        return allsameobj
    def allsameattr(objs):
        return all([attr(x) == attr(objs[0]) for x in objs[1:]])
    return allsameattr

allsamelen = allsame(len)