from collections.abc import Sequence
from numbers import Integral

from warnings import warn

import numpy
import crosspy as xp

from crosspy.core.creation import shape_factory
from crosspy.device import Device
from crosspy.context import context
from crosspy.utils.array import is_array


def seed(s):
    # TODO manage libraries
    numpy.random.seed(s)
    from crosspy import cupy
    cupy.random.seed(s)

def rand(*shape, device=None, mode='raise', **kwargs):
    return shape_factory('random.rand', *shape, device=device, mode=mode, **kwargs)
