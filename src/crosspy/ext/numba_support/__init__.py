"""
Two ways of implementing Numba support:
1. CUDA Array Interface (`__cuda_array_interface__`)
    CuPy implements this.
2. `typeof_impl` (not recommended)
"""
from crosspy.core import CrossPyArray
from numba.core.typing.typeof import typeof_impl
from numba.core.types import Buffer

class NumbaAdaptor(Buffer): pass

@typeof_impl.register
def _typeof_type(val: CrossPyArray, c):
    raise NotImplementedError()
    return NumbaAdaptor(val.dtype, val.ndim, 'C')