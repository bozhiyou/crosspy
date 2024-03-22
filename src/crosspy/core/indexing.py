from typing import Union
from collections.abc import Iterable

from crosspy.utils.array import is_array, is_scalar_or_1d
from .utils import normalize_slice

BasicIndexType = Union[int, slice, Iterable[int]]  # non-recursive
IndexType = Union[BasicIndexType, tuple[BasicIndexType, ...]]  # recursive

class Index:
    @classmethod
    def parse(cls, index: IndexType, shape) -> tuple[IndexType]:
        """Check before set and get item"""
        # unify the form to tuple
        if not isinstance(index, tuple):
            index = (index, )

        def _check_each(d: int, index) -> IndexType:
            assert isinstance(index, (int, slice)) or (
                isinstance(index, list) and
                all(isinstance(i, (int, slice)) for i in index)
            ) or (
                is_array(index) and is_scalar_or_1d(index)
            ), NotImplementedError(
                "Only support 1-D indexing by integers, slices, and/or arrays"
            )

            if isinstance(index, slice):
                index = normalize_slice(index, shape[d])

            return index
        index = tuple(_check_each(d, i) for d, i in enumerate(index))

        # allow optional ellipsis [d0, d1, ...]
        if len(index) - len(shape) == 1 and index[-1] is Ellipsis:
            index = index[:-1]

        return index