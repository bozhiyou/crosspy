from typing import Union
from collections.abc import Iterable

from crosspy.utils.array import is_array

BasicIndexType = Union[int, slice, Iterable[int]]  # non-recursive
IndexType = Union[BasicIndexType, tuple[BasicIndexType, ...]]  # recursive

def is_empty_slice(slice_: slice):
    """Assume normalized slice"""
    return slice_.start == slice_.stop or (
        slice_.start > slice_.stop and slice_.step) or (
        slice_.start < slice_.stop and slice_.step < 0)

def shape_of_slice(normalized_slice_: slice):
    """Assume normalized slice"""
    return (normalized_slice_.stop - normalized_slice_.start) // normalized_slice_.step

def normalize_slice(slice_: slice, len_: int):
    def _wrap_to_positive(i):
        return i and int(i + len_ if i < 0 else i)  # convert numpy int to python int; TODO step?
    return slice(_wrap_to_positive(slice_.start or 0), _wrap_to_positive(slice_.stop if slice_.stop is not None else len_), slice_.step if slice_.step is not None else 1)

def _check_indexing(index: IndexType, shape) -> tuple[IndexType]:
    """Check before set and get item"""
    # unify the form to tuple
    if not isinstance(index, tuple):
        index = (index, )

    def _check_each(d: int, index) -> IndexType:
        assert isinstance(index, (int, slice)) or (
            isinstance(index, list) and
            all(isinstance(i, (int, slice)) for i in index)
        ) or (
            is_array(index) and getattr(
                index, 'ndim', len(getattr(index, 'shape', (len(index), )))
            ) == 1
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
