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
