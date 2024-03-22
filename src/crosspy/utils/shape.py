from typing import Optional, Tuple, Union, Iterable, Sequence
from crosspy.utils import allsame, allsamelen

import logging
logger = logging.getLogger(__name__)

ShapeType=Union[Tuple[()], Tuple[int, ...]]


def compose_shape(shapes: Sequence[ShapeType], axis) -> ShapeType:
    """
    :param shapes: shapes of each subarray
    :return: shape of the aggregated array
    """
    logger.debug(shapes)
    if len(shapes) == 1:
        final_shape = shapes[0]
        return (1,) + final_shape if axis is None else final_shape

    # TODO: "same" is strict; relax to compatable: allow trivial dimension of 1 to align
    assert allsamelen(shapes), ValueError("Array dimensions mismatch")

    if axis is None:  # stack
        assert allsame(shapes), "cannot stack different shapes"
        return (len(shapes),) + shapes[0]

    assert axis == 0, NotImplementedError("only support concat along axis 0")
    assert all(axis < len(s) for s in shapes)
    # reduce shapes
    logger.debug("Concat over dim ", axis)
    shape = list(shapes[0])
    shape[axis] = sum([s[axis] for s in shapes])
    final_shape: ShapeType = tuple(shape)
    return final_shape
