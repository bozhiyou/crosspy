import crosspy as xp

from crosspy.context import context
from crosspy.utils import get_length

def gather(input: xp.ndarray, index, *, out: xp.ndarray=None):
    """
    :param index: (out_ndev, in_ndev)
    """
    assert out is None or len(index) == out.ndev
    for subscripts, dst in zip(index, out.device_array.values()):
        assert len(subscripts) <= input.ndev
        start = 0
        for subsc, src in zip(subscripts, input.device_array.values()):
            with context(src) as ctx:
                val = src[subsc]
            stop = start + get_length(val)
            with context(dst) as ctx:
                ctx.pull(val)
                dst[start:stop] = val
                start = stop
