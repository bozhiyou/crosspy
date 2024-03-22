"""
MSI protocol
Async IO
"""
from collections import defaultdict

import numpy
from crosspy.context import context
from crosspy.device import get_device
from crosspy.utils.array import get_array_module
from crosspy.transfer.cache import fetch
from crosspy.utils.cupy import _pinned_memory_empty


class Vocabulary(list):
    def index(self, word):
        try:
            return super().index(word)
        except ValueError:
            new_id = len(self)
            self.append(word)
            return new_id

class _Metadata:
    def __init__(self, nblocks, *, alloc=numpy.empty):
        # (len(blk0), len(blk1), ...)
        self.block_offset = alloc(nblocks, dtype=numpy.int64)
        # device id for each block
        self.device_idx = alloc(nblocks, dtype=numpy.int64)
        # device-local = global index - device offset
        self.device_offset = alloc(nblocks, dtype=numpy.int64)
        """
        dev_id, dev_offset = device_idx[blk_id], device_offset[blk_id]
        if blk_id:
            device_array[dev_id][index - block_offset[blk_id - 1] + dev_offset]
        else:
            device_array[dev_id][index]
        """

    def __len__(self):
        return len(self.block_offset)

    def __iter__(self):
        """Implemented to support unpacking"""
        return iter((self.block_offset, self.device_idx, self.device_offset))
    

    def __eq__(self, __value: '_Metadata') -> bool:
        return self is __value or all(
            (a is b or numpy.array_equal(a, b) for a, b in zip(self, __value))
        )

    def _global_index_to_block_id(self, i, out=None, stream=None):
        """
        Computes which block the referred element resides in

        Return shape and device is same to input indices unless out is given
        """
        with context(i) as ctx:
            _py = ctx.module
            assert hasattr(_py, 'searchsorted'), TypeError("Indices can only be numpy.ndarray or cupy.ndarray, not %s" % type(i))
            block_offset_ = ctx.pull(self.block_offset, copy=False, stream=stream)
            blk_id = _py.searchsorted(block_offset_, i, side='right')
        if out is not None:
            out[...] = blk_id
        return blk_id


class X1D:
    """Unidimensional heterogeneous array."""
    # TODO __slots__

    def __init__(self, objs, axis) -> None:
        device_vocab = Vocabulary()
        self._metadata = _Metadata(len(objs), alloc=_pinned_memory_empty)

        # delay actuall concatenation
        block_idx = defaultdict(list)
        per_device_size = defaultdict(int)
        global_size = 0
        for i, block in enumerate(objs):
            did = device_vocab.index(get_device(block))
            self._metadata.device_idx[i] = did
            size = block.shape[axis]
            self._metadata.device_offset[i] = global_size - per_device_size[did]
            per_device_size[did] += size
            self._metadata.block_offset[i] = global_size + size
            global_size = self._metadata.block_offset[i]
            block_idx[did].append(i)
        # concatenate once to avoid reallocation overhead
        # device id (int) -> grouped array
        self.device_array = {}
        for did, bids in block_idx.items():
            with device_vocab[did]:
                self.device_array[did] = get_array_module(objs[bids[0]]).concatenate([objs[i] for i in bids], axis=axis)

    @property
    def metadata(self):
        return self._metadata

    def __len__(self):
        return self._metadata.block_offset[-1]

    def get_block_index(self, i, stream=None):
        """
        Returned value is of the same size and on the same device as i.
        """
        with get_device(i) as here:
            local_block_offset = fetch(self._metadata.block_offset, here, stream)
        lib = get_array_module(i)
        blk_id = lib.searchsorted(local_block_offset, i, side='right')
        return blk_id
    
    def _get_device_index(self, blki, stream=None):
        """
        Input must be block id!
        """
        with get_device(blki) as here:
            local_device_index = fetch(self.device_idx, here, stream)
            return local_device_index[blki]
        
    def _get_local_index(self, i, blk_id, stream):
        with get_device(blk_id) as here:
            local_block_offset = fetch(self._metadata.block_offset, here, stream)
            local_device_offset = fetch(self._metadata.device_offset, here, stream)
            mask = (blk_id > 0)
            i[mask] = i[mask] - local_device_offset[blk_id[mask]]
            return i

    def __getitem__(self, index):
        assert isinstance(index, int), "experimental for ints only"
        i = index
        blk_id = self.get_block_index(i)
        dev_id, dev_offset = self._metadata.device_idx[blk_id], self._metadata.device_offset[blk_id]
        return self.device_array[dev_id][int(i - dev_offset)]
        