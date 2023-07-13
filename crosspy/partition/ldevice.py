"""
Logical device collections provide a way to map logical devices (and their associates partitions of the data) to
physical devices for execution.
The classes in this module provide tools for mapping 1-d and 2-d arrangements of logical devices onto physical devices.

i is numerical range
For regular placement:
    token is corresponding data
    memory is movement callable
For arbitrary coloring:
    token is (device, index) pair
    memory is retriving from device_to_data dict
"""

from typing import List, Optional, Tuple, Collection, Mapping, Union, Any
from numbers import Integral

from warnings import warn

import functools
import inspect
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from functools import reduce
from math import floor, ceil

from crosspy.context import context
from crosspy.utils.array import is_array

from ..core.ndarray import IndexType
from ..device import Device, Memory, MemoryKind
from .placement import PlacementSource, get_placement_for_any


def _first_i(index) -> int:
    """Return first integer of index"""
    if isinstance(index, slice):
        return index.start or 0
    try:
        return int(index)
    except:
        return _first_i(index[0])

# slice object is not hashable
class PartitionScheme:
    """
    1D partitioning schema. Dense implementation.
    """
    NO_DEVICE = -1
    _device_vocab = []

    def __init__(self, target_shape):
        self._shape = target_shape
        self._devid_to_index = defaultdict(list)  # device id -> []
        self._keys = {}

    @property
    def shape(self):
        return self._shape

    @property
    def n_ldevices(self) -> int:
        return len(self._keys)

    @property
    def devices(self):
        return (self.get_device(did) for did in self._devid_to_index.keys())

    def get_key(self, i):
        return self._keys[sorted(self._keys.keys())[i]]

    def _get_device_id(self, device):
        try:
            return self._device_vocab.index(device)
        except ValueError:
            id = len(self._device_vocab)
            self._device_vocab.append(device)
            return id

    def get_device(self, device_id):
        return self._device_vocab[device_id]

    def __len__(self) -> int:
        return self.shape[0]

    def __setitem__(self, index: IndexType, device):
        device_id = self._get_device_id(device)
        data_key = (device_id, len(self._devid_to_index[device_id]))
        self._devid_to_index[device_id].append(index)
        part_order = _first_i(index)
        if part_order in self._keys:
            warn(RuntimeWarning("Overwritten device mapping at index %d" % part_order))
        self._keys[part_order] = data_key

    def __getitem__(self, index: IndexType):
        # for device_id, indices in self._scheme.items():
        #     if index in indices:
        #         return self._get_device(device_id)
        raise NotImplementedError("Cannot find the device for the given index")
    
    def get_memory(self, device_id, local_id):
        return self._devid_to_index[device_id][local_id]

    def __repr__(self) -> str:
        return repr(self._devid_to_index)


class LDeviceSequenceArbitrary:
    """
    A 1-d collection of logical devices which are assigned to physical devices in contiguous blocks.
    """
    def __init__(self, scheme: PartitionScheme):
        # self._inds, self._devs = scheme._indices, scheme._devices
        # assert len(self._inds) == len(self._devs)
        # n_ldevices = len(self._inds)
        # placement = self._devs
        # super().__init__(scheme.n_ldevices)
        # TODO scheme sanity check
        self._scheme = scheme
        self.n_ldevices = scheme.n_ldevices

    def __repr__(self):
        return "{}({})<{}>".format(
            type(self).__name__, self.n_ldevices, len(self.devices)
        )
    
    def memory(self, data, kind: Optional[MemoryKind] = None):
        # self._scheme._keys = sorted(self._scheme._keys.items(), key=lambda x: x[0])
        moved_data = {}
        for dev_id, indices in self._scheme._devid_to_index.items():
            device = self._scheme.get_device(dev_id)
            moved_data[dev_id] = [device.memory(kind)(data[s]) for s in indices]
        return lambda k: moved_data[k[0]][k[1]]

    def partition_tensor(self, data, overlap=0, wrapper=lambda x: x, memory_kind: Optional[MemoryKind] = None):
        """
        Partition a tensor along its first dimension, potentially with overlap.

        :param data: A numpy-compatible tensor.
        :param overlap: The number of elements by which partitions should overlap.
        :param memory_kind: The kind of memory in which to place the partitions.
        :return: A :class:`list` of the partition tensors returned by lambda copied to the appropriate device.
        """
        (n, *rest) = getattr(data, 'shape', (len(data),))
        """
        A function `(int) -> T` where `T` is any type that can be copied by
            `~parla.device.Memory` objects. This function is called for each logical device, passed as an index, to
            get the data for that logical device. The function may also take parameters named `device` and/or `memory`,
            in which case the device and/or memory associated with the logical device is passed along with the index.
        """
        obj = _wrapper_for_partition_function(
            lambda i: self._scheme.get_key(i)
        )
        map_key_to_memory = self.memory(data, kind=memory_kind)
        return [
            wrapper(
                map_key_to_memory(self._scheme.get_key(i))
            ) for i in range(self.n_ldevices)
        ]


def split(array_like, distribution, axis=None, mode=None):
    """
    :param distribution:
        - An iterable of devices. In this case, the array-like object can will be
        devided (almost) equally. Size of each partition is guaranteed to be
        either ceil(s) or floor(s) where s = total_size / len(devices)
        - An iterable of integers. In this case, the array-like object will be
        devided accordingly without changing its device (i.e. no communication)
        except when the iterable is an array and the partitions will be on the
        device of `distribution`. The numbers should be sizes of each partition
        unless the last digit equals the size of `array_like`, in which case
        the numbers are treated as size prefixes.
        - An iterable of (device, integer) pairs, which can typically be a zip
        or dict.items object
    :param mode:
        Specifies how out-of-bounds `distribution` will behave.
        - None - Skip all checks (default)
        - 'raise' - raise an error
        - 'wrap' - wrap around
        - 'clip' - clip to the range
        'clip' mode means that sizes that are larger than the size of `array_like`
        will be ignored.
    """
    # TODO support axis with swapaxis. reference numpy.split
    # TODO support single integer split
    # TODO flatten array when axis is None
    # TODO support singleton (non-list)
    # TODO pairs by crosspy array
    assert mode is None, NotImplementedError("TODO")
    # sizes by array
    if is_array(distribution):
        if len(distribution) == 0:
            return []
        assert distribution.dtype.kind == 'i', f"array of sizes must have integer dtype, not {distribution.dtype}"
        partitions = []
        with context(distribution) as ctx:
            array_like_ = ctx.pull(array_like)
            if distribution[len(distribution) - 1] == len(array_like_):
                stops = distribution
            else:
                _py = ctx.module
                stops = _py.cumsum(distribution)
            start = 0
            for stop in stops:
                partitions.append(array_like_[start:stop])
                start = stop
        return partitions

    try:
        nparts = len(distribution)
    except TypeError:
        distribution = (*distribution,)
        nparts = len(distribution)

    if nparts == 0:
        return []
    
    partitions = []
    try:
        peeked = distribution[0]
    except TypeError:
        distribution = (*distribution,)
        peeked = distribution[0]
    start = 0
    if isinstance(peeked, Device):
        size = array_like.shape[axis] if axis else len(array_like)
        subsize, nbalance = divmod(size, nparts)
        for i in range(nbalance):
            device = distribution[i]
            # assert isinstance(device, Device), "inconsistent distribution element type"
            stop = start + subsize + 1
            with device as ctx:
                partitions.append(ctx.pull(array_like[start:stop]))
            start = stop
        for i in range(nbalance, nparts):
            device = distribution[i]
            # assert isinstance(device, Device), "inconsistent distribution element type"
            stop = start + subsize
            with device as ctx:
                partitions.append(ctx.pull(array_like[start:stop]))
            start = stop
        # OLD IMPLEMENTATION
        # from .ldevice import LDeviceSequenceBlocked
        # Partitioner = LDeviceSequenceBlocked
        # mapper = Partitioner(len(distribution), placement=distribution)
        # arr_p = mapper.partition_tensor(array_like)
        # return arr_p
        return partitions
    if isinstance(peeked, Integral):
        if distribution[len(distribution) - 1] == len(array_like):
            for stop in distribution:
                # assert isinstance(size, int), "inconsistent distribution element type"
                partitions.append(array_like[start:stop])
                start = stop
        else:
            for size in distribution:
                # assert isinstance(size, int), "inconsistent distribution element type"
                stop = start + size
                partitions.append(array_like[start:stop])
                start = stop
        return partitions
    if isinstance(peeked, tuple):
        for device, size in distribution:
            assert isinstance(device, Device), TypeError("inconsistent distribution element type")
            assert isinstance(size, int), TypeError("inconsistent distribution element type")
            stop = start + size
            with device as ctx:
                partitions.append(ctx.pull(array_like[start:stop]))
            start = stop
        return partitions
    raise TypeError(f"Unrecognizable distribution with type {type(peeked)}")

def _factors(n: int) -> List[int]:
    for m in range(2, ceil(n**0.5) + 1):
        if n % m == 0:
            return [m] + _factors(n // m)
    return [n]


def _split_number(n: int) -> Tuple[int, int]:
    f = _factors(n)
    if len(f) == 1:
        f += [1]
    fa, fb = f[:len(f) // 2], f[len(f) // 2:]
    return reduce(int.__mul__, fa), reduce(int.__mul__, fb)


class LDeviceCollection(metaclass=ABCMeta):
    """
    A collection of logical devices mapped to physical devices.
    """
    def __init__(self, placement=None):
        """
        :param placement: The physical devices to use or None to use all physical devices.
        """
        devices = get_placement_for_any(placement)
        self._devices = tuple(devices)

    @property
    def devices(self):
        """
        The physical devices used by this collection.
        """
        return self._devices

    @property
    def n_devices(self) -> int:
        """len(self.devices)"""
        return len(self.devices)

    @property
    @abstractmethod
    def n_ldevices(self) -> int:
        pass

    def memory(self, *args: int, kind: Optional[MemoryKind] = None) -> Memory:
        """
        :param args: The indices of the logical device.
        :param kind: The kind of memory to return.
        :return: The physical memory associated with the specified logical device and memory kind.
        """
        return self.device(*args).memory(kind)

    @abstractmethod
    def device(self, *args: int) -> Device:
        """
        :param args: The indices of the logical device.
        :return: The physical device implementing the specified logical device.
        """
        pass

    @property
    @abstractmethod
    def assignments(self) -> Mapping[Tuple, Device]:
        """
        The mapping from valid indices to the associated physical devices.
        """
        pass


class LDeviceSequence(LDeviceCollection):
    """
    A 1-d collection of logical devices.
    """
    def __init__(self, n_ldevices, placement=None):
        """
        :param n_ldevices: The number of logical devices in this collection.
        :param placement: The physical devices to use or None to use all physical devices.
        """
        super().__init__(placement)
        self._n_ldevices = n_ldevices
        if self.n_ldevices < len(self.devices):
            warn(
                Warning(
                    "There are not enough partitions to cover the available devices in mapper: {} < {} (dropping {})"
                    .format(
                        self.n_ldevices, len(self.devices),
                        self._devices[:-self.n_ldevices]
                    )
                )
            )
            self._devices = self._devices[-self.n_ldevices:]

    @property
    def n_ldevices(self) -> int:
        return self._n_ldevices

    @abstractmethod
    def device(self, i: int) -> Device:
        """
        :param i: The index of the logical device.
        :return: The physical device.
        """
        pass

    def partition_tensor(self, data, overlap=0, wrapper=lambda x: x, memory_kind: Optional[MemoryKind] = None):
        """
        Partition a tensor along its first dimension, potentially with overlap.

        :param data: A numpy-compatible tensor.
        :param overlap: The number of elements by which partitions should overlap.
        :param memory_kind: The kind of memory in which to place the partitions.
        :return: A :class:`list` of the partition tensors returned by lambda copied to the appropriate device.
        """
        (n, *rest) = getattr(data, 'shape', (len(data),))
        """
        A function `(int) -> T` where `T` is any type that can be copied by
            `~parla.device.Memory` objects. This function is called for each logical device, passed as an index, to
            get the data for that logical device. The function may also take parameters named `device` and/or `memory`,
            in which case the device and/or memory associated with the logical device is passed along with the index.
        """
        obj = _wrapper_for_partition_function(
            lambda i: data[self.slice(i, n, overlap=overlap)]
        )
        return [
            wrapper(obj(
                i,
                map_key_to_memory=self.memory(i, kind=memory_kind),
                device=self.device(i)
            )) for i in range(self.n_ldevices)
        ]

    @abstractmethod
    def slice(self, i: int, n: int, step: int = 1, overlap: int = 0):
        """
        Get a slice object which will select the elements of sequence (of length `n`) which are in partition `i`.

        :param i: The index of the partition.
        :param n: The number of elements to slice (i.e., the length of the sequence this slice will be used on)
        :param step: The step of the slice *within* the partition. If this is non-zero, then the resulting slices
            (for `0 <= i < self.n_ldevices`) will only cover a portion of the values `0 <= j < n`.
        :param overlap: The number of element by which the slices should overlap
            (e.g., the overlap between `i=0` and `i=1`).
        :return: A `slice` object.
        """
        pass

    @property
    def assignments(self):
        return {(i, ): self.device(i) for i in range(self.n_ldevices)}


class LDeviceGrid(LDeviceCollection):
    """
    A 2-d collection of logical devices arranged in a grid.
    """
    n_ldevices_x: int
    n_ldevices_y: int

    def __init__(self, n_ldevices_x, n_ldevices_y, placement=None):
        """
        :param n_ldevices_x: The number of logical devices along the 1st dimension of this grid.
        :param n_ldevices_y: The number of logical devices along the 2nd dimension of this grid.
        :param placement: The physical devices to use or None to use all physical devices.
        """
        super().__init__(placement)
        self.n_ldevices_x = n_ldevices_x
        self.n_ldevices_y = n_ldevices_y
        if self.n_ldevices < len(self.devices):
            warn(
                Warning(
                    "There are not enough partitions to cover the available devices in mapper: {} < {} (dropping {})"
                    .format(
                        self.n_ldevices, len(self.devices),
                        self._devices[:-self.n_ldevices]
                    )
                )
            )
            self._devices = self._devices[:self.n_ldevices]

    @property
    def n_ldevices(self):
        return self.n_ldevices_y * self.n_ldevices_x

    @abstractmethod
    def device(self, i: int, j: int) -> Device:
        """
        :param i: The 1st index of the logical device.
        :param j: The 2nd index of the logical device.
        :return: The physical device.
        """
        pass

    def partition_tensor(self, data, overlap=0, memory_kind: MemoryKind = None):
        """
        Partition a tensor in its first two dimension, potentially with overlap.

        :param data: A numpy-compatible tensor.
        :param overlap: The number of elements by which partitions should overlap.
        :param memory_kind: The kind of memory in which to store the partitions.
        :return: A :class:`list` of lists of the partition tensors.
        """
        (n_x, n_y, *rest) = data.shape
        """
        A function `(int, int) -> T` where `T` is any type that can be copied by
            `~parla.device.Memory` objects. This function is called for each logical device, passed as indices, to
            get the data for that logical device. The function may also take parameters named `device` and/or `memory`,
            in which case the device and/or memory associated with the logical device is passed along with the indices.
        """
        obj = _wrapper_for_partition_function(
            lambda i, j: data[self.slice_x(i, n_x, overlap=overlap),
                              self.slice_y(j, n_y, overlap=overlap), ...]
        )
        return [[
            obj(
                i, j,
                map_key_to_memory=self.memory(i, j, kind=memory_kind),
                device=self.device(i, j)
            ) for j in range(self.n_ldevices_y)] for i in range(self.n_ldevices_x)]

    def slice_x(self, i: int, n: int, step: int = 1, *, overlap: int = 0):
        """
        :return: A slice along the 1st dimension of this grid
        :see: `~LDeviceSequence.slice`
        """
        return _partition_slice(
            i, n, self.n_ldevices_x, overlap=overlap, step=step
        )

    def slice_y(self, j: int, n: int, step: int = 1, *, overlap: int = 0):
        """
        :return: A slice along the 2st dimension of this grid
        :see: `~LDeviceSequence.slice`
        """
        return _partition_slice(
            j, n, self.n_ldevices_y, overlap=overlap, step=step
        )

    @property
    def assignments(self):
        return {
            (i, j): self.device(i, j)
            for i in range(self.n_ldevices_x) for j in range(self.n_ldevices_y)
        }


class LDeviceSequenceBlocked(LDeviceSequence):
    """
    A 1-d collection of logical devices which are assigned to physical devices in contiguous blocks.
    """
    def __init__(
        self,
        n_ldevices: int,
        placement: Union[Collection[PlacementSource], Any, None] = None
    ):
        super().__init__(n_ldevices, placement)
        self._divisor = self.n_ldevices / self.n_devices
        assert floor(self._divisor * self.n_devices) == self.n_ldevices

    def device(self, i):
        if not (0 <= i < self.n_ldevices):
            raise ValueError(i)
        return self.devices[floor(i / self._divisor)]

    def __repr__(self):
        return "{}({})<{}>".format(
            type(self).__name__, self.n_ldevices, len(self.devices)
        )

    def slice(self, i: int, n: int, step: int = 1, *, overlap: int = 0):
        return _partition_slice(
            i, n, self.n_ldevices, overlap=overlap, step=step
        )


class LDeviceGridBlocked(LDeviceGrid):
    """
    A 2-d collection of logical devices which are assigned to physical devices in contiguous blocks in both dimensions.
    """
    def __init__(
        self,
        n_ldevices_x: int,
        n_ldevices_y: int,
        placement: Union[Collection[PlacementSource], Any, None] = None
    ):
        super().__init__(n_ldevices_x, n_ldevices_y, placement)
        self._n, self._m = _split_number(self.n_devices)
        assert self._n * self._m == self.n_devices
        if self.n_ldevices_x < self._n or self.n_ldevices_y < self._m:
            warn(
                Warning(
                    "The logical device grid is not large enough to cover the physical device grid: ({}, {}) < ({}, {})"
                    .format(
                        self.n_ldevices_x, self.n_ldevices_y, self._n, self._m
                    )
                )
            )
        self._divisor_x = self.n_ldevices_x / self._n
        self._divisor_y = self.n_ldevices_y / self._m

    def device(self, i, j):
        if not (0 <= i < self.n_ldevices_x and 0 <= j < self.n_ldevices_y):
            raise ValueError((i, j))
        x = floor(i / self._divisor_x)
        y = floor(j / self._divisor_y)
        return self.devices[(x * self._m) + y]

    def __repr__(self):
        return "{}({}, {})<{}, {}>".format(
            type(self).__name__, self.n_ldevices_x, self.n_ldevices_y, self._n,
            self._m
        )


class LDeviceGridRaveled(LDeviceGrid):
    """
    A 2-d collection of logical devices which are assigned to physical devices as if `LDeviceSequenceBlocked` were
    applied to a "ravelled" version of the grid of logical devices.
    """
    def __init__(
        self,
        n_ldevices_x: int,
        n_ldevices_y: int,
        placement: Union[Collection[PlacementSource], Any, None] = None
    ):
        super().__init__(n_ldevices_x, n_ldevices_y, placement)
        self._divisor = self.n_ldevices / self.n_devices

    def device(self, i, j):
        if not (0 <= i < self.n_ldevices_x and 0 <= j < self.n_ldevices_y):
            raise ValueError((i, j))
        return self.devices[floor((i * self.n_ldevices_x + j) / self._divisor)]

    def __repr__(self):
        return "{}({}, {})<{}>".format(
            type(self).__name__, self.n_ldevices_x, self.n_ldevices_y,
            self.n_devices
        )


def _partition_slice(i, n, partitions, step=1, *, overlap=0):
    partition_size = n / partitions
    return slice(
        max(0,
            ceil(i * partition_size) - overlap),
        min(n,
            ceil((i + 1) * partition_size) + overlap), step
    )


def _wrapper_for_partition_function(get_key):
    """
    :param data: A function `(int, int) -> T` where `T` is any type that can be copied by
        `~parla.device.Memory` objects. This function is called for each logical device, passed as indices, to
        get the data for that logical device. The function may also take parameters named `device` and/or `memory`,
        in which case the device and/or memory associated with the logical device is passed along with the indices.
    :param memory_kind: The kind of memory in which to place the data.
    """
    # TODO: cache the checks for better performance
    arg_names, args_arg_name, kws_arg_name = inspect.getargs(get_key.__code__)
    has_memory = "memory" in arg_names
    has_device = "device" in arg_names
    if kws_arg_name is not None or (has_memory and has_device):
        _kwargs = lambda memory, device: {'memory': memory, 'device': device}
    elif has_memory:
        _kwargs = lambda memory, device: {'memory': memory}
    elif has_device:
        _kwargs = lambda memory, device: {'device': device}
    else:
        _kwargs = lambda memory, device: {}

    # noinspection PyUnusedLocal
    @functools.wraps(get_key)
    def wrapper(*args, map_key_to_memory, device):
        return map_key_to_memory(get_key(*args, **_kwargs(map_key_to_memory, device)))

    return wrapper
