Indexing, Slicing and Iterating
-------------------------------

A CrossPy array is a heterogeneous array that can be sliced with integers or Python
built-in slices. The result of slicing is also a CrossPy array.

>>> import crosspy as xp
>>> import numpy as np
>>> import cupy as cp
>>> a = xp.array([np.array([0, 2, 5]), cp.array([4, 1])], axis=0)
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 3),): array([0, 2, 5]),
       ((3, 5),): array([4, 1])}
>>> a[0]             # slicing with a single integer  # doctest: +NORMALIZE_WHITESPACE
0
>>> a[[1, 4, 3]]     # slicing with a list of integers  # doctest: +NORMALIZE_WHITESPACE
array {((0, 1),): array([2]),
       ((1, 3),): array([1, 4])}
>>> a[2:4]           # slicing with Python built-in slices  # doctest: +NORMALIZE_WHITESPACE
array {((0, 1),): array([5]),
       ((1, 2),): array([4])}

**Iterating** over a CrossPy array can be done with respect to either *partitions blocks* or *devices*.

`block_view()` returns the list of underlying partition blocks.

>>> a.block_view()
[array([0, 2, 5]), array([4, 1])]

You can also apply a function to each partition block, quivalent to `map(func, a.block_view())`.

>>> a.block_view(lambda x: x + 1)
[array([1, 3, 6]), array([5, 2])]
>>> a  # unchanged since the lambda is not inplace  # doctest: +NORMALIZE_WHITESPACE
array {((0, 3),): array([0, 2, 5]),
       ((3, 5),): array([4, 1])}
>>> a.block_view(lambda x: x.sort())  # apply inplace changes
[None, None]
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 3),): array([0, 2, 5]),
       ((3, 5),): array([1, 4])}

`device_view()` returns an iterable of lists where each list holds partition blocks 
on the same device.

>>> type(a.device_view())
<class 'generator'>
>>> tuple(a.device_view())  # doctest: +NORMALIZE_WHITESPACE
([array([0, 2, 5])], [array([1, 4])])
>>> for device_id, blocks_on_device_i in enumerate(a.device_view()):
...   print(device_id, blocks_on_device_i)
0 [array([0, 2, 5])]
1 [array([1, 4])]