Data Exchange
-------------

In scientific computing, a common data exchange pattern is ``b = a[indices]`` 
where `a` and `b` are arrays and `indices` is a collection of discrete integers.
This communication form is similar to the `alltoallv` MPI call. CrossPy supports
this data exchange pattern by providing the `alltoallv` function.

>>> import crosspy as xp
>>> import numpy as np
>>> import cupy as cp
>>> with cp.cuda.Device(0):
...   a0 = cp.array([1, 3, 5])
...   b0 = cp.array([22, 44])
>>> with cp.cuda.Device(1):
...   a1 = cp.array([2, 4])
...   b1 = cp.array([11, 33, 55])
>>> a = xp.array([a0, a1], axis=0)
>>> b = xp.array([b0, b1], axis=0)
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 3),): array([1, 3, 5]), ((3, 5),): array([2, 4])}
>>> b  # doctest: +NORMALIZE_WHITESPACE
array {((0, 2),): array([22, 44]), ((2, 5),): array([11, 33, 55])}
>>> xp.alltoallv(a, np.array([0, 3, 1, 4, 2]), b)  # semantics: b = a[[0, 3, 1, 4, 2]]
>>> b  # doctest: +NORMALIZE_WHITESPACE
array {((0, 2),): array([1, 2]), ((2, 5),): array([3, 4, 5])}

CrossPy also provides `assignment` for writeback `b[indices] = a`.

>>> xp.assignment(b, np.arange(len(b)), a, None)  # semantics: b[[0, 1, 2, 3, 4]] = a
>>> b  # doctest: +NORMALIZE_WHITESPACE
array {((0, 2),): array([1, 3]), ((2, 5),): array([5, 2, 4])}