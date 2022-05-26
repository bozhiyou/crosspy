Array Creation
--------------

A CrossPy array can be created from:

- a NumPy/CuPy array;

.. - any object that can be used to create a NumPy/CuPy array;

- a composition of them.

You can create an CrossPy array using the `array` function:

>>> import crosspy as xp
>>> import numpy as np
>>> xp.array(np.arange(4))  # doctest: +NORMALIZE_WHITESPACE
array {((0, 4),): array([0, 1, 2, 3])}
>>> import cupy as cp
>>> xp.array(cp.zeros((2, 3)))  # doctest: +NORMALIZE_WHITESPACE
array {((0, 2), (0, 3)): array([[0., 0., 0.], [0., 0., 0.]])}

.. 
    >>> xp.array([1, 2, 3])  # doctest: +NORMALIZE_WHITESPACE
    array {((0, 3),): [1, 2, 3]}

.. >>> x_gpu = cp.array([[4, 5]])
.. >>> x_gpu
.. array([[4, 5]])
.. >>> x_gpu.device
.. <CUDA Device 0>

.. >>> x_cross = xp.array([x_cpu, x_gpu])
.. >>> x_cross
.. array {((0, 1), (0, 3)): array([[1, 2, 3]]), 
..        ((0, 1), (3, 5)): array([[4, 5]])}

A CrossPy array ``x_cross`` is printed as a dictionary, where:

- the key stands for a span, which is a tuple of size equal to the number of 
  dimensions and each element is a pair representing a left-closed-right-open
  interval. For example, ``((0, 2), (0, 3))`` stands for a 2-D span ranging
  `[0:2]` on the first dimension and `[0:3]` on the second. 

- the value is the data component that makes up the span.

>>> a = xp.array([np.arange(3), cp.arange(3)])
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 1), (0, 3)): array([0, 1, 2]),
       ((1, 2), (0, 3)): array([0, 1, 2])}
>>> a.shape
(2, 3)

In this example, the input object is a composition of a NumPy array and a CuPy 
array. Typically, you may want to "concatenate" the heterogeneous components 
along some axis (currently only supports axis 0). This can be specified with 
the `axis` parameter.

>>> a = xp.array([np.arange(3), cp.arange(3)], axis=0)
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 3),): array([0, 1, 2]), ((3, 6),): array([0, 1, 2])}
>>> a.shape
(6,)

..
    By default, CrossPy concatenates arrays along the only dimension where their sizes
    differ; if all dimensions share the same size, arrays are concatenated along the
    first dimension (or the dimension specified using
    the `dim` argument, see `crosspy.array`); Otherwise, arrays are not compatible
    and a `ValueError` is thrown.

..
    >>> x = xp.array([x_gpu, x_gpu])  # merge 1x2 and 1x2 on the first dimension
    >>> x # doctest: +NORMALIZE_WHITESPACE
    array {((0, 1), (0, 2)): array([[4, 5]]),
        ((1, 2), (0, 2)): array([[4, 5]])}
    >>> x.shape
    (2, 2)
    >>> x = xp.array([x_gpu, x_gpu], dim=1)  # merge 1x2 and 1x2 on the second dimension
    >>> x # doctest: +NORMALIZE_WHITESPACE
    array {((0, 1), (0, 2)): array([[4, 5]]),
        ((0, 1), (2, 4)): array([[4, 5]])}
    >>> x.shape
    (1, 4)

    >>> xp.array([x_cpu, x_cpu.T])  # merge 1x3 and 3x1
    Traceback (most recent call last):
        ...
    ValueError: Incompatible shapes with 2 different dims
    >>> xp.array([x_cpu, x_cpu.reshape(1, 1, 3)])  # merge 1x3 and 1x1x3
    Traceback (most recent call last):
        ...
    ValueError: Array dimensions mismatch