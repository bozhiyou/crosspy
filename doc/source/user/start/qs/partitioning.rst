Heterogeneous Partitioning
--------------------------

CrossPy provides notations for heterogeneous devices. Specifically, one can use
``crosspy.cpu`` and/or ``crosspy.gpu`` to refer to `all` CPU and/or GPU devices

>>> import crosspy as xp
>>> from crosspy import cpu, gpu

To refer to a specific CPU/GPU device, pass an integer as the device ID, `e.g.`
``gpu(0)``. With the device notations, a CrossPy array can be created with
initial ``distribution``. Data will be equally distributed to the specified devices
accordingly.

>>> import numpy as np
>>> a = xp.array(np.arange(6), distribution=[cpu(0), gpu(0), gpu(1)])
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 1), (0, 2)): array([0, 1]),
       ((1, 2), (0, 2)): array([2, 3]),
       ((2, 3), (0, 2)): array([4, 5])}

Note that if the parameter ``axis`` is not specified, there will be an additional
dimension for partitioning. To keep the shape of the original object, set ``axis``
as the dimension along which the partition is expected to perform.

>>> a = xp.array(np.arange(6), distribution=[cpu(0), gpu(0), gpu(1)], axis=0)
>>> a  # doctest: +NORMALIZE_WHITESPACE
array {((0, 2),): array([0, 1]),
       ((2, 4),): array([2, 3]),
       ((4, 6),): array([4, 5])}
>>> a.device_map  # doctest: +NORMALIZE_WHITESPACE
{((0, 2),): 'cpu',
 ((2, 4),): <CUDA Device 0>,
 ((4, 6),): <CUDA Device 1>}

More flexible partitioning scheme (aka "coloring") can be expressed with the help of an auxiliary
``PartitionScheme``, which is conceptually a mask over some shape indicating the
device of each element. The following example creates a partitioning scheme for
any 1-D array of size 6.

>>> from crosspy import PartitionScheme
>>> partition = PartitionScheme(6, default_device=cpu(0))

Note that one can specify ``default_device`` for the schema so that all elements
are by default mapped to this device. If ``default_device`` is not specified or
`None`, the mapping is uninitialized - be careful! In this case, the scheme is
invalid until all elements have their devices specified.

To specify the coloring scheme, assign devices to corresponding parts.

.. >>> partition[4] = cpu(0)

>>> partition[0:2] = cpu(0)
>>> partition[2:6] = gpu(1)

.. >>> partition[[0, 5]] = gpu(1)

With the ``PartitionScheme`` object, a CrossPy array can be created accordingly
by passing the scheme as ``distribution``.

>>> a = xp.array(np.arange(6), distribution=partition, axis=0)
>>> a # doctest: +NORMALIZE_WHITESPACE
array {((0, 2),): array([0, 1]),
       ((2, 6),): array([2, 3, 4, 5])}
>>> a.device_map # doctest: +NORMALIZE_WHITESPACE
{((0, 2),): 'cpu',
 ((2, 6),): <CUDA Device 1>}

.. 
       note::
       ``PartitionScheme`` is implemented as a simple Python dictionary
       from parts (integer indices, slices, etc.) to devices. For example, the
       scheme above could be expressed as

       :code:`{(0, 4, 5): cpu(0), slice(1, 3): gpu(0), 3: gpu(1)}`

       Python slice objects are not hashable and thus cannot be dictionary keys.