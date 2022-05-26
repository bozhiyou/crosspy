CrossPy Quickstart
==================

Designed as a superset of the well-known `NumPy`_ and `CuPy`_ packages, **CrossPy**’s 
main object is the **heterogeneous** multidimensional array. It is *used* as 
a table of numbers, indexed by a tuple of non-negative integers, the same 
abstraction as NumPy/CuPy -- under the hood, the table can be composed of arrays
of different types. Following NumPy conventions, dimensions are called *axes*, 
(currently at most one) of which could be heterogeneous.

.. include:: qs/array_creation.rst
.. include:: qs/indexing_slicing.rst
.. _partitioning:
.. include:: qs/partitioning.rst
.. _data_exchange:
.. include:: qs/communication.rst



.. _NumPy: https://numpy.org/doc/stable/index.html
.. _CuPy: https://docs.cupy.dev/en/stable/index.html