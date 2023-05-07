# crosspy
A Python library that provides heterogeneous array interface with interoperability to NumPy and CuPy.

Temporary documentation at https://users.oden.utexas.edu/~byou/crosspy/

** ND-array: single node: multigpu, (also multi-numa, and hybrid)
+ support copy from-to cupy, numpy arrays with slicing
  x = parray
  y = cp...
  y[in] = x[in1]
+ support for different groups of devices
+ resizing an array
  repartitioning, data redistribution
+ arbitrary slicing: irregular scatter gather (like for a sort or an alltoallv)
  x = parray(n,2)
  y = parray(n,4)
  in = parray(n,3) # random permutation
  x = y(in)
+ coloring: overlapping partitions / noncontiguous partition
  color, read_overlap, write_overlap = user_create_four_colors(n)
  x = parray(n, (color,r_o, w_o))
+ separate from Parla? how to interface with Parla (copies decided by scheduler)
+ support for abstract ops
  primitives: broadcast, reduce, scatter, gather, all2all, all2allv
  derivatives: allreduce, allscatter, allgather, scan, sort
  advanced: atomic writes
** competition/ alternatives
+ dask, legate,
+ https://github.com/enthought/distarray
+ nonpython
  GASnet, Legion, Regent


numpy >= 1.15.0 to support stable sort kind

comparison:
same in values
same in structure
same in devices