import cupy
import numpy

mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

# Create an array on CPU.
# NumPy allocates 400 bytes in CPU (not managed by CuPy memory pool).
a_cpu = numpy.ndarray(200, dtype=numpy.float32)
print(a_cpu.nbytes)                      # 400

# You can access statistics of these memory pools.
print(mempool.used_bytes())              # 0
print(mempool.total_bytes())             # 0
print(pinned_mempool.n_free_blocks())    # 0

# Transfer the array from CPU to GPU.
# This allocates 400 bytes from the device memory pool, and another 400
# bytes from the pinned memory pool.  The allocated pinned memory will be
# released just after the transfer is complete.  Note that the actual
# allocation size may be rounded to larger value than the requested size
# for performance.
a = cupy.array(a_cpu)
print(a.nbytes)                          # 400
print(mempool.used_bytes())              # 512
print(mempool.total_bytes())             # 512
print(pinned_mempool.n_free_blocks())    # 1

print("""
# When the array goes out of scope, the allocated device memory is released
# and kept in the pool for future reuse.
""")
b = a[:1]  # view
b = cupy.array(a[:1])  # copy
a = None  # (or `del a`)
print(mempool.used_bytes())              # 0
print(mempool.total_bytes())             # 512
print(pinned_mempool.n_free_blocks())    # 1

print("""
# You can clear the memory pool by calling `free_all_blocks`.
""")
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()
print(mempool.used_bytes())              # 0
print(mempool.total_bytes())             # 0
print(pinned_mempool.n_free_blocks())    # 0